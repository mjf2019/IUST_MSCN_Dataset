"""Conservative learned correction layer for fixed CDR-MLC routing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import fit_fixed_cdr
from learned_router_cdr_mlc import (
    LearnedRouterConfig, _expert_outputs, _gate_features, _oracle_route,
    _refit_experts_with_frozen_router,
)


@dataclass(frozen=True)
class SelectiveRouterConfig:
    window: int = 3
    expert_trees: int = 20
    correction_trees: int = 50
    max_depth: int = 8
    min_samples_leaf: int = 20
    expert_train_fraction: float = .60
    correction_train_fraction: float = .20
    threshold_candidates: tuple[float, ...] = (.50, .60, .70, .80, .90, .95, 1.01)
    min_expert_confidence_gain: float = .02
    random_state: int = 42

    def validate(self):
        if not 0 < self.expert_train_fraction < 1:
            raise ValueError("expert_train_fraction must be in (0,1)")
        if not 0 < self.correction_train_fraction < 1:
            raise ValueError("correction_train_fraction must be in (0,1)")
        if self.expert_train_fraction + self.correction_train_fraction >= 1:
            raise ValueError("a nonempty threshold-selection partition is required")
        if any(value < 0 for value in self.threshold_candidates):
            raise ValueError("threshold candidates must be nonnegative")
        return self


def three_way_split(frame, expert_fraction, correction_fraction):
    parts = {"expert": [], "correction": [], "threshold": []}
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        first = int(len(group) * expert_fraction)
        second = int(len(group) * (expert_fraction + correction_fraction))
        if not 0 < first < second < len(group):
            raise ValueError("capture too short for selective-router split")
        parts["expert"].append(group.iloc[:first].copy())
        parts["correction"].append(group.iloc[first:second].copy())
        parts["threshold"].append(group.iloc[second:].copy())
    return {key: pd.concat(value) for key, value in parts.items()}


def _fit_small_rf(x, y, config, seed_offset):
    unique = np.unique(y)
    if len(unique) == 1:
        return None, int(unique[0])
    model = RandomForestClassifier(
        n_estimators=config.correction_trees,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.random_state + seed_offset,
    ).fit(x, y)
    return model, None


def _class_one_probability(model, constant, x):
    if model is None:
        return np.full(len(x), float(constant == 1))
    result = np.zeros(len(x))
    classes = list(model.classes_)
    if 1 in classes:
        result = model.predict_proba(x)[:, classes.index(1)]
    return result


def _model_prediction(model, constant, x):
    if model is None:
        return np.full(len(x), constant, dtype=int)
    return model.predict(x).astype(int)


def _select_routes(minibatch_route, alternative, error_probability, probabilities,
                   threshold, min_gain):
    row = np.arange(len(minibatch_route))
    expert_confidence = probabilities.max(axis=2)
    current = expert_confidence[row, minibatch_route]
    proposed = expert_confidence[row, alternative]
    override = (
        (error_probability >= threshold)
        & (alternative != minibatch_route)
        & ((proposed - current) >= min_gain)
    )
    route = minibatch_route.copy()
    route[override] = alternative[override]
    return route, override


def fit_selective_router(source, config: SelectiveRouterConfig):
    config.validate()
    split = three_way_split(
        source, config.expert_train_fraction, config.correction_train_fraction
    )
    # Fit the final unsupervised geometry on all source records. No application
    # labels are used by StandardScaler or MiniBatchKMeans.
    full_model = fit_fixed_cdr(
        source, config.window, config.random_state, config.expert_trees
    )
    helper_config = LearnedRouterConfig(
        window=config.window, expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    preliminary = _refit_experts_with_frozen_router(
        full_model, split["expert"], helper_config
    )
    raw, _, distances, kroute, probabilities, predictions = _expert_outputs(
        preliminary, split["correction"]
    )
    truth = raw.traffic_label.to_numpy()
    row = np.arange(len(raw))
    error_target = (predictions[row, kroute] != truth).astype(int)
    oracle_target = _oracle_route(truth, probabilities, predictions)
    features = _gate_features(distances, kroute, probabilities)
    error_model, error_constant = _fit_small_rf(features, error_target, config, 11)
    alternative_mask = error_target.astype(bool)
    if alternative_mask.any():
        selector_model, selector_constant = _fit_small_rf(
            features[alternative_mask], oracle_target[alternative_mask], config, 23
        )
    else:
        selector_model, selector_constant = None, 0

    # Select an override threshold on a later source-only temporal partition.
    threshold_raw, _, threshold_dist, threshold_kroute, threshold_prob, threshold_pred = (
        _expert_outputs(preliminary, split["threshold"])
    )
    threshold_truth = threshold_raw.traffic_label.to_numpy()
    threshold_features = _gate_features(
        threshold_dist, threshold_kroute, threshold_prob
    )
    error_probability = _class_one_probability(
        error_model, error_constant, threshold_features
    )
    alternative = _model_prediction(
        selector_model, selector_constant, threshold_features
    )
    candidates = []
    threshold_row = np.arange(len(threshold_raw))
    for threshold in config.threshold_candidates:
        route, override = _select_routes(
            threshold_kroute, alternative, error_probability, threshold_prob,
            threshold, config.min_expert_confidence_gain,
        )
        prediction = threshold_pred[threshold_row, route]
        score = float(f1_score(
            threshold_truth, prediction, average="macro", zero_division=0
        ))
        candidates.append({
            "threshold": float(threshold), "macro_f1": score,
            "overrides": int(override.sum()),
            "override_fraction": float(override.mean()),
        })
    # Conservative tie-break: prefer the larger threshold/fewer overrides.
    selected = max(candidates, key=lambda item: (
        item["macro_f1"], item["threshold"], -item["overrides"]
    ))
    full_model.update({
        "selective_config": config,
        "error_model": error_model,
        "error_constant": error_constant,
        "selector_model": selector_model,
        "selector_constant": selector_constant,
        "selected_threshold": selected["threshold"],
        "threshold_trials": candidates,
        "correction_training_rows": len(raw),
        "threshold_selection_rows": len(threshold_raw),
        "error_rate_correction_partition": float(error_target.mean()),
        "oracle_target_counts": {
            str(cluster): int((oracle_target == cluster).sum()) for cluster in range(3)
        },
    })
    return full_model


def predict_all(model, frame):
    raw, _, distances, kroute, probabilities, predictions = _expert_outputs(
        model, frame
    )
    features = _gate_features(distances, kroute, probabilities)
    error_probability = _class_one_probability(
        model["error_model"], model["error_constant"], features
    )
    alternative = _model_prediction(
        model["selector_model"], model["selector_constant"], features
    )
    selective_route, override = _select_routes(
        kroute, alternative, error_probability, probabilities,
        model["selected_threshold"],
        model["selective_config"].min_expert_confidence_gain,
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    row = np.arange(len(raw))
    report = pd.DataFrame({
        "minibatch_kmeans_route": kroute,
        "alternative_route": alternative,
        "selective_route": selective_route,
        "oracle_route": oracle,
        "minibatch_kmeans_error_probability": error_probability,
        "route_overridden": override,
        "selective_matches_oracle": selective_route == oracle,
        "minibatch_kmeans_matches_oracle": kroute == oracle,
    }, index=raw.index)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, kroute],
        "CDR_MLC_selective_router": predictions[row, selective_route],
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": report,
    }
