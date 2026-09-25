"""Selective CDR-MLC router enriched with causal congestion descriptors.

The original three-feature MiniBatchKMeans and its expert bank are left unchanged.
Congestion descriptors are used only by the conservative correction layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig,
    congestion_feature_frame,
)
from learned_router_cdr_mlc import LearnedRouterConfig, _expert_outputs, _gate_features, _oracle_route, _refit_experts_with_frozen_router
from selective_router_cdr_mlc import (
    _class_one_probability, _fit_small_rf, _model_prediction, _select_routes,
    three_way_split,
)
from compare_clean_valid import fit_fixed_cdr


@dataclass(frozen=True)
class CongestionSelectiveConfig:
    window: int = 3
    congestion_window: int = 10
    congestion_features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    expert_trees: int = 20
    correction_trees: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 15
    expert_train_fraction: float = .60
    correction_train_fraction: float = .20
    threshold_candidates: tuple[float, ...] = (.50, .60, .70, .80, .90, .95, 1.01)
    min_expert_confidence_gain: float = .02
    random_state: int = 42

    def validate(self):
        if self.window < 1 or self.congestion_window < 3:
            raise ValueError("base window must be positive and congestion window >= 3")
        if len(set(self.congestion_features)) < 3:
            raise ValueError("at least three congestion features are required")
        if not 0 < self.expert_train_fraction < 1:
            raise ValueError("expert_train_fraction must be in (0,1)")
        if not 0 < self.correction_train_fraction < 1:
            raise ValueError("correction_train_fraction must be in (0,1)")
        if self.expert_train_fraction + self.correction_train_fraction >= 1:
            raise ValueError("a nonempty threshold-selection partition is required")
        return self


def _congestion_config(config):
    return CongestionRouterConfig(
        window=config.congestion_window,
        features=tuple(config.congestion_features),
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    ).validate()


def _enhanced_outputs(model, frame, congestion_config):
    raw, z, distances, kroute, probabilities, predictions = _expert_outputs(model, frame)
    congestion, _ = congestion_feature_frame(frame, congestion_config)
    rows = raw.index[raw.index.isin(congestion.index)]
    if len(rows) == 0:
        raise ValueError("no common base/congestion windows")
    positions = raw.index.get_indexer(rows)
    raw = raw.loc[rows]
    distances = distances[positions]
    kroute = kroute[positions]
    probabilities = probabilities[positions]
    predictions = predictions[positions]
    congestion_columns = [column for column in congestion if column.startswith("router_")]
    congestion_values = congestion.loc[rows, congestion_columns].to_numpy(dtype=float)
    base_features = _gate_features(distances, kroute, probabilities)
    return (
        raw, distances, kroute, probabilities, predictions,
        np.column_stack([base_features, congestion_values]), congestion_columns,
    )


def fit_congestion_selective_router(source, config: CongestionSelectiveConfig):
    config.validate()
    congestion_config = _congestion_config(config)
    split = three_way_split(
        source, config.expert_train_fraction, config.correction_train_fraction
    )
    # This is exactly the original fixed CDR-MLC geometry and expert bank.
    full_model = fit_fixed_cdr(
        source, config.window, config.random_state, config.expert_trees
    )
    helper = LearnedRouterConfig(
        window=config.window, expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    preliminary = _refit_experts_with_frozen_router(
        full_model, split["expert"], helper
    )

    raw, _, kroute, probabilities, predictions, features, congestion_columns = (
        _enhanced_outputs(preliminary, split["correction"], congestion_config)
    )
    truth = raw.traffic_label.to_numpy()
    row = np.arange(len(raw))
    error_target = (predictions[row, kroute] != truth).astype(int)
    oracle_target = _oracle_route(truth, probabilities, predictions)
    error_model, error_constant = _fit_small_rf(features, error_target, config, 101)
    incorrect = error_target.astype(bool)
    if incorrect.any():
        selector_model, selector_constant = _fit_small_rf(
            features[incorrect], oracle_target[incorrect], config, 203
        )
    else:
        selector_model, selector_constant = None, 0

    threshold_raw, _, threshold_kroute, threshold_prob, threshold_pred, threshold_features, _ = (
        _enhanced_outputs(preliminary, split["threshold"], congestion_config)
    )
    threshold_truth = threshold_raw.traffic_label.to_numpy()
    error_probability = _class_one_probability(
        error_model, error_constant, threshold_features
    )
    alternative = _model_prediction(
        selector_model, selector_constant, threshold_features
    )
    threshold_row = np.arange(len(threshold_raw))
    candidates = []
    for threshold in config.threshold_candidates:
        route, override = _select_routes(
            threshold_kroute, alternative, error_probability, threshold_prob,
            threshold, config.min_expert_confidence_gain,
        )
        prediction = threshold_pred[threshold_row, route]
        candidates.append({
            "threshold": float(threshold),
            "macro_f1": float(f1_score(
                threshold_truth, prediction, average="macro", zero_division=0
            )),
            "overrides": int(override.sum()),
            "override_fraction": float(override.mean()),
        })
    selected = max(candidates, key=lambda item: (
        item["macro_f1"], item["threshold"], -item["overrides"]
    ))
    full_model.update({
        "congestion_selective_config": config,
        "congestion_config": congestion_config,
        "congestion_columns": congestion_columns,
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
    raw, _, kroute, probabilities, predictions, features, _ = _enhanced_outputs(
        model, frame, model["congestion_config"]
    )
    error_probability = _class_one_probability(
        model["error_model"], model["error_constant"], features
    )
    alternative = _model_prediction(
        model["selector_model"], model["selector_constant"], features
    )
    config = model["congestion_selective_config"]
    route, override = _select_routes(
        kroute, alternative, error_probability, probabilities,
        model["selected_threshold"], config.min_expert_confidence_gain,
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    row = np.arange(len(raw))
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, kroute],
        "CDR_MLC_congestion_selective_router": predictions[row, route],
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "minibatch_kmeans_route": kroute,
            "alternative_route": alternative,
            "selective_route": route,
            "oracle_route": oracle,
            "minibatch_kmeans_error_probability": error_probability,
            "route_overridden": override,
            "selective_matches_oracle": route == oracle,
            "minibatch_kmeans_matches_oracle": kroute == oracle,
        }, index=raw.index),
    }
