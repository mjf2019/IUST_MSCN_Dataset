"""Strict temporal, leakage-safe meta-stacking for fixed CDR-MLC.

This module intentionally coexists with ``meta_stacked_cdr_mlc.py`` so legacy
results remain reproducible. The scaler and MiniBatchKMeans router are fitted only on
the earliest expert partition and then frozen.  Later partitions train the
utility and meta models without any feature-distribution look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import fit_fixed_cdr
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig
from congestion_selective_router_cdr_mlc import _enhanced_outputs
from learned_router_cdr_mlc import LearnedRouterConfig, _oracle_route, _refit_experts_with_frozen_router
from utility_router_cdr_mlc import (
    _balanced_weights, _fit_utility_models, _level_scores, _utility_matrix,
    _utility_routes,
)


@dataclass(frozen=True)
class MetaStackConfig:
    window: int = 3
    congestion_window: int = 10
    congestion_features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    expert_trees: int = 20
    utility_trees: int = 150
    meta_trees: int = 20
    utility_max_depth: int = 12
    utility_min_samples_leaf: int = 12
    meta_max_depth: int = 14
    meta_min_samples_leaf: int = 8
    expert_fraction: float = .55
    utility_fraction: float = .15
    meta_fraction: float = .15
    meta_confidence_candidates: tuple[float, ...] = (.00, .40, .50, .60, .70, .80, .90, 1.01)
    random_state: int = 42

    def validate(self):
        fractions = (self.expert_fraction, self.utility_fraction, self.meta_fraction)
        if any(value <= 0 for value in fractions) or sum(fractions) >= 1:
            raise ValueError("positive fractions with a nonempty selection tail are required")
        if self.window < 1 or self.congestion_window < 3:
            raise ValueError("base window must be positive and congestion window >= 3")
        if len(set(self.congestion_features)) < 3:
            raise ValueError("at least three congestion features are required")
        return self


def four_way_split(frame, config):
    parts = {"expert": [], "utility": [], "meta": [], "selection": []}
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        a = int(len(group) * config.expert_fraction)
        b = int(len(group) * (config.expert_fraction + config.utility_fraction))
        c = int(len(group) * (
            config.expert_fraction + config.utility_fraction + config.meta_fraction
        ))
        if not 0 < a < b < c < len(group):
            raise ValueError("capture too short for four-way temporal split")
        parts["expert"].append(group.iloc[:a].copy())
        parts["utility"].append(group.iloc[a:b].copy())
        parts["meta"].append(group.iloc[b:c].copy())
        parts["selection"].append(group.iloc[c:].copy())
    return {name: pd.concat(groups) for name, groups in parts.items()}


def _utility_config(config):
    # _fit_utility_models uses these attribute names structurally.
    return SimpleNamespace(
        utility_trees=config.utility_trees,
        max_depth=config.utility_max_depth,
        min_samples_leaf=config.utility_min_samples_leaf,
        random_state=config.random_state,
    )


def _congestion_config(config):
    return CongestionRouterConfig(
        window=config.congestion_window,
        features=tuple(config.congestion_features),
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    ).validate()


def _meta_features(base_features, utility):
    return np.column_stack([base_features, utility])


def _aligned_meta_probabilities(model, features):
    raw = model.predict_proba(features)
    result = np.zeros((len(features), len(APPLICATIONS)), dtype=float)
    positions = {label: index for index, label in enumerate(APPLICATIONS)}
    for source, label in enumerate(model.classes_):
        result[:, positions[label]] = raw[:, source]
    return result


def _hybrid_prediction(meta_probability, hard_prediction, threshold):
    confidence = meta_probability.max(axis=1)
    meta_prediction = np.asarray(APPLICATIONS)[meta_probability.argmax(axis=1)]
    use_meta = confidence >= threshold
    prediction = np.asarray(hard_prediction, dtype=object).copy()
    prediction[use_meta] = meta_prediction[use_meta]
    return prediction, meta_prediction, confidence, use_meta


def fit_meta_stacker(source, config: MetaStackConfig):
    config.validate()
    split = four_way_split(source, config)
    congestion_config = _congestion_config(config)

    # Strict chronology: fit every preliminary preprocessing component,
    # including StandardScaler and MiniBatchKMeans, on the earliest partition only.
    # The router/scaler are frozen for all later partitions and final inference.
    initial = fit_fixed_cdr(
        split["expert"], config.window, config.random_state, config.expert_trees
    )
    helper = LearnedRouterConfig(
        window=config.window, expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    preliminary = initial

    utility_raw, _, utility_route, _, utility_predictions, utility_features, columns = _enhanced_outputs(
        preliminary, split["utility"], congestion_config
    )
    utility_truth = utility_raw.traffic_label.to_numpy()
    utility_models, utility_constants, correctness_counts = _fit_utility_models(
        utility_features, utility_predictions, utility_truth,
        utility_raw.congestion_level.to_numpy(), _utility_config(config),
    )

    meta_raw, _, meta_route, _, meta_predictions, meta_base, _ = _enhanced_outputs(
        preliminary, split["meta"], congestion_config
    )
    meta_utility = _utility_matrix(utility_models, utility_constants, meta_base)
    meta_x = _meta_features(meta_base, meta_utility)
    meta_truth = meta_raw.traffic_label.to_numpy()
    meta_model = RandomForestClassifier(
        n_estimators=config.meta_trees,
        max_depth=config.meta_max_depth,
        min_samples_leaf=config.meta_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.random_state + 500,
    ).fit(meta_x, meta_truth)
    level_balanced_meta_model = RandomForestClassifier(
        n_estimators=config.meta_trees,
        max_depth=config.meta_max_depth,
        min_samples_leaf=config.meta_min_samples_leaf,
        max_features="sqrt",
        class_weight=None,
        n_jobs=-1,
        random_state=config.random_state + 700,
    ).fit(
        meta_x, meta_truth,
        sample_weight=_balanced_weights(
            meta_raw.congestion_level.to_numpy(), meta_truth
        ),
    )
    meta_models = {
        "class_balanced": meta_model,
        "level_class_balanced": level_balanced_meta_model,
    }

    selection_raw, _, selection_route, selection_prob, selection_predictions, selection_base, _ = _enhanced_outputs(
        preliminary, split["selection"], congestion_config
    )
    selection_utility = _utility_matrix(
        utility_models, utility_constants, selection_base
    )
    selection_x = _meta_features(selection_base, selection_utility)
    hard_route, _, _, _ = _utility_routes(selection_route, selection_utility, 0.0)
    row = np.arange(len(selection_raw))
    hard_prediction = selection_predictions[row, hard_route]
    truth = selection_raw.traffic_label.to_numpy()
    levels = selection_raw.congestion_level.to_numpy()
    hard_level_scores = _level_scores(truth, hard_prediction, levels)
    trials = []
    for variant, candidate_model in meta_models.items():
        meta_probability = _aligned_meta_probabilities(
            candidate_model, selection_x
        )
        for threshold in config.meta_confidence_candidates:
            prediction, _, confidence, use_meta = _hybrid_prediction(
                meta_probability, hard_prediction, threshold
            )
            level_scores = _level_scores(truth, prediction, levels)
            deltas = {
                level: level_scores[level] - hard_level_scores[level]
                for level in level_scores
            }
            trials.append({
                "meta_variant": variant,
                "meta_confidence_threshold": float(threshold),
                "macro_f1": float(f1_score(
                    truth, prediction, average="macro", zero_division=0
                )),
                "meta_rows": int(use_meta.sum()),
                "meta_fraction": float(use_meta.mean()),
                "mean_meta_confidence": float(confidence.mean()),
                "level_macro_f1": level_scores,
                "level_macro_f1_delta_vs_hard": deltas,
                "worst_level_delta_vs_hard": float(min(deltas.values())),
            })
    feasible = [trial for trial in trials if trial["worst_level_delta_vs_hard"] >= -1e-12]
    selected = max(feasible, key=lambda trial: (
        trial["worst_level_delta_vs_hard"], trial["macro_f1"],
        trial["meta_confidence_threshold"], -trial["meta_rows"],
    ))
    # Refit only the expert classifiers/preprocessor on all permitted
    # development rows. The scaler and MiniBatchKMeans fitted on the expert partition
    # remain frozen, so the meta feature geometry does not see future rows.
    final = _refit_experts_with_frozen_router(initial, source, helper)
    final.update({
        "meta_stack_config": config,
        "congestion_config": congestion_config,
        "congestion_columns": columns,
        "utility_models": utility_models,
        "utility_constants": utility_constants,
        "utility_correctness_counts": correctness_counts,
        "meta_models": meta_models,
        "selected_meta_variant": selected["meta_variant"],
        "selected_meta_confidence": selected["meta_confidence_threshold"],
        "meta_selection_trials": trials,
        "partition_rows": {
            "expert_raw": len(split["expert"]),
            "router_fit_eligible": len(initial["source_eligible_index"]),
            "utility": len(utility_raw), "meta": len(meta_raw),
            "selection": len(selection_raw),
        },
        "leakage_control": {
            "router_algorithm": initial["router_algorithm"],
            "router_parameters": initial["router_audit"],
            "router_scaler_fit_partition": "expert_only",
            "router_scaler_frozen_after_fit": True,
            "utility_meta_selection_are_strictly_later": True,
            "final_experts_refit_on_full_development": True,
        },
    })
    return final


def predict_all(model, frame):
    raw, _, kroute, probabilities, predictions, base_features, _ = _enhanced_outputs(
        model, frame, model["congestion_config"]
    )
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], base_features
    )
    utility_route, _, utility_gain, _ = _utility_routes(kroute, utility, 0.0)
    row = np.arange(len(raw))
    hard_prediction = predictions[row, utility_route]
    meta_x = _meta_features(base_features, utility)
    meta_probability = _aligned_meta_probabilities(
        model["meta_models"][model["selected_meta_variant"]], meta_x
    )
    stacked, meta_prediction, confidence, use_meta = _hybrid_prediction(
        meta_probability, hard_prediction, model["selected_meta_confidence"]
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, kroute],
        "CDR_MLC_utility_router": hard_prediction,
        "CDR_MLC_meta_stacker": stacked,
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "minibatch_kmeans_route": kroute, "utility_route": utility_route,
            "oracle_route": oracle, "utility_gain": utility_gain,
            "meta_prediction": meta_prediction, "meta_confidence": confidence,
            "used_meta_prediction": use_meta,
            "meta_prediction_correct": meta_prediction == truth,
        }, index=raw.index),
    }
