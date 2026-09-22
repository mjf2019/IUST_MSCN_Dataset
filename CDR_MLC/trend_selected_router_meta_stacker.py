"""Leakage-safe CDR-MLC with training-only trend-selected KMeans inputs.

The paper's fixed router inputs (TcpRtt, SynAck, AckDat) are replaced by three
features selected before router fitting. Selection uses only the earliest
expert-training partition, is label-free, congestion-label-free and causal.
After selection, the trio is frozen for StandardScaler, KMeans, expert fitting,
all later training partitions and target inference. The safe utility/meta
pipeline otherwise remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from adaptive_cdr_mlc import (
    APPLICATIONS, aligned_probabilities, make_preprocessor,
    select_classifier_columns, trend_frame,
)
from compare_clean_valid import STATS, rf_config
from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig,
    congestion_feature_frame,
)
from learned_router_cdr_mlc import (
    LearnedRouterConfig, _gate_features, _oracle_route,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    _aligned_meta_probabilities, _balanced_weights, _hybrid_prediction,
    _level_scores, _meta_features, _utility_matrix, _utility_routes,
    four_way_split,
)
from proactive_meta_stacked_cdr_mlc import select_proactive_features
from utility_router_cdr_mlc import _fit_utility_models


@dataclass(frozen=True)
class TrendSelectedRouterConfig:
    window: int = 3
    congestion_window: int = 10
    router_candidates: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    context_features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    selected_feature_count: int = 3
    trend_window: int = 3
    redundancy_limit: float = .95
    expert_trees: int = 20
    utility_trees: int = 10
    meta_trees: int = 20
    utility_max_depth: int = 12
    utility_min_samples_leaf: int = 12
    meta_max_depth: int = 14
    meta_min_samples_leaf: int = 8
    expert_fraction: float = .55
    utility_fraction: float = .15
    meta_fraction: float = .15
    meta_confidence_candidates: tuple[float, ...] = (
        .00, .40, .50, .60, .70, .80, .90, 1.01,
    )
    random_state: int = 42

    @property
    def candidate_features(self):
        # Structural interface required by select_proactive_features().
        return self.router_candidates

    def validate(self):
        fractions = (
            self.expert_fraction, self.utility_fraction, self.meta_fraction,
        )
        if any(value <= 0 for value in fractions) or sum(fractions) >= 1:
            raise ValueError(
                "positive fractions with a nonempty selection tail are required"
            )
        if self.window != 3 or self.trend_window != 3:
            raise ValueError(
                "the paper protocol requires router and trend windows of 3"
            )
        if self.congestion_window < 3:
            raise ValueError("congestion_window must be >= 3")
        if self.selected_feature_count != 3:
            raise ValueError("the router must use exactly three features")
        if len(set(self.router_candidates)) < 3:
            raise ValueError("at least three router candidates are required")
        if len(set(self.context_features)) < 3:
            raise ValueError("at least three context features are required")
        if not 0 <= self.redundancy_limit <= 1:
            raise ValueError("redundancy_limit must be in [0,1]")
        return self


def fit_selected_cdr(source, router_features, config):
    """Fit the original three-cluster CDR-MLC with a selected feature trio."""
    router_features = tuple(router_features)
    if len(router_features) != 3 or len(set(router_features)) != 3:
        raise ValueError("exactly three unique router features are required")
    view = trend_frame(source, list(router_features), config.window)
    trend_columns = [
        f"{feature}_{stat}"
        for feature in router_features
        for stat in STATS
    ]
    if len(view) < 3:
        raise ValueError(
            "trend-selected CDR-MLC has fewer than three complete windows"
        )
    scaler = StandardScaler().fit(view[trend_columns])
    z = scaler.transform(view[trend_columns])
    router = MiniBatchKMeans(
        n_clusters=3,
        batch_size=1024,
        n_init=10,
        max_iter=100,
        random_state=config.random_state,
    ).fit(z)
    routes = router.predict(z)
    if len(np.unique(routes)) != 3:
        raise ValueError("trend-selected router produced fewer than 3 clusters")

    raw = source.loc[view.index]
    # Match the paper's design: router inputs are not reused by the experts.
    numeric, categorical = select_classifier_columns(
        raw, list(router_features)
    )
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    labels = raw.traffic_label.to_numpy()
    experts, cluster_counts = {}, []
    for cluster in range(3):
        mask = routes == cluster
        if not mask.any():
            raise ValueError(f"source cluster {cluster} is empty")
        experts[cluster] = RandomForestClassifier(
            **rf_config(config.random_state, config.expert_trees)
        ).fit(x[mask], labels[mask])
        counts = pd.Series(labels[mask]).value_counts().to_dict()
        cluster_counts.append({
            "cluster": cluster,
            "rows": int(mask.sum()),
            **{
                f"class_{label}": int(counts.get(label, 0))
                for label in APPLICATIONS
            },
        })
    return {
        "window": config.window,
        "router_features": router_features,
        "trend_columns": trend_columns,
        "scaler": scaler,
        "router": router,
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "experts": experts,
        "source_eligible_index": view.index.tolist(),
        "cluster_counts": cluster_counts,
    }


def _selected_expert_outputs(model, frame):
    features = list(model["router_features"])
    view = trend_frame(frame, features, model["window"])
    raw = frame.loc[view.index]
    z = model["scaler"].transform(view[model["trend_columns"]])
    distances = model["router"].transform(z)
    routes = model["router"].predict(z)
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(raw[columns])
    probabilities, predictions = [], []
    for cluster in range(3):
        probability = aligned_probabilities(
            model["experts"][cluster], x, APPLICATIONS
        )
        probabilities.append(probability)
        predictions.append(
            np.asarray(APPLICATIONS)[probability.argmax(axis=1)]
        )
    return (
        raw, z, distances, routes,
        np.stack(probabilities, axis=1),
        np.stack(predictions, axis=1),
    )


def _selected_enhanced_outputs(model, frame):
    raw, _, distances, routes, probabilities, predictions = (
        _selected_expert_outputs(model, frame)
    )
    congestion, _ = congestion_feature_frame(
        frame, model["congestion_config"]
    )
    rows = raw.index[raw.index.isin(congestion.index)]
    if not len(rows):
        raise ValueError("no common router/context windows")
    positions = raw.index.get_indexer(rows)
    congestion_columns = [
        column for column in congestion
        if column.startswith("router_")
    ]
    context = congestion.loc[
        rows, congestion_columns
    ].to_numpy(dtype=float)
    base = _gate_features(
        distances[positions], routes[positions], probabilities[positions]
    )
    return (
        raw.loc[rows],
        distances[positions],
        routes[positions],
        probabilities[positions],
        predictions[positions],
        np.column_stack([base, context]),
        congestion_columns,
    )


def _refit_selected_experts(initial, full_source, config):
    features = list(initial["router_features"])
    view = trend_frame(full_source, features, initial["window"])
    raw = full_source.loc[view.index]
    z = initial["scaler"].transform(view[initial["trend_columns"]])
    routes = initial["router"].predict(z)
    numeric, categorical = select_classifier_columns(raw, features)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    labels = raw.traffic_label.to_numpy()
    experts, counts = {}, []
    for cluster in range(3):
        mask = routes == cluster
        if not mask.any():
            raise ValueError(f"empty full-source cluster {cluster}")
        experts[cluster] = RandomForestClassifier(
            **rf_config(
                config.random_state + cluster, config.expert_trees
            )
        ).fit(x[mask], labels[mask])
        counts.append({"cluster": cluster, "rows": int(mask.sum())})
    final = dict(initial)
    final.update({
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "experts": experts,
        "full_source_rows": len(raw),
        "source_eligible_index": view.index.tolist(),
        "full_source_cluster_counts": counts,
    })
    return final


def _utility_config(config):
    return SimpleNamespace(
        utility_trees=config.utility_trees,
        max_depth=config.utility_max_depth,
        min_samples_leaf=config.utility_min_samples_leaf,
        random_state=config.random_state,
    )


def fit_trend_selected_meta_stacker(
    source, config: TrendSelectedRouterConfig
):
    config.validate()
    split = four_way_split(source, config)

    # The only adaptive feature-selection step. It is performed before any
    # router fit and cannot see labels, congestion labels or later partitions.
    selected_features, trend_ranking = select_proactive_features(
        split["expert"], config
    )
    initial = fit_selected_cdr(
        split["expert"], selected_features, config
    )
    congestion_config = CongestionRouterConfig(
        window=config.congestion_window,
        features=tuple(config.context_features),
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    ).validate()
    preliminary = dict(initial)
    preliminary["congestion_config"] = congestion_config

    (
        utility_raw, _, _, _, utility_predictions,
        utility_features, context_columns,
    ) = _selected_enhanced_outputs(preliminary, split["utility"])
    utility_truth = utility_raw.traffic_label.to_numpy()
    utility_models, utility_constants, correctness_counts = (
        _fit_utility_models(
            utility_features,
            utility_predictions,
            utility_truth,
            utility_raw.congestion_level.to_numpy(),
            _utility_config(config),
        )
    )

    (
        meta_raw, _, _, _, _, meta_base, _,
    ) = _selected_enhanced_outputs(preliminary, split["meta"])
    meta_utility = _utility_matrix(
        utility_models, utility_constants, meta_base
    )
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
    level_balanced_model = RandomForestClassifier(
        n_estimators=config.meta_trees,
        max_depth=config.meta_max_depth,
        min_samples_leaf=config.meta_min_samples_leaf,
        max_features="sqrt",
        class_weight=None,
        n_jobs=-1,
        random_state=config.random_state + 700,
    ).fit(
        meta_x,
        meta_truth,
        sample_weight=_balanced_weights(
            meta_raw.congestion_level.to_numpy(), meta_truth
        ),
    )
    meta_models = {
        "class_balanced": meta_model,
        "level_class_balanced": level_balanced_model,
    }

    (
        selection_raw, _, selection_route, _, selection_predictions,
        selection_base, _,
    ) = _selected_enhanced_outputs(preliminary, split["selection"])
    selection_utility = _utility_matrix(
        utility_models, utility_constants, selection_base
    )
    selection_x = _meta_features(selection_base, selection_utility)
    hard_route, _, _, _ = _utility_routes(
        selection_route, selection_utility, 0.0
    )
    row = np.arange(len(selection_raw))
    hard_prediction = selection_predictions[row, hard_route]
    truth = selection_raw.traffic_label.to_numpy()
    levels = selection_raw.congestion_level.to_numpy()
    hard_scores = _level_scores(truth, hard_prediction, levels)

    trials = []
    for variant, candidate_model in meta_models.items():
        probability = _aligned_meta_probabilities(
            candidate_model, selection_x
        )
        for threshold in config.meta_confidence_candidates:
            prediction, _, confidence, use_meta = _hybrid_prediction(
                probability, hard_prediction, threshold
            )
            level_scores = _level_scores(truth, prediction, levels)
            deltas = {
                level: level_scores[level] - hard_scores[level]
                for level in level_scores
            }
            trials.append({
                "meta_variant": variant,
                "meta_confidence_threshold": float(threshold),
                "macro_f1": float(f1_score(
                    truth, prediction,
                    average="macro", zero_division=0,
                )),
                "meta_rows": int(use_meta.sum()),
                "meta_fraction": float(use_meta.mean()),
                "mean_meta_confidence": float(confidence.mean()),
                "level_macro_f1": level_scores,
                "level_macro_f1_delta_vs_hard": deltas,
                "worst_level_delta_vs_hard": float(min(deltas.values())),
            })
    feasible = [
        trial for trial in trials
        if trial["worst_level_delta_vs_hard"] >= -1e-12
    ]
    if not feasible:
        raise RuntimeError("no non-degrading meta threshold was found")
    selected_trial = max(feasible, key=lambda trial: (
        trial["worst_level_delta_vs_hard"],
        trial["macro_f1"],
        trial["meta_confidence_threshold"],
        -trial["meta_rows"],
    ))

    # Router feature trio, scaler and KMeans remain frozen. Only expert
    # classifiers are refitted on all permitted source-development rows.
    helper = LearnedRouterConfig(
        window=config.window,
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    final = _refit_selected_experts(initial, source, helper)
    final.update({
        "trend_selected_router_config": config,
        "congestion_config": congestion_config,
        "selected_router_features": selected_features,
        "router_trend_ranking": trend_ranking.to_dict(orient="records"),
        "context_columns": context_columns,
        "utility_models": utility_models,
        "utility_constants": utility_constants,
        "utility_correctness_counts": correctness_counts,
        "meta_models": meta_models,
        "selected_meta_variant": selected_trial["meta_variant"],
        "selected_meta_confidence": selected_trial[
            "meta_confidence_threshold"
        ],
        "meta_selection_trials": trials,
        "partition_rows": {
            "expert_raw": len(split["expert"]),
            "router_fit_eligible": len(initial["source_eligible_index"]),
            "utility": len(utility_raw),
            "meta": len(meta_raw),
            "selection": len(selection_raw),
        },
        "leakage_control": {
            "router_feature_selection_partition": "expert_only",
            "router_feature_selection_uses_traffic_labels": False,
            "router_feature_selection_uses_congestion_labels": False,
            "router_feature_selection_is_causal": True,
            "selected_router_features_frozen_after_selection": True,
            "router_scaler_fit_partition": "expert_only",
            "kmeans_fit_partition": "expert_only",
            "utility_meta_selection_are_strictly_later": True,
            "final_experts_refit_on_full_development": True,
            "selected_router_features_excluded_from_expert_inputs": True,
        },
    })
    return final


def predict_all(model, frame):
    (
        raw, _, route, probabilities, predictions, base, _,
    ) = _selected_enhanced_outputs(model, frame)
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], base
    )
    utility_route, _, utility_gain, _ = _utility_routes(
        route, utility, 0.0
    )
    row = np.arange(len(raw))
    hard_prediction = predictions[row, utility_route]
    meta_x = _meta_features(base, utility)
    meta_probability = _aligned_meta_probabilities(
        model["meta_models"][model["selected_meta_variant"]], meta_x
    )
    stacked, meta_prediction, confidence, use_meta = _hybrid_prediction(
        meta_probability,
        hard_prediction,
        model["selected_meta_confidence"],
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, route],
        "CDR_MLC_utility_router": hard_prediction,
        "CDR_MLC_trend_selected_meta_stacker": stacked,
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "kmeans_route": route,
            "utility_route": utility_route,
            "oracle_route": oracle,
            "utility_gain": utility_gain,
            "meta_prediction": meta_prediction,
            "meta_confidence": confidence,
            "used_meta_prediction": use_meta,
            "meta_prediction_correct": meta_prediction == truth,
        }, index=raw.index),
    }
