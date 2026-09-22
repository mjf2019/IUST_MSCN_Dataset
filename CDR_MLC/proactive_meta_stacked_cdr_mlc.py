"""Leakage-safe proactive meta-stacking for fixed CDR-MLC.

The proactive phase ranks congestion candidates on the earliest training
partition only.  It uses no traffic labels and no congestion-level labels.
For each candidate, causal three-sample trends are measured independently
inside each flow sequence.  Three non-redundant candidates are frozen and
their current value, slope, acceleration, persistence and one-step-ahead
forecast are appended to the complete context-aware feature bank used by the
utility and meta layers.

This module coexists with the legacy and leakage-safe context-aware versions
so their published results remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import fit_fixed_cdr
from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES, CongestionRouterConfig,
)
from congestion_selective_router_cdr_mlc import _enhanced_outputs
from learned_router_cdr_mlc import (
    LearnedRouterConfig, _oracle_route, _refit_experts_with_frozen_router,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    _aligned_meta_probabilities, _balanced_weights, _hybrid_prediction,
    _level_scores, _meta_features, _utility_matrix, _utility_routes,
    four_way_split,
)
from utility_router_cdr_mlc import _fit_utility_models


@dataclass(frozen=True)
class ProactiveMetaStackConfig:
    window: int = 3
    congestion_window: int = 10
    candidate_features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
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

    def validate(self):
        fractions = (
            self.expert_fraction, self.utility_fraction, self.meta_fraction,
        )
        if any(value <= 0 for value in fractions) or sum(fractions) >= 1:
            raise ValueError(
                "positive fractions with a nonempty selection tail are required"
            )
        if self.window < 1 or self.congestion_window < 3:
            raise ValueError(
                "base window must be positive and congestion window >= 3"
            )
        if self.trend_window != 3:
            raise ValueError(
                "the proactive protocol currently requires trend_window=3"
            )
        if self.selected_feature_count != 3:
            raise ValueError(
                "the proactive protocol currently selects exactly 3 features"
            )
        if len(set(self.candidate_features)) < self.selected_feature_count:
            raise ValueError("at least three unique candidates are required")
        if not 0 <= self.redundancy_limit <= 1:
            raise ValueError("redundancy_limit must be in [0, 1]")
        return self


def _ordered_groups(frame):
    for _, group in frame.groupby("sequence_id", sort=False):
        yield group.sort_values(["timestamp", "source_row"], kind="stable")


def _robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    q10, q90 = np.quantile(values, [.10, .90])
    scale = float(q90 - q10)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.median(np.abs(values)))
    return max(scale, 1e-12)


def rank_causal_trends(frame, candidates):
    """Rank candidates without labels, using only trailing triples.

    Score = robust median absolute slope x directional persistence x coverage.
    Every statistic is learned exclusively from the supplied training frame.
    """
    available = [
        name for name in dict.fromkeys(candidates)
        if name in frame.columns
    ]
    if len(available) < 3:
        raise ValueError(
            f"only {len(available)} proactive candidates exist in the data"
        )

    records = []
    for feature in available:
        complete_slopes, persistence_values = [], []
        possible = 0
        raw_values = pd.to_numeric(frame[feature], errors="coerce")
        scale = _robust_scale(raw_values.to_numpy())
        for group in _ordered_groups(frame):
            values = pd.to_numeric(
                group[feature], errors="coerce"
            ).to_numpy(dtype=float)
            if len(values) < 3:
                continue
            possible += len(values) - 2
            x0, x1, x2 = values[:-2], values[1:-1], values[2:]
            valid = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(x2)
            if not valid.any():
                continue
            d1, d2 = x1[valid] - x0[valid], x2[valid] - x1[valid]
            complete_slopes.extend(((x2[valid] - x0[valid]) / 2.0).tolist())
            persistence_values.extend(
                ((d1 * d2 >= 0) & ((np.abs(d1) + np.abs(d2)) > 0)).tolist()
            )
        valid_count = len(complete_slopes)
        coverage = valid_count / possible if possible else 0.0
        normalized_slopes = np.abs(
            np.asarray(complete_slopes, dtype=float)
        ) / scale
        trend = (
            float(np.median(normalized_slopes))
            if len(normalized_slopes) else 0.0
        )
        persistence = (
            float(np.mean(persistence_values))
            if persistence_values else 0.0
        )
        score = trend * (.5 + .5 * persistence) * coverage
        records.append({
            "feature": feature,
            "trend_score": float(score),
            "median_abs_normalized_slope": trend,
            "directional_persistence": persistence,
            "valid_triple_coverage": float(coverage),
            "valid_triples": int(valid_count),
            "robust_scale": float(scale),
        })
    ranking = pd.DataFrame(records).sort_values(
        ["trend_score", "valid_triples", "feature"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return ranking


def select_proactive_features(frame, config):
    ranking = rank_causal_trends(frame, config.candidate_features)
    numeric = frame[ranking.feature.tolist()].apply(
        pd.to_numeric, errors="coerce"
    )
    correlation = numeric.corr(method="spearman").abs()

    selected = []
    rejection_reason = {}
    for feature in ranking.feature:
        redundant_with = [
            previous for previous in selected
            if (
                feature in correlation.index
                and previous in correlation.columns
                and np.isfinite(correlation.loc[feature, previous])
                and correlation.loc[feature, previous] > config.redundancy_limit
            )
        ]
        if redundant_with:
            rejection_reason[feature] = (
                "redundant_with:" + ",".join(redundant_with)
            )
            continue
        selected.append(feature)
        if len(selected) == config.selected_feature_count:
            break

    # A strict redundancy threshold must not make the experiment impossible.
    # The fallback is deterministic and is recorded explicitly in the audit.
    if len(selected) < config.selected_feature_count:
        for feature in ranking.feature:
            if feature not in selected:
                selected.append(feature)
                rejection_reason[feature] = "selected_by_rank_fallback"
                if len(selected) == config.selected_feature_count:
                    break
    if len(selected) != config.selected_feature_count:
        raise ValueError("unable to select three proactive features")

    ranking = ranking.copy()
    ranking["selected"] = ranking.feature.isin(selected)
    ranking["selection_order"] = ranking.feature.map(
        {feature: index + 1 for index, feature in enumerate(selected)}
    )
    ranking["selection_note"] = ranking.feature.map(rejection_reason).fillna("")
    return tuple(selected), ranking


def proactive_feature_frame(frame, selected_features):
    """Build causal trend/forecast descriptors aligned to the current row."""
    pieces = []
    for group in _ordered_groups(frame):
        values = group.loc[:, selected_features].apply(
            pd.to_numeric, errors="coerce"
        )
        descriptor = {}
        for feature in selected_features:
            current = values[feature]
            previous = current.shift(1)
            previous2 = current.shift(2)
            d1 = previous - previous2
            d2 = current - previous
            slope = (current - previous2) / 2.0
            descriptor[f"{feature}__current"] = current
            descriptor[f"{feature}__slope3"] = slope
            descriptor[f"{feature}__acceleration3"] = d2 - d1
            descriptor[f"{feature}__persistence3"] = (
                ((d1 * d2 >= 0) & ((d1.abs() + d2.abs()) > 0)).astype(float)
            )
            # Least-squares slope for equally spaced x={0,1,2} is
            # (x_t - x_t-2)/2.  This is a strictly one-step-ahead forecast.
            descriptor[f"{feature}__forecast_t_plus_1"] = current + slope
        block = pd.DataFrame(descriptor, index=group.index)
        block = block.replace([np.inf, -np.inf], np.nan).dropna()
        pieces.append(block)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces).sort_index(kind="stable")


def _utility_config(config):
    return SimpleNamespace(
        utility_trees=config.utility_trees,
        max_depth=config.utility_max_depth,
        min_samples_leaf=config.utility_min_samples_leaf,
        random_state=config.random_state,
    )


def _proactive_outputs(model, frame):
    raw, distances, kroute, probabilities, predictions, base, columns = (
        _enhanced_outputs(model, frame, model["congestion_config"])
    )
    proactive = proactive_feature_frame(
        frame, model["selected_proactive_features"]
    )
    common = raw.index[raw.index.isin(proactive.index)]
    if not len(common):
        raise ValueError("no rows have both router and proactive histories")
    positions = raw.index.get_indexer(common)
    if (positions < 0).any():
        raise RuntimeError("proactive alignment failed")
    proactive_x = model["proactive_scaler"].transform(
        proactive.loc[common].to_numpy(dtype=float)
    )
    augmented = np.column_stack([base[positions], proactive_x])
    return (
        raw.loc[common], distances[positions], kroute[positions],
        probabilities[positions], predictions[positions], augmented,
        [*columns, *proactive.columns.tolist()],
    )


def fit_proactive_meta_stacker(source, config: ProactiveMetaStackConfig):
    config.validate()
    split = four_way_split(source, config)

    # Feature selection is label-free and sees only the earliest expert block.
    selected_features, trend_ranking = select_proactive_features(
        split["expert"], config
    )
    proactive_expert = proactive_feature_frame(
        split["expert"], selected_features
    )
    if proactive_expert.empty:
        raise ValueError("expert partition is too short for proactive trends")
    proactive_scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(
        proactive_expert.to_numpy(dtype=float)
    )
    # Preserve the complete context-aware descriptor bank.  The three
    # label-free trend winners are appended as proactive signals; they do not
    # replace the original context features.
    congestion_config = CongestionRouterConfig(
        window=config.congestion_window,
        features=tuple(config.candidate_features),
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    ).validate()

    # StandardScaler, KMeans, selector and proactive scaler are all frozen
    # before utility/meta/selection rows are exposed.
    initial = fit_fixed_cdr(
        split["expert"], config.window, config.random_state,
        config.expert_trees,
    )
    preliminary = dict(initial)
    preliminary.update({
        "congestion_config": congestion_config,
        "selected_proactive_features": selected_features,
        "proactive_scaler": proactive_scaler,
    })
    helper = LearnedRouterConfig(
        window=config.window, expert_trees=config.expert_trees,
        random_state=config.random_state,
    )

    (
        utility_raw, _, _, _, utility_predictions, utility_features, columns,
    ) = _proactive_outputs(preliminary, split["utility"])
    utility_truth = utility_raw.traffic_label.to_numpy()
    utility_models, utility_constants, correctness_counts = (
        _fit_utility_models(
            utility_features, utility_predictions, utility_truth,
            utility_raw.congestion_level.to_numpy(),
            _utility_config(config),
        )
    )

    (
        meta_raw, _, _, _, meta_predictions, meta_base, _,
    ) = _proactive_outputs(preliminary, split["meta"])
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
        meta_x, meta_truth,
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
    ) = _proactive_outputs(preliminary, split["selection"])
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
    hard_level_scores = _level_scores(truth, hard_prediction, levels)

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
    feasible = [
        trial for trial in trials
        if trial["worst_level_delta_vs_hard"] >= -1e-12
    ]
    if not feasible:
        raise RuntimeError("no non-degrading meta threshold was found")
    selected = max(feasible, key=lambda trial: (
        trial["worst_level_delta_vs_hard"], trial["macro_f1"],
        trial["meta_confidence_threshold"], -trial["meta_rows"],
    ))

    final = _refit_experts_with_frozen_router(initial, source, helper)
    final.update({
        "proactive_meta_stack_config": config,
        "congestion_config": congestion_config,
        "selected_proactive_features": selected_features,
        "proactive_trend_ranking": trend_ranking.to_dict(orient="records"),
        "proactive_scaler": proactive_scaler,
        "proactive_columns": columns,
        "utility_models": utility_models,
        "utility_constants": utility_constants,
        "utility_correctness_counts": correctness_counts,
        "meta_models": meta_models,
        "selected_meta_variant": selected["meta_variant"],
        "selected_meta_confidence": selected[
            "meta_confidence_threshold"
        ],
        "meta_selection_trials": trials,
        "partition_rows": {
            "expert_raw": len(split["expert"]),
            "router_fit_eligible": len(initial["source_eligible_index"]),
            "proactive_scaler_rows": len(proactive_expert),
            "utility": len(utility_raw),
            "meta": len(meta_raw),
            "selection": len(selection_raw),
        },
        "leakage_control": {
            "trend_feature_selection_partition": "expert_only",
            "trend_selection_uses_traffic_labels": False,
            "trend_selection_uses_congestion_labels": False,
            "context_features_preserved": True,
            "proactive_features_are_additive": True,
            "trend_window_is_causal": True,
            "forecast_horizon": "t_plus_1",
            "router_scaler_fit_partition": "expert_only",
            "proactive_scaler_fit_partition": "expert_only",
            "all_preprocessors_frozen_after_expert_partition": True,
            "utility_meta_selection_are_strictly_later": True,
            "final_experts_refit_on_full_development": True,
        },
    })
    return final


def predict_all(model, frame):
    (
        raw, _, kroute, probabilities, predictions, base_features, _,
    ) = _proactive_outputs(model, frame)
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], base_features
    )
    utility_route, _, utility_gain, _ = _utility_routes(
        kroute, utility, 0.0
    )
    row = np.arange(len(raw))
    hard_prediction = predictions[row, utility_route]
    meta_x = _meta_features(base_features, utility)
    meta_probability = _aligned_meta_probabilities(
        model["meta_models"][model["selected_meta_variant"]], meta_x
    )
    stacked, meta_prediction, confidence, use_meta = _hybrid_prediction(
        meta_probability, hard_prediction,
        model["selected_meta_confidence"],
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, kroute],
        "CDR_MLC_utility_router": hard_prediction,
        "CDR_MLC_proactive_meta_stacker": stacked,
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "kmeans_route": kroute,
            "utility_route": utility_route,
            "oracle_route": oracle,
            "utility_gain": utility_gain,
            "meta_prediction": meta_prediction,
            "meta_confidence": confidence,
            "used_meta_prediction": use_meta,
            "meta_prediction_correct": meta_prediction == truth,
        }, index=raw.index),
    }
