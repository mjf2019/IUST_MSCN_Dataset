"""OOD-aware residual meta-stacking for trend-selected CDR-MLC.

This conservative extension preserves the selected KMeans feature trio and the
110-tree budget. The meta model acts as a residual correction to the actual
KMeans-routed expert rather than replacing it unconditionally. Corrections are
allowed only for sufficiently confident, high-margin, in-distribution rows.
All policy parameters are selected on source-only temporal blocks with a
non-degradation constraint relative to the actual KMeans router.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

from adaptive_cdr_mlc import APPLICATIONS
from congestion_feature_cdr_mlc import CongestionRouterConfig
from learned_router_cdr_mlc import _oracle_route
from meta_stacked_cdr_mlc_leakage_safe import (
    _aligned_meta_probabilities, _balanced_weights, _level_scores,
    _meta_features, _utility_matrix, _utility_routes, four_way_split,
)
from proactive_meta_stacked_cdr_mlc import select_proactive_features
from trend_selected_router_meta_stacker import (
    TrendSelectedRouterConfig, _refit_selected_experts,
    _selected_enhanced_outputs, _selected_expert_outputs,
    _utility_config, fit_selected_cdr,
)
from utility_router_cdr_mlc import _fit_utility_models


@dataclass(frozen=True)
class OODResidualConfig(TrendSelectedRouterConfig):
    blend_candidates: tuple[float, ...] = (.00, .25, .50, .75, 1.00)
    meta_margin_candidates: tuple[float, ...] = (.00, .05, .10, .20)
    ood_quantile_candidates: tuple[float, ...] = (.90, .95, .99, 1.01)
    temporal_selection_blocks: int = 4
    meta_mix_candidates: tuple[float, ...] = (.00, .25, .50, .75, 1.00)
    class_recall_tolerance: float = .005
    dg_probability_temperatures: tuple[float, ...] = (1.0, 1.5, 2.0)
    dg_distance_scales: tuple[float, ...] = (1.0, 1.25, 1.5)
    dg_capture_tolerance: float = .01
    shift_capture_quantile: float = .95
    shift_history_window: int = 10

    def validate(self):
        super().validate()
        if not self.blend_candidates:
            raise ValueError("at least one blend candidate is required")
        if any(not 0 <= value <= 1 for value in self.blend_candidates):
            raise ValueError("blend candidates must be in [0,1]")
        if 0.0 not in self.blend_candidates:
            raise ValueError("blend candidates must include 0 fallback")
        if any(value < 0 for value in self.meta_margin_candidates):
            raise ValueError("meta margins must be nonnegative")
        if any(
            not (0 < value <= 1 or value > 1)
            for value in self.ood_quantile_candidates
        ):
            raise ValueError("OOD quantiles must be positive")
        if self.temporal_selection_blocks < 2:
            raise ValueError("at least two temporal blocks are required")
        if not self.meta_mix_candidates or any(
            not 0 <= value <= 1 for value in self.meta_mix_candidates
        ):
            raise ValueError("meta mix candidates must be in [0,1]")
        if self.class_recall_tolerance < 0:
            raise ValueError("class recall tolerance must be nonnegative")
        if not self.dg_probability_temperatures or any(
            value < 1 for value in self.dg_probability_temperatures
        ):
            raise ValueError(
                "DG probability temperatures must be at least 1"
            )
        if not self.dg_distance_scales or any(
            value < 1 for value in self.dg_distance_scales
        ):
            raise ValueError("DG distance scales must be at least 1")
        if len(self.dg_probability_temperatures) != len(
            self.dg_distance_scales
        ):
            raise ValueError(
                "DG temperature and distance stress lists must align"
            )
        if self.dg_capture_tolerance < 0:
            raise ValueError("DG capture tolerance must be nonnegative")
        if not 0 < self.shift_capture_quantile <= 1:
            raise ValueError("shift capture quantile must be in (0,1]")
        if self.shift_history_window < 1:
            raise ValueError("shift history window must be positive")
        return self


def _temporal_blocks(raw, block_count):
    """Assign relative-time blocks independently inside each capture."""
    result = pd.Series(index=raw.index, dtype=int)
    for _, group in raw.groupby("sequence_id", sort=False):
        group = group.sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        positions = np.arange(len(group))
        blocks = np.minimum(
            block_count - 1,
            (positions * block_count) // max(len(group), 1),
        )
        result.loc[group.index] = blocks
    return result.loc[raw.index].to_numpy(dtype=int)


def _block_scores(truth, prediction, blocks):
    scores = {}
    for block in np.unique(blocks):
        mask = blocks == block
        scores[str(int(block))] = float(f1_score(
            truth[mask], prediction[mask],
            labels=APPLICATIONS, average="macro", zero_division=0,
        ))
    return scores


def _class_recalls(truth, prediction):
    values = recall_score(
        truth, prediction, labels=APPLICATIONS,
        average=None, zero_division=0,
    )
    return {
        label: float(value)
        for label, value in zip(APPLICATIONS, values)
    }


def _meta_margin(probability):
    ordered = np.sort(probability, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _temperature_scale(probability, temperature):
    """Flatten meta evidence to emulate confidence loss under domain shift."""
    if temperature == 1:
        return probability
    scaled = np.power(
        np.clip(probability, np.finfo(float).tiny, 1.0),
        1.0 / temperature,
    )
    return scaled / scaled.sum(axis=1, keepdims=True)


def _capture_accuracy(truth, prediction, capture_ids):
    return {
        str(capture): float(np.mean(prediction[capture_ids == capture] == truth[
            capture_ids == capture
        ]))
        for capture in np.unique(capture_ids)
    }


def _shift_severity(minimum_distance, probability, distance_scale):
    """Label-free shift score from router distance and meta uncertainty."""
    safe_scale = max(float(distance_scale), np.finfo(float).eps)
    confidence = probability.max(axis=1)
    margin = _meta_margin(probability)
    return (
        minimum_distance / safe_scale
        + (1.0 - confidence)
        + (1.0 - margin)
    )


def _causal_shift_scores(raw, row_severity, window):
    """Trailing median per sequence; never reads a future observation."""
    scores = pd.Series(index=raw.index, dtype=float)
    severity = pd.Series(row_severity, index=raw.index, dtype=float)
    for _, group in raw.groupby("sequence_id", sort=False):
        ordered = group.sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        scores.loc[ordered.index] = (
            severity.loc[ordered.index]
            .rolling(window=window, min_periods=1)
            .median()
            .to_numpy()
        )
    return scores.loc[raw.index].to_numpy(dtype=float)


def _ood_threshold(quantile, reference_distances):
    if quantile > 1:
        return float("inf")
    return float(np.quantile(reference_distances, quantile))


def _residual_prediction(
    actual_probability,
    meta_probability,
    minimum_distance,
    alpha,
    confidence_threshold,
    margin_threshold,
    distance_threshold,
):
    confidence = meta_probability.max(axis=1)
    margin = _meta_margin(meta_probability)
    eligible = (
        (confidence >= confidence_threshold)
        & (margin >= margin_threshold)
        & (minimum_distance <= distance_threshold)
        & (alpha > 0)
    )
    final_probability = actual_probability.copy()
    if eligible.any():
        final_probability[eligible] = (
            (1.0 - alpha) * actual_probability[eligible]
            + alpha * meta_probability[eligible]
        )
    prediction = np.asarray(APPLICATIONS)[
        final_probability.argmax(axis=1)
    ]
    return prediction, final_probability, confidence, margin, eligible


def fit_ood_residual_meta_stacker(source, config: OODResidualConfig):
    config.validate()
    split = four_way_split(source, config)

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

    # Source-only reference geometry for OOD rejection.
    _, _, expert_distances, _, _, _ = _selected_expert_outputs(
        initial, split["expert"]
    )
    expert_min_distance = expert_distances.min(axis=1)

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
    class_balanced = RandomForestClassifier(
        n_estimators=config.meta_trees,
        max_depth=config.meta_max_depth,
        min_samples_leaf=config.meta_min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.random_state + 500,
    ).fit(meta_x, meta_truth)
    level_balanced = RandomForestClassifier(
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
        "class_balanced": class_balanced,
        "level_class_balanced": level_balanced,
    }

    (
        selection_raw, selection_distances, selection_route, _,
        selection_predictions, selection_base, _,
    ) = _selected_enhanced_outputs(preliminary, split["selection"])
    selection_utility = _utility_matrix(
        utility_models, utility_constants, selection_base
    )
    selection_x = _meta_features(selection_base, selection_utility)
    row = np.arange(len(selection_raw))
    actual_probability = _selected_expert_outputs(
        preliminary, split["selection"]
    )[4]
    # Align the unfiltered expert output to context-eligible selection rows.
    expert_raw = _selected_expert_outputs(
        preliminary, split["selection"]
    )[0]
    positions = expert_raw.index.get_indexer(selection_raw.index)
    if (positions < 0).any():
        raise RuntimeError("selection probability alignment failed")
    actual_probability = actual_probability[
        positions, selection_route
    ]
    actual_prediction = selection_predictions[row, selection_route]
    truth = selection_raw.traffic_label.to_numpy()
    levels = selection_raw.congestion_level.to_numpy()
    blocks = _temporal_blocks(
        selection_raw, config.temporal_selection_blocks
    )
    actual_block_scores = _block_scores(
        truth, actual_prediction, blocks
    )
    actual_level_scores = _level_scores(
        truth, actual_prediction, levels
    )
    minimum_distance = selection_distances.min(axis=1)

    trials = []
    for variant, candidate_model in meta_models.items():
        meta_probability = _aligned_meta_probabilities(
            candidate_model, selection_x
        )
        for quantile in config.ood_quantile_candidates:
            distance_threshold = _ood_threshold(
                quantile, expert_min_distance
            )
            for alpha in config.blend_candidates:
                for confidence_threshold in (
                    config.meta_confidence_candidates
                ):
                    for margin_threshold in (
                        config.meta_margin_candidates
                    ):
                        prediction, _, confidence, margin, eligible = (
                            _residual_prediction(
                                actual_probability,
                                meta_probability,
                                minimum_distance,
                                alpha,
                                confidence_threshold,
                                margin_threshold,
                                distance_threshold,
                            )
                        )
                        block_scores = _block_scores(
                            truth, prediction, blocks
                        )
                        block_deltas = {
                            block: (
                                block_scores[block]
                                - actual_block_scores[block]
                            )
                            for block in block_scores
                        }
                        level_scores = _level_scores(
                            truth, prediction, levels
                        )
                        level_deltas = {
                            level: (
                                level_scores[level]
                                - actual_level_scores[level]
                            )
                            for level in level_scores
                        }
                        trials.append({
                            "meta_variant": variant,
                            "alpha": float(alpha),
                            "confidence_threshold": float(
                                confidence_threshold
                            ),
                            "margin_threshold": float(margin_threshold),
                            "ood_quantile": float(quantile),
                            "ood_distance_threshold": float(
                                distance_threshold
                            ),
                            "macro_f1": float(f1_score(
                                truth, prediction,
                                labels=APPLICATIONS,
                                average="macro", zero_division=0,
                            )),
                            "corrected_rows": int(eligible.sum()),
                            "corrected_fraction": float(eligible.mean()),
                            "mean_meta_confidence": float(
                                confidence.mean()
                            ),
                            "mean_meta_margin": float(margin.mean()),
                            "temporal_block_macro_f1": block_scores,
                            "temporal_block_delta_vs_actual": (
                                block_deltas
                            ),
                            "worst_temporal_block_delta_vs_actual": (
                                float(min(block_deltas.values()))
                            ),
                            "level_macro_f1": level_scores,
                            "level_delta_vs_actual": level_deltas,
                        })

    feasible = [
        trial for trial in trials
        if trial["worst_temporal_block_delta_vs_actual"] >= -1e-12
    ]
    if not feasible:
        raise RuntimeError("alpha=0 fallback was unexpectedly infeasible")
    selected_trial = max(feasible, key=lambda trial: (
        trial["macro_f1"],
        trial["worst_temporal_block_delta_vs_actual"],
        -trial["corrected_fraction"],
        -trial["alpha"],
    ))

    # Class-balanced policy: combine the two already-trained meta models,
    # then require both temporal robustness and bounded per-class recall loss.
    selection_meta_probability = {
        variant: _aligned_meta_probabilities(model, selection_x)
        for variant, model in meta_models.items()
    }
    actual_class_recalls = _class_recalls(
        truth, actual_prediction
    )
    class_balanced_trials = []
    for meta_mix in config.meta_mix_candidates:
        mixed_probability = (
            meta_mix
            * selection_meta_probability["class_balanced"]
            + (1.0 - meta_mix)
            * selection_meta_probability["level_class_balanced"]
        )
        for quantile in config.ood_quantile_candidates:
            distance_threshold = _ood_threshold(
                quantile, expert_min_distance
            )
            for alpha in config.blend_candidates:
                for confidence_threshold in (
                    config.meta_confidence_candidates
                ):
                    for margin_threshold in (
                        config.meta_margin_candidates
                    ):
                        prediction, _, confidence, margin, eligible = (
                            _residual_prediction(
                                actual_probability,
                                mixed_probability,
                                minimum_distance,
                                alpha,
                                confidence_threshold,
                                margin_threshold,
                                distance_threshold,
                            )
                        )
                        block_scores = _block_scores(
                            truth, prediction, blocks
                        )
                        block_deltas = {
                            block: (
                                block_scores[block]
                                - actual_block_scores[block]
                            )
                            for block in block_scores
                        }
                        class_recalls = _class_recalls(
                            truth, prediction
                        )
                        class_deltas = {
                            label: (
                                class_recalls[label]
                                - actual_class_recalls[label]
                            )
                            for label in APPLICATIONS
                        }
                        class_balanced_trials.append({
                            "meta_mix_class_balanced": float(meta_mix),
                            "meta_mix_level_balanced": float(
                                1.0 - meta_mix
                            ),
                            "alpha": float(alpha),
                            "confidence_threshold": float(
                                confidence_threshold
                            ),
                            "margin_threshold": float(margin_threshold),
                            "ood_quantile": float(quantile),
                            "ood_distance_threshold": float(
                                distance_threshold
                            ),
                            "accuracy": float(np.mean(
                                prediction == truth
                            )),
                            "balanced_accuracy": float(
                                balanced_accuracy_score(
                                    truth, prediction
                                )
                            ),
                            "macro_f1": float(f1_score(
                                truth, prediction,
                                labels=APPLICATIONS,
                                average="macro", zero_division=0,
                            )),
                            "corrected_rows": int(eligible.sum()),
                            "corrected_fraction": float(eligible.mean()),
                            "mean_meta_confidence": float(
                                confidence.mean()
                            ),
                            "mean_meta_margin": float(margin.mean()),
                            "temporal_block_macro_f1": block_scores,
                            "temporal_block_delta_vs_actual": (
                                block_deltas
                            ),
                            "worst_temporal_block_delta_vs_actual": (
                                float(min(block_deltas.values()))
                            ),
                            "class_recall": class_recalls,
                            "class_recall_delta_vs_actual": class_deltas,
                            "worst_class_recall_delta_vs_actual": (
                                float(min(class_deltas.values()))
                            ),
                        })
    class_feasible = [
        trial for trial in class_balanced_trials
        if (
            trial["worst_temporal_block_delta_vs_actual"] >= -1e-12
            and trial["worst_class_recall_delta_vs_actual"]
            >= -config.class_recall_tolerance
        )
    ]
    if not class_feasible:
        raise RuntimeError(
            "alpha=0 class-balanced fallback was unexpectedly infeasible"
        )
    selected_class_policy = max(
        class_feasible,
        key=lambda trial: (
            trial["macro_f1"],
            trial["balanced_accuracy"],
            trial["worst_class_recall_delta_vs_actual"],
            trial["worst_temporal_block_delta_vs_actual"],
            -trial["corrected_fraction"],
            -trial["alpha"],
        ),
    )

    # Domain-generalized policy selection. Each capture is treated as an
    # independently held-out environment for policy scoring. In addition,
    # source-only router evidence is stressed by flattening meta confidence
    # and inflating KMeans distance. No synthetic row is used to fit a tree.
    capture_ids = selection_raw.sequence_id.astype(str).to_numpy()
    actual_capture_accuracy = _capture_accuracy(
        truth, actual_prediction, capture_ids
    )
    dg_trials = []
    for meta_mix in config.meta_mix_candidates:
        original_probability = (
            meta_mix
            * selection_meta_probability["class_balanced"]
            + (1.0 - meta_mix)
            * selection_meta_probability["level_class_balanced"]
        )
        for quantile in config.ood_quantile_candidates:
            distance_threshold = _ood_threshold(
                quantile, expert_min_distance
            )
            for alpha in config.blend_candidates:
                for confidence_threshold in (
                    config.meta_confidence_candidates
                ):
                    for margin_threshold in (
                        config.meta_margin_candidates
                    ):
                        environments = []
                        for temperature, distance_scale in zip(
                            config.dg_probability_temperatures,
                            config.dg_distance_scales,
                        ):
                            stressed_probability = _temperature_scale(
                                original_probability, temperature
                            )
                            prediction, _, _, _, eligible = (
                                _residual_prediction(
                                    actual_probability,
                                    stressed_probability,
                                    minimum_distance * distance_scale,
                                    alpha,
                                    confidence_threshold,
                                    margin_threshold,
                                    distance_threshold,
                                )
                            )
                            block_scores = _block_scores(
                                truth, prediction, blocks
                            )
                            block_deltas = {
                                block: (
                                    block_scores[block]
                                    - actual_block_scores[block]
                                )
                                for block in block_scores
                            }
                            capture_accuracy = _capture_accuracy(
                                truth, prediction, capture_ids
                            )
                            capture_deltas = {
                                capture: (
                                    capture_accuracy[capture]
                                    - actual_capture_accuracy[capture]
                                )
                                for capture in capture_accuracy
                            }
                            class_recalls = _class_recalls(
                                truth, prediction
                            )
                            class_deltas = {
                                label: (
                                    class_recalls[label]
                                    - actual_class_recalls[label]
                                )
                                for label in APPLICATIONS
                            }
                            environments.append({
                                "probability_temperature": float(
                                    temperature
                                ),
                                "distance_scale": float(distance_scale),
                                "macro_f1": float(f1_score(
                                    truth, prediction,
                                    labels=APPLICATIONS,
                                    average="macro",
                                    zero_division=0,
                                )),
                                "balanced_accuracy": float(
                                    balanced_accuracy_score(
                                        truth, prediction
                                    )
                                ),
                                "corrected_fraction": float(
                                    eligible.mean()
                                ),
                                "worst_capture_accuracy_delta": float(
                                    min(capture_deltas.values())
                                ),
                                "worst_temporal_block_delta": float(
                                    min(block_deltas.values())
                                ),
                                "worst_class_recall_delta": float(
                                    min(class_deltas.values())
                                ),
                            })
                        dg_trials.append({
                            "meta_mix_class_balanced": float(meta_mix),
                            "meta_mix_level_balanced": float(
                                1.0 - meta_mix
                            ),
                            "alpha": float(alpha),
                            "confidence_threshold": float(
                                confidence_threshold
                            ),
                            "margin_threshold": float(margin_threshold),
                            "ood_quantile": float(quantile),
                            "ood_distance_threshold": float(
                                distance_threshold
                            ),
                            "worst_environment_macro_f1": float(min(
                                item["macro_f1"] for item in environments
                            )),
                            "mean_environment_macro_f1": float(np.mean([
                                item["macro_f1"] for item in environments
                            ])),
                            "worst_environment_balanced_accuracy": float(min(
                                item["balanced_accuracy"]
                                for item in environments
                            )),
                            "worst_capture_accuracy_delta": float(min(
                                item["worst_capture_accuracy_delta"]
                                for item in environments
                            )),
                            "worst_temporal_block_delta": float(min(
                                item["worst_temporal_block_delta"]
                                for item in environments
                            )),
                            "worst_class_recall_delta": float(min(
                                item["worst_class_recall_delta"]
                                for item in environments
                            )),
                            "mean_corrected_fraction": float(np.mean([
                                item["corrected_fraction"]
                                for item in environments
                            ])),
                            "stress_environment_count": len(environments),
                        })
    dg_feasible = [
        trial for trial in dg_trials
        if (
            trial["worst_capture_accuracy_delta"]
            >= -config.dg_capture_tolerance
            and trial["worst_temporal_block_delta"] >= -1e-12
            and trial["worst_class_recall_delta"]
            >= -config.class_recall_tolerance
        )
    ]
    if not dg_feasible:
        raise RuntimeError("alpha=0 DG fallback was unexpectedly infeasible")
    selected_dg_policy = max(
        dg_feasible,
        key=lambda trial: (
            trial["worst_environment_macro_f1"],
            trial["worst_environment_balanced_accuracy"],
            trial["mean_environment_macro_f1"],
            trial["worst_capture_accuracy_delta"],
            trial["worst_class_recall_delta"],
            -trial["mean_corrected_fraction"],
            -trial["alpha"],
        ),
    )

    # Label-free shift selector. Its distance scale and capture threshold
    # are calibrated exclusively from source partitions. Traffic labels are
    # deliberately not consulted: the selector only decides whether a whole
    # capture uses the mild-shift CB policy or the severe-shift DG policy.
    shift_reference_probability = (
        0.5 * selection_meta_probability["class_balanced"]
        + 0.5 * selection_meta_probability["level_class_balanced"]
    )
    shift_distance_scale = _ood_threshold(.95, expert_min_distance)
    source_shift_severity = _shift_severity(
        minimum_distance,
        shift_reference_probability,
        shift_distance_scale,
    )
    source_causal_shift_scores = _causal_shift_scores(
        selection_raw,
        source_shift_severity,
        config.shift_history_window,
    )
    shift_capture_threshold = float(np.quantile(
        source_causal_shift_scores,
        config.shift_capture_quantile,
    ))
    selected_shift_adaptive_policy = {
        "mild_shift_policy": "class_balanced_ood_residual",
        "severe_shift_policy": "domain_generalized_meta",
        "distance_scale": float(shift_distance_scale),
        "capture_score": "causal_trailing_median_shift_severity",
        "capture_threshold_quantile": float(
            config.shift_capture_quantile
        ),
        "capture_threshold": shift_capture_threshold,
        "history_window": int(config.shift_history_window),
        "source_score_summary": {
            "minimum": float(source_causal_shift_scores.min()),
            "median": float(np.median(source_causal_shift_scores)),
            "maximum": float(source_causal_shift_scores.max()),
        },
        "uses_traffic_labels": False,
        "uses_congestion_labels": False,
    }

    final = _refit_selected_experts(initial, source, config)
    final.update({
        "ood_residual_config": config,
        "congestion_config": congestion_config,
        "selected_router_features": selected_features,
        "router_trend_ranking": trend_ranking.to_dict(orient="records"),
        "context_columns": context_columns,
        "utility_models": utility_models,
        "utility_constants": utility_constants,
        "utility_correctness_counts": correctness_counts,
        "meta_models": meta_models,
        "selected_residual_policy": selected_trial,
        "residual_policy_trials": trials,
        "selected_class_balanced_policy": selected_class_policy,
        "class_balanced_policy_trials": class_balanced_trials,
        "selected_domain_generalized_policy": selected_dg_policy,
        "domain_generalized_policy_trials": dg_trials,
        "selected_shift_adaptive_policy": selected_shift_adaptive_policy,
        "expert_min_distance_quantiles": {
            str(value): _ood_threshold(value, expert_min_distance)
            for value in config.ood_quantile_candidates
        },
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
            "ood_reference_partition": "expert_only",
            "utility_partition": "strictly_after_expert",
            "meta_partition": "strictly_after_utility",
            "policy_selection_partition": "strictly_after_meta",
            "policy_baseline": "actual_kmeans_router",
            "policy_constraint": (
                "non_degrading_each_relative_time_block"
            ),
            "class_balanced_policy_constraints": {
                "non_degrading_each_relative_time_block": True,
                "maximum_per_class_recall_loss": (
                    config.class_recall_tolerance
                ),
                "selection_objectives": [
                    "macro_f1", "balanced_accuracy",
                    "worst_class_recall_delta"
                ],
            },
            "domain_generalized_policy": {
                "capture_group_validation": True,
                "source_only_probability_temperature_stress": list(
                    config.dg_probability_temperatures
                ),
                "source_only_kmeans_distance_stress": list(
                    config.dg_distance_scales
                ),
                "maximum_capture_accuracy_loss": (
                    config.dg_capture_tolerance
                ),
                "extra_fitted_trees": 0,
            },
            "shift_adaptive_policy": {
                "selection_unit": "causal_row",
                "shift_evidence": [
                    "KMeans minimum distance",
                    "meta confidence",
                    "meta probability margin"
                ],
                "threshold_partition": "source_selection_only",
                "traffic_or_congestion_labels_used": False,
                "extra_fitted_trees": 0,
            },
            "target_or_test_used_for_policy_selection": False,
        },
    })
    return final


def predict_all(model, frame):
    (
        raw, distances, route, probabilities, predictions, base, _,
    ) = _selected_enhanced_outputs(model, frame)
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], base
    )
    utility_route, _, utility_gain, _ = _utility_routes(
        route, utility, 0.0
    )
    row = np.arange(len(raw))
    utility_prediction = predictions[row, utility_route]
    actual_probability = probabilities[row, route]
    actual_prediction = predictions[row, route]
    meta_x = _meta_features(base, utility)
    policy = model["selected_residual_policy"]
    meta_probability = _aligned_meta_probabilities(
        model["meta_models"][policy["meta_variant"]], meta_x
    )
    residual, final_probability, confidence, margin, corrected = (
        _residual_prediction(
            actual_probability,
            meta_probability,
            distances.min(axis=1),
            policy["alpha"],
            policy["confidence_threshold"],
            policy["margin_threshold"],
            policy["ood_distance_threshold"],
        )
    )
    class_policy = model["selected_class_balanced_policy"]
    class_probability = _aligned_meta_probabilities(
        model["meta_models"]["class_balanced"], meta_x
    )
    level_probability = _aligned_meta_probabilities(
        model["meta_models"]["level_class_balanced"], meta_x
    )
    mixed_probability = (
        class_policy["meta_mix_class_balanced"] * class_probability
        + class_policy["meta_mix_level_balanced"] * level_probability
    )
    (
        class_balanced_residual, class_final_probability,
        class_confidence, class_margin, class_corrected,
    ) = _residual_prediction(
        actual_probability,
        mixed_probability,
        distances.min(axis=1),
        class_policy["alpha"],
        class_policy["confidence_threshold"],
        class_policy["margin_threshold"],
        class_policy["ood_distance_threshold"],
    )
    dg_policy = model["selected_domain_generalized_policy"]
    dg_probability = (
        dg_policy["meta_mix_class_balanced"] * class_probability
        + dg_policy["meta_mix_level_balanced"] * level_probability
    )
    (
        domain_generalized_residual, dg_final_probability,
        dg_confidence, dg_margin, dg_corrected,
    ) = _residual_prediction(
        actual_probability,
        dg_probability,
        distances.min(axis=1),
        dg_policy["alpha"],
        dg_policy["confidence_threshold"],
        dg_policy["margin_threshold"],
        dg_policy["ood_distance_threshold"],
    )
    shift_policy = model["selected_shift_adaptive_policy"]
    shift_reference_probability = (
        0.5 * class_probability + 0.5 * level_probability
    )
    shift_severity = _shift_severity(
        distances.min(axis=1),
        shift_reference_probability,
        shift_policy["distance_scale"],
    )
    causal_shift_score = _causal_shift_scores(
        raw,
        shift_severity,
        shift_policy["history_window"],
    )
    severe_shift = (
        causal_shift_score > shift_policy["capture_threshold"]
    )
    shift_adaptive_prediction = np.where(
        severe_shift,
        domain_generalized_residual,
        class_balanced_residual,
    )
    truth = raw.traffic_label.to_numpy()
    oracle = _oracle_route(truth, probabilities, predictions)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": actual_prediction,
        "CDR_MLC_utility_router": utility_prediction,
        "CDR_MLC_ood_residual_meta_stacker": residual,
        "CDR_MLC_class_balanced_ood_residual_meta_stacker": (
            class_balanced_residual
        ),
        "CDR_MLC_domain_generalized_meta_stacker": (
            domain_generalized_residual
        ),
        "CDR_MLC_shift_adaptive_meta_stacker": (
            shift_adaptive_prediction
        ),
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": pd.DataFrame({
            "kmeans_route": route,
            "utility_route": utility_route,
            "oracle_route": oracle,
            "utility_gain": utility_gain,
            "meta_confidence": confidence,
            "meta_margin": margin,
            "corrected_by_meta": corrected,
            "minimum_cluster_distance": distances.min(axis=1),
            "residual_probability_max": final_probability.max(axis=1),
            "class_balanced_meta_confidence": class_confidence,
            "class_balanced_meta_margin": class_margin,
            "corrected_by_class_balanced_meta": class_corrected,
            "class_balanced_probability_max": (
                class_final_probability.max(axis=1)
            ),
            "domain_generalized_meta_confidence": dg_confidence,
            "domain_generalized_meta_margin": dg_margin,
            "corrected_by_domain_generalized_meta": dg_corrected,
            "domain_generalized_probability_max": (
                dg_final_probability.max(axis=1)
            ),
            "shift_severity": shift_severity,
            "causal_shift_score": causal_shift_score,
            "severe_shift_window": severe_shift,
            "shift_adaptive_selected_policy": np.where(
                severe_shift, "DG-Meta", "CB-Meta"
            ),
        }, index=raw.index),
    }
