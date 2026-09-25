"""Leakage-safe component ablation for S-Meta evaluation protocols.

Each learned ablation preserves the same strict four-way temporal partitions,
frozen expert-only congestion geometry, test set, seed, and tree budgets.  The
only change is the component named by the ablation.  The full reference calls
the production ``fit_meta_stacker`` and ``predict_all`` functions directly.
Scenarios 1--3 train on the complete source level and evaluate on the complete,
untouched target level; no target record is used during model development.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import SCENARIOS, TIMING, fit_fixed_cdr, metrics
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from congestion_selective_router_cdr_mlc import _enhanced_outputs
from learned_router_cdr_mlc import LearnedRouterConfig, _refit_experts_with_frozen_router
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    _aligned_meta_probabilities,
    _congestion_config,
    _hybrid_prediction,
    _meta_features,
    fit_meta_stacker,
    four_way_split,
    predict_all as predict_full,
)
from benchmarks.console_output import print_compact_results
from mixed_level_protocols_leakage_safe import (
    PROTOCOLS,
    build_protocol,
    composition,
    frame_identity,
)
from utility_router_cdr_mlc import (
    _balanced_weights,
    _fit_utility_models,
    _level_scores,
    _utility_matrix,
    _utility_routes,
)


@dataclass(frozen=True)
class AblationSpec:
    use_congestion_context: bool = True
    use_utility_meta_features: bool = True
    use_level_balanced_candidate: bool = True
    use_confidence_hybrid: bool = True


LEARNED_ABLATIONS = {
    "No_Congestion_Context": AblationSpec(use_congestion_context=False),
    "No_Utility_Meta_Features": AblationSpec(use_utility_meta_features=False),
    "No_Level_Balancing": AblationSpec(use_level_balanced_candidate=False),
    "Always_Meta_No_Hybrid": AblationSpec(use_confidence_hybrid=False),
}

METHOD_ORDER = (
    "S_Meta_Full",
    "No_Congestion_Context",
    "No_Utility_Meta_Features",
    "No_Level_Balancing",
    "Always_Meta_No_Hybrid",
    "Utility_Router_Only",
    "Actual_Router_Only",
)


def _utility_config(config: MetaStackConfig):
    return SimpleNamespace(
        utility_trees=config.utility_trees,
        max_depth=config.utility_max_depth,
        min_samples_leaf=config.utility_min_samples_leaf,
        random_state=config.random_state,
    )


def _without_congestion_context(features, congestion_columns, spec):
    if spec.use_congestion_context:
        return features
    count = len(congestion_columns)
    if count == 0 or count >= features.shape[1]:
        raise ValueError("cannot identify congestion-context columns")
    return features[:, :-count]


def _ablation_meta_features(base_features, utility, spec):
    if spec.use_utility_meta_features:
        return _meta_features(base_features, utility)
    return base_features


def _fit_meta_model(x, truth, levels, config, level_balanced, seed_offset):
    model = RandomForestClassifier(
        n_estimators=config.meta_trees,
        max_depth=config.meta_max_depth,
        min_samples_leaf=config.meta_min_samples_leaf,
        max_features="sqrt",
        class_weight=None if level_balanced else "balanced",
        n_jobs=-1,
        random_state=config.random_state + seed_offset,
    )
    if level_balanced:
        return model.fit(x, truth, sample_weight=_balanced_weights(levels, truth))
    return model.fit(x, truth)


def fit_ablation(source, config: MetaStackConfig, name: str, spec: AblationSpec):
    """Fit one structural ablation without changing temporal data access."""
    split = four_way_split(source, config)
    congestion_config = _congestion_config(config)
    initial = fit_fixed_cdr(
        split["expert"], config.window, config.random_state, config.expert_trees
    )
    helper = LearnedRouterConfig(
        window=config.window,
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    )

    utility_raw, _, _, _, utility_predictions, utility_full, columns = _enhanced_outputs(
        initial, split["utility"], congestion_config
    )
    utility_base = _without_congestion_context(utility_full, columns, spec)
    utility_truth = utility_raw.traffic_label.to_numpy()
    utility_models, utility_constants, correctness_counts = _fit_utility_models(
        utility_base,
        utility_predictions,
        utility_truth,
        utility_raw.congestion_level.to_numpy(),
        _utility_config(config),
    )

    meta_raw, _, _, _, _, meta_full, _ = _enhanced_outputs(
        initial, split["meta"], congestion_config
    )
    meta_base = _without_congestion_context(meta_full, columns, spec)
    meta_utility = _utility_matrix(utility_models, utility_constants, meta_base)
    meta_x = _ablation_meta_features(meta_base, meta_utility, spec)
    meta_truth = meta_raw.traffic_label.to_numpy()
    meta_levels = meta_raw.congestion_level.to_numpy()
    meta_models = {
        "class_balanced": _fit_meta_model(
            meta_x, meta_truth, meta_levels, config, False, 500
        )
    }
    if spec.use_level_balanced_candidate:
        meta_models["level_class_balanced"] = _fit_meta_model(
            meta_x, meta_truth, meta_levels, config, True, 700
        )

    selection_raw, _, selection_route, _, selection_predictions, selection_full, _ = (
        _enhanced_outputs(initial, split["selection"], congestion_config)
    )
    selection_base = _without_congestion_context(selection_full, columns, spec)
    selection_utility = _utility_matrix(
        utility_models, utility_constants, selection_base
    )
    selection_x = _ablation_meta_features(
        selection_base, selection_utility, spec
    )
    hard_route, _, _, _ = _utility_routes(
        selection_route, selection_utility, 0.0
    )
    row = np.arange(len(selection_raw))
    hard_prediction = selection_predictions[row, hard_route]
    truth = selection_raw.traffic_label.to_numpy()
    levels = selection_raw.congestion_level.to_numpy()
    hard_level_scores = _level_scores(truth, hard_prediction, levels)
    thresholds = (
        config.meta_confidence_candidates
        if spec.use_confidence_hybrid else (0.0,)
    )
    trials = []
    for variant, candidate_model in meta_models.items():
        meta_probability = _aligned_meta_probabilities(candidate_model, selection_x)
        for threshold in thresholds:
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

    if spec.use_confidence_hybrid:
        eligible = [
            trial for trial in trials
            if trial["worst_level_delta_vs_hard"] >= -1e-12
        ]
        if not eligible:
            raise RuntimeError(f"{name}: no non-degrading hybrid candidate")
        selected = max(eligible, key=lambda trial: (
            trial["worst_level_delta_vs_hard"],
            trial["macro_f1"],
            trial["meta_confidence_threshold"],
            -trial["meta_rows"],
        ))
    else:
        selected = max(trials, key=lambda trial: trial["macro_f1"])

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
        "ablation_name": name,
        "ablation_spec": asdict(spec),
        "partition_rows": {
            "expert_raw": len(split["expert"]),
            "router_fit_eligible": len(initial["source_eligible_index"]),
            "utility": len(utility_raw),
            "meta": len(meta_raw),
            "selection": len(selection_raw),
        },
        "leakage_control": {
            "router_scaler_fit_partition": "expert_only",
            "router_scaler_frozen_after_fit": True,
            "utility_meta_selection_are_strictly_later": True,
            "final_experts_refit_on_full_development": True,
        },
    })
    return final


def predict_ablation(model, frame):
    spec = AblationSpec(**model["ablation_spec"])
    raw, _, kroute, _, predictions, full_features, columns = _enhanced_outputs(
        model, frame, model["congestion_config"]
    )
    base = _without_congestion_context(full_features, columns, spec)
    utility = _utility_matrix(
        model["utility_models"], model["utility_constants"], base
    )
    utility_route, _, _, _ = _utility_routes(kroute, utility, 0.0)
    row = np.arange(len(raw))
    hard_prediction = predictions[row, utility_route]
    meta_x = _ablation_meta_features(base, utility, spec)
    probability = _aligned_meta_probabilities(
        model["meta_models"][model["selected_meta_variant"]], meta_x
    )
    prediction, _, confidence, use_meta = _hybrid_prediction(
        probability, hard_prediction, model["selected_meta_confidence"]
    )
    return raw, prediction, confidence, use_meta


def evaluate_split(
    development, test, evaluation_name, evaluation_type, protocol, config,
    scenario="", source="", target="",
):
    full_model = fit_meta_stacker(development, config)
    full_result = predict_full(full_model, test)
    observed = full_result["observed"]
    truth = observed.traffic_label.to_numpy()
    predictions = {
        "S_Meta_Full": full_result["CDR_MLC_meta_stacker"],
        "Utility_Router_Only": full_result["CDR_MLC_utility_router"],
        "Actual_Router_Only": full_result["CDR_MLC_actual_router"],
    }
    prediction_details = {
        "S_Meta_Full": {
            "selected_meta_variant": full_model["selected_meta_variant"],
            "selected_meta_confidence": full_model["selected_meta_confidence"],
            "partition_rows": full_model["partition_rows"],
        }
    }
    route_columns = pd.DataFrame(index=observed.index)
    route_columns["S_Meta_Full_used_meta"] = (
        full_result["routes"]["used_meta_prediction"].astype(bool)
    )
    route_columns["S_Meta_Full_confidence"] = (
        full_result["routes"]["meta_confidence"]
    )

    for name, spec in LEARNED_ABLATIONS.items():
        model = fit_ablation(development, config, name, spec)
        ablation_observed, prediction, confidence, use_meta = predict_ablation(
            model, test
        )
        if frame_identity(ablation_observed) != frame_identity(observed):
            raise RuntimeError(f"{evaluation_name}/{name}: evaluated rows differ")
        predictions[name] = prediction
        route_columns[f"{name}_used_meta"] = pd.Series(
            use_meta, index=observed.index
        )
        route_columns[f"{name}_confidence"] = pd.Series(
            confidence, index=observed.index
        )
        prediction_details[name] = {
            "spec": asdict(spec),
            "selected_meta_variant": model["selected_meta_variant"],
            "selected_meta_confidence": model["selected_meta_confidence"],
            "partition_rows": model["partition_rows"],
        }

    scores = {
        name: metrics(truth, prediction, APPLICATIONS)
        for name, prediction in predictions.items()
    }
    reference = scores["S_Meta_Full"]
    rows = []
    for method in METHOD_ORDER:
        score = scores[method]
        rows.append({
            "protocol": evaluation_name,
            "evaluation_type": evaluation_type,
            "scenario": scenario,
            "source": source,
            "target": target,
            "method": method,
            **{key: score[key] for key in (
                "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
            )},
            "accuracy_change_vs_full": score["accuracy"] - reference["accuracy"],
            "balanced_accuracy_change_vs_full": (
                score["balanced_accuracy"] - reference["balanced_accuracy"]
            ),
            "macro_f1_change_vs_full": score["macro_f1"] - reference["macro_f1"],
        })

    prediction_frame = observed[[
        "timestamp", "source_file", "source_row",
        "traffic_label", "congestion_level",
    ]].copy()
    for method in METHOD_ORDER:
        prediction_frame[f"prediction_{method}"] = predictions[method]
    prediction_frame = prediction_frame.join(route_columns)
    audit = {
        "evaluation_name": evaluation_name,
        "evaluation_type": evaluation_type,
        "scenario": scenario,
        "source": source,
        "target": target,
        "protocol": protocol,
        "development_rows_raw": len(development),
        "test_rows_raw": len(test),
        "evaluated_rows": len(observed),
        "development_identity": frame_identity(development),
        "test_identity": frame_identity(test),
        "evaluated_identity": frame_identity(observed),
        "development_composition": composition(development),
        "test_composition": composition(test),
        "models": prediction_details,
    }
    return rows, prediction_frame, scores, audit


def evaluate_mixed_protocol(data, protocol_name, train_fraction, config):
    development, test, protocol = build_protocol(
        data, protocol_name, train_fraction
    )
    specification = PROTOCOLS[protocol_name]
    if protocol_name == "ALL-80-20":
        source, target = "Low+Medium+High", "Low+Medium+High"
    else:
        source = "+".join(specification["train_levels"])
        target = specification["test_level"]
    return evaluate_split(
        development, test, protocol_name, "mixed_level", protocol, config,
        source=source, target=target,
    )


def evaluate_full_target_scenario(data, scenario, config):
    source, target = SCENARIOS[scenario]
    development = data[data.congestion_level.eq(source)].copy().reset_index(drop=True)
    test = data[data.congestion_level.eq(target)].copy().reset_index(drop=True)
    if development.empty or test.empty:
        raise ValueError(f"scenario {scenario}: empty source or target level")
    development_ids = set(zip(development.source_file, development.source_row))
    test_ids = set(zip(test.source_file, test.source_row))
    overlap = development_ids & test_ids
    if overlap:
        raise RuntimeError(
            f"scenario {scenario}: {len(overlap)} development/test overlaps"
        )
    evaluation_name = f"Scenario-{scenario}-{source}-to-{target}"
    protocol = {
        "kind": "complete_source_level_to_complete_target_level",
        "adaptation_fraction": 0.0,
        "target_test_fraction": 1.0,
        "all_target_rows_are_test_rows": True,
        "target_level_used_in_training_or_selection": False,
    }
    return evaluate_split(
        development, test, evaluation_name, "full_target_scenario",
        protocol, config, scenario=scenario, source=source, target=target,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/s_meta_ablation_leakage_safe",
    )
    parser.add_argument(
        "--protocols", nargs="+", choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=["1", "2", "3"],
        default=["1", "2", "3"],
    )
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument(
        "--congestion-features", nargs="+",
        default=list(DEFAULT_CONGESTION_FEATURES),
    )
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("train-fraction must be in (0,1)")

    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    all_rows = []
    manifest = {
        "config": asdict(config),
        "protocols": args.protocols,
        "scenario_definitions": {
            scenario: SCENARIOS[scenario] for scenario in args.scenarios
        },
        "all_level_train_fraction": args.train_fraction,
        "scenario_target_test_fraction": 1.0,
        "scenario_uses_complete_target_level": True,
        "scenario_adaptation_fraction": 0.0,
        "method_order": METHOD_ORDER,
        "learned_ablation_specs": {
            name: asdict(spec) for name, spec in LEARNED_ABLATIONS.items()
        },
        "same_temporal_partitions_for_all_variants": True,
        "full_model_uses_production_implementation": True,
        "protocol_audits": {},
    }
    for protocol_name in args.protocols:
        rows, predictions, scores, audit = evaluate_mixed_protocol(
            data, protocol_name, args.train_fraction, config
        )
        all_rows.extend(rows)
        protocol_dir = args.output / protocol_name.lower()
        protocol_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(protocol_dir / "ablation_predictions.csv", index=False)
        (protocol_dir / "ablation_metrics.json").write_text(
            json.dumps(scores, indent=2) + "\n", encoding="utf-8"
        )
        (protocol_dir / "ablation_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        manifest["protocol_audits"][protocol_name] = audit

    for scenario in args.scenarios:
        source, target = SCENARIOS[scenario]
        evaluation_name = f"Scenario-{scenario}-{source}-to-{target}"
        rows, predictions, scores, audit = evaluate_full_target_scenario(
            data, scenario, config
        )
        all_rows.extend(rows)
        scenario_dir = args.output / (
            f"scenario_{scenario}_{source.lower()}_to_{target.lower()}"
        )
        scenario_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(
            scenario_dir / "ablation_predictions.csv", index=False
        )
        (scenario_dir / "ablation_metrics.json").write_text(
            json.dumps(scores, indent=2) + "\n", encoding="utf-8"
        )
        (scenario_dir / "ablation_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        manifest["protocol_audits"][evaluation_name] = audit

    summary = pd.DataFrame(all_rows)
    summary["method"] = pd.Categorical(
        summary.method, categories=METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["evaluation_type", "protocol", "method"])
    summary.to_csv(args.output / "s_meta_ablation_summary.csv", index=False)
    (args.output / "s_meta_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(summary)


if __name__ == "__main__":
    main()
