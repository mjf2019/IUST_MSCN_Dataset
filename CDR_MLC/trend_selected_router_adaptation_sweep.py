"""Adaptation sweep for the trend-selected CDR-MLC router.

For each transfer scenario, the final target tail is reserved once as a fixed
test set. A chronological target prefix (0/10/20/30 percent by default) is
added to the complete source-level training data. The middle target region is
unused. Consequently every adaptation fraction is evaluated on exactly the
same target rows and no target-test row participates in fitting, feature
selection, preprocessing or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import TIMING, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc import (
    MetaStackConfig,
    fit_meta_stacker as fit_legacy_meta,
    predict_all as predict_legacy_meta,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    fit_meta_stacker as fit_safe_meta,
    predict_all as predict_safe_meta,
)
from trend_selected_router_meta_stacker import (
    TrendSelectedRouterConfig,
    fit_trend_selected_meta_stacker,
    predict_all as predict_trend_selected,
)


SCENARIOS = {
    "1": (("Low",), "Medium"),
    "2": (("Low",), "High"),
    "3": (("Medium",), "High"),
    "LM-H": (("Low", "Medium"), "High"),
    "LH-M": (("Low", "High"), "Medium"),
    "MH-L": (("Medium", "High"), "Low"),
}


def frame_identity(frame):
    ordered = frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    )[["sequence_id", "source_file", "source_row"]]
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def fixed_test_and_adaptation(target, adaptation_fraction, test_fraction):
    """Return a chronological prefix and a fraction-invariant target tail."""
    adaptation, test, audit = [], [], []
    for sequence_id, group in target.groupby("sequence_id", sort=False):
        group = group.sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        n = len(group)
        adaptation_end = int(n * adaptation_fraction)
        test_start = int(n * (1.0 - test_fraction))
        if not 0 < test_start < n:
            raise ValueError(
                f"{sequence_id}: invalid fixed test split for {n} rows"
            )
        if adaptation_end > test_start:
            raise ValueError(
                f"{sequence_id}: adaptation prefix overlaps fixed test tail"
            )
        if adaptation_end:
            adaptation.append(group.iloc[:adaptation_end].copy())
        test.append(group.iloc[test_start:].copy())
        audit.append({
            "sequence_id": sequence_id,
            "raw_target_rows": n,
            "adaptation_rows": adaptation_end,
            "unused_middle_rows": test_start - adaptation_end,
            "fixed_test_rows": n - test_start,
        })
    empty = target.iloc[:0].copy()
    return (
        pd.concat(adaptation, ignore_index=True)
        if adaptation else empty,
        pd.concat(test, ignore_index=True),
        audit,
    )


def chronological_prefix(frame, fraction):
    """Take the same fraction from the beginning of every capture."""
    pieces, audit = [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        end = int(len(group) * fraction)
        if end:
            pieces.append(group.iloc[:end].copy())
        audit.append({
            "sequence_id": sequence_id,
            "raw_rows": len(group),
            "adaptation_rows": end,
        })
    empty = frame.iloc[:0].copy()
    return (
        pd.concat(pieces, ignore_index=True) if pieces else empty,
        audit,
    )


def as_series(result, prefix):
    index = result["observed"].index
    return {
        f"{prefix}{name}": pd.Series(prediction, index=index)
        for name, prediction in result.items()
        if name.startswith("CDR_")
    }


def fraction_tag(value):
    return f"{int(round(100 * value)):02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/trend_selected_router_adaptation",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS),
        default=list(SCENARIOS),
    )
    parser.add_argument(
        "--adaptation-fractions", nargs="+", type=float,
        default=[0.0, 0.10, 0.20, 0.30],
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--adaptation-scope",
        choices=("target", "all-missing"),
        default="target",
        help=(
            "target: add only the evaluated target prefix; "
            "all-missing: add prefixes from every level not in source_levels"
        ),
    )
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument(
        "--congestion-features", nargs="+",
        default=list(DEFAULT_CONGESTION_FEATURES),
    )
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fractions = tuple(dict.fromkeys(args.adaptation_fractions))
    if not fractions or any(value < 0 or value >= 1 for value in fractions):
        raise ValueError("adaptation fractions must be in [0,1)")
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be in (0,1)")
    if any(value + args.test_fraction > 1 for value in fractions):
        raise ValueError(
            "adaptation-fraction + test-fraction must not exceed 1"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([
        *TIMING, *args.congestion_features,
    ]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    meta_config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()
    trend_config = TrendSelectedRouterConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        router_candidates=tuple(args.congestion_features),
        context_features=tuple(args.congestion_features),
        selected_feature_count=3,
        trend_window=3,
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()

    rows, audits = [], {}
    for scenario in args.scenarios:
        source_levels, target_level = SCENARIOS[scenario]
        source = data[data.congestion_level.isin(source_levels)].copy()
        raw_target = data[data.congestion_level.eq(target_level)].copy()
        if source.empty or raw_target.empty:
            raise ValueError(f"{scenario}: empty source or target")

        scenario_test_identity = None
        for fraction in fractions:
            target_calibration, target, split_audit = (
                fixed_test_and_adaptation(
                    raw_target, fraction, args.test_fraction
                )
            )
            calibration_parts = [target_calibration]
            calibration_levels = [target_level]
            auxiliary_audits = {}
            if args.adaptation_scope == "all-missing":
                for level in ("Low", "Medium", "High"):
                    if level in source_levels or level == target_level:
                        continue
                    level_frame = data[
                        data.congestion_level.eq(level)
                    ].copy()
                    level_prefix, level_audit = chronological_prefix(
                        level_frame, fraction
                    )
                    calibration_parts.append(level_prefix)
                    calibration_levels.append(level)
                    auxiliary_audits[level] = level_audit
            calibration = pd.concat(
                calibration_parts, ignore_index=True
            )
            if scenario_test_identity is None:
                scenario_test_identity = frame_identity(target)
            elif frame_identity(target) != scenario_test_identity:
                raise RuntimeError(
                    f"{scenario}: fixed test changed across fractions"
                )
            development = pd.concat(
                [source, calibration], ignore_index=True
            )

            legacy_model = fit_legacy_meta(development, meta_config)
            eligible_development = development.loc[
                legacy_model["source_eligible_index"]
            ]
            rf_clean_valid = fit_rf(
                eligible_development, (), args.seed, args.rf_trees
            )
            rf_expert_inputs = fit_rf(
                eligible_development, TIMING, args.seed, args.rf_trees
            )
            safe_model = fit_safe_meta(development, meta_config)
            trend_model = fit_trend_selected_meta_stacker(
                development, trend_config
            )

            legacy_result = predict_legacy_meta(legacy_model, target)
            safe_result = predict_safe_meta(safe_model, target)
            trend_result = predict_trend_selected(
                trend_model, target
            )
            prediction_series = {
                **as_series(legacy_result, "Legacy_"),
                **as_series(safe_result, "Safe_"),
                **as_series(trend_result, "TrendSelected_"),
            }
            common = sorted(set.intersection(*(
                set(series.index)
                for series in prediction_series.values()
            )))
            if not common:
                raise ValueError(
                    f"{scenario}/{fraction}: no common target rows"
                )
            observed = target.loc[common].copy()
            truth = observed.traffic_label.to_numpy()
            predictions = {
                method: series.loc[common].to_numpy()
                for method, series in prediction_series.items()
            }
            predictions["RF_Clean_Valid"] = predict_rf(
                rf_clean_valid, observed
            )
            predictions["RF_Expert_Inputs"] = predict_rf(
                rf_expert_inputs, observed
            )

            run_key = (
                f"{scenario}_adapt_{fraction_tag(fraction)}"
            )
            run_dir = args.output / f"scenario_{run_key.lower()}"
            run_dir.mkdir(parents=True, exist_ok=True)
            detailed = {}
            for method, prediction in predictions.items():
                score = metrics(truth, prediction, APPLICATIONS)
                detailed[method] = score
                rows.append({
                    "adaptation_fraction": fraction,
                    "fixed_test_fraction": args.test_fraction,
                    "scenario": scenario,
                    "train_levels": "+".join(source_levels),
                    "adaptation_scope": args.adaptation_scope,
                    "adaptation_levels": "+".join(calibration_levels),
                    "test_level": target_level,
                    "development_rows": len(development),
                    "adaptation_rows": len(calibration),
                    "n": score["n"],
                    "method": method,
                    "accuracy": score["accuracy"],
                    "balanced_accuracy": score[
                        "balanced_accuracy"
                    ],
                    "macro_f1": score["macro_f1"],
                    "weighted_f1": score["weighted_f1"],
                })

            prediction_frame = observed[[
                "timestamp", "traffic_label", "congestion_level"
            ]].copy()
            for method, prediction in predictions.items():
                prediction_frame[f"prediction_{method}"] = prediction
            prediction_frame.to_csv(
                run_dir / "predictions.csv", index=False
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(detailed, indent=2) + "\n",
                encoding="utf-8",
            )
            audits[run_key] = {
                "scenario": scenario,
                "source_levels": list(source_levels),
                "adaptation_scope": args.adaptation_scope,
                "adaptation_levels": calibration_levels,
                "adaptation_fraction": fraction,
                "fixed_test_fraction": args.test_fraction,
                "raw_source_rows": len(source),
                "adaptation_rows": len(calibration),
                "development_rows": len(development),
                "raw_target_rows": len(raw_target),
                "fixed_test_rows_before_windowing": len(target),
                "evaluated_common_test_rows": len(observed),
                "fixed_test_identity": scenario_test_identity,
                "target_split_by_sequence": split_audit,
                "auxiliary_adaptation_by_level": auxiliary_audits,
                "selected_router_features": list(
                    trend_model["selected_router_features"]
                ),
                "router_trend_ranking": trend_model[
                    "router_trend_ranking"
                ],
                "trend_router_cluster_counts": trend_model[
                    "full_source_cluster_counts"
                ],
                "trend_router_partition_rows": trend_model[
                    "partition_rows"
                ],
                "trend_router_leakage_control": trend_model[
                    "leakage_control"
                ],
                "legacy_selected_meta_variant": legacy_model[
                    "selected_meta_variant"
                ],
                "legacy_selected_meta_confidence": legacy_model[
                    "selected_meta_confidence"
                ],
                "safe_selected_meta_variant": safe_model[
                    "selected_meta_variant"
                ],
                "safe_selected_meta_confidence": safe_model[
                    "selected_meta_confidence"
                ],
                "trend_selected_meta_variant": trend_model[
                    "selected_meta_variant"
                ],
                "trend_selected_meta_confidence": trend_model[
                    "selected_meta_confidence"
                ],
            }

    summary = pd.DataFrame(rows).sort_values([
        "scenario", "adaptation_fraction", "method",
    ])
    summary.to_csv(
        args.output / "trend_selected_adaptation_summary.csv",
        index=False,
    )
    manifest = {
        "protocol": (
            "fixed_target_tail_with_chronological_target_prefix_adaptation"
        ),
        "adaptation_fractions": list(fractions),
        "adaptation_scope": args.adaptation_scope,
        "fixed_test_fraction": args.test_fraction,
        "target_test_is_identical_across_fractions": True,
        "target_middle_region_is_unused": True,
        "target_test_used_in_training_or_selection": False,
        "scenarios": {
            key: {
                "source_levels": list(SCENARIOS[key][0]),
                "target_adaptation_and_test_level": SCENARIOS[key][1],
                "additional_adaptation_levels": (
                    [
                        level for level in ("Low", "Medium", "High")
                        if level not in SCENARIOS[key][0]
                        and level != SCENARIOS[key][1]
                    ]
                    if args.adaptation_scope == "all-missing"
                    else []
                ),
            }
            for key in args.scenarios
        },
        "meta_config": asdict(meta_config),
        "trend_selected_router_config": asdict(trend_config),
        "rf_clean_valid": {
            "trees": args.rf_trees,
            "features": "all reviewed Clean-Valid classifier fields",
        },
        "rf_expert_inputs": {
            "trees": args.rf_trees,
            "excluded_features": TIMING,
        },
        "common_fixed_test_rows_for_all_methods": True,
        "oracle_is_diagnostic_not_deployable": True,
        "audits": audits,
    }
    (args.output / "trend_selected_adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
