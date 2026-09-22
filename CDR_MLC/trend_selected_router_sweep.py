"""Evaluate training-only trend-selected KMeans features across six transfers.

Scenarios:
  1: Low -> Medium
  2: Low -> High
  3: Medium -> High
  LM-H: Low + Medium -> High
  LH-M: Low + High -> Medium
  MH-L: Medium + High -> Low

The target level is never used for fitting, selection, preprocessing, or
threshold choice. All deployable methods are evaluated on the same target rows.
Oracle-router outputs are retained only as non-deployable diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import (
    TIMING, fit_rf, metrics, predict_rf,
)
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc import (
    MetaStackConfig, fit_meta_stacker as fit_legacy_meta,
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


def frame_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    )[["sequence_id", "source_file", "source_row"]]
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def as_series(result: dict, prefix: str) -> dict[str, pd.Series]:
    index = result["observed"].index
    return {
        f"{prefix}{name}": pd.Series(prediction, index=index)
        for name, prediction in result.items()
        if name.startswith("CDR_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/trend_selected_router_all_scenarios",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS),
        default=list(SCENARIOS),
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

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()

    trend_router_config = TrendSelectedRouterConfig(
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
        target = data[data.congestion_level.eq(target_level)].copy()
        if source.empty or target.empty:
            raise ValueError(f"{scenario}: empty source or target")

        legacy_model = fit_legacy_meta(source, config)
        # The standalone fixed CDR-MLC row was identical to
        # Legacy_CDR_MLC_actual_router. Reuse the legacy model's eligible
        # source rows for both RF baselines and avoid fitting it twice.
        eligible_source = source.loc[
            legacy_model["source_eligible_index"]
        ]
        rf_clean_valid = fit_rf(
            eligible_source, (), args.seed, args.rf_trees
        )
        rf_expert_inputs = fit_rf(
            eligible_source, TIMING, args.seed, args.rf_trees
        )

        safe_model = fit_safe_meta(source, config)
        trend_router_model = fit_trend_selected_meta_stacker(
            source, trend_router_config
        )
        legacy_result = predict_legacy_meta(legacy_model, target)
        safe_result = predict_safe_meta(safe_model, target)
        trend_router_result = predict_trend_selected(
            trend_router_model, target
        )

        prediction_series = {
            **as_series(legacy_result, "Legacy_"),
            **as_series(safe_result, "Safe_"),
            **as_series(trend_router_result, "TrendSelected_"),
        }
        common = sorted(set.intersection(*(
            set(series.index) for series in prediction_series.values()
        )))
        if not common:
            raise ValueError(f"{scenario}: no common target rows")
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

        scenario_dir = args.output / f"scenario_{scenario.lower()}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        detailed = {}
        for method, prediction in predictions.items():
            score = metrics(truth, prediction, APPLICATIONS)
            detailed[method] = score
            rows.append({
                "scenario": scenario,
                "train_levels": "+".join(source_levels),
                "test_level": target_level,
                "method": method,
                **{key: score[key] for key in (
                    "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "weighted_f1",
                )},
            })

        prediction_frame = observed[[
            "timestamp", "traffic_label", "congestion_level"
        ]].copy()
        for method, prediction in predictions.items():
            prediction_frame[f"prediction_{method}"] = prediction
        prediction_frame.to_csv(
            scenario_dir / "predictions.csv", index=False
        )
        (scenario_dir / "metrics.json").write_text(
            json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
        )
        audits[scenario] = {
            "train_levels": list(source_levels),
            "test_level": target_level,
            "raw_train_rows": len(source),
            "fixed_cdr_eligible_train_rows": len(eligible_source),
            "raw_test_rows": len(target),
            "evaluated_common_test_rows": len(observed),
            "raw_test_identity": frame_identity(target),
            "evaluated_test_identity": frame_identity(observed),
            "fixed_cluster_counts": legacy_model["cluster_counts"],
            "legacy_partition_rows": legacy_model["partition_rows"],
            "safe_partition_rows": safe_model["partition_rows"],
            "trend_router_partition_rows": (
                trend_router_model["partition_rows"]
            ),
            "selected_router_features": list(
                trend_router_model["selected_router_features"]
            ),
            "router_trend_ranking": (
                trend_router_model["router_trend_ranking"]
            ),
            "trend_router_cluster_counts": (
                trend_router_model["cluster_counts"]
            ),
            "trend_router_full_source_cluster_counts": (
                trend_router_model["full_source_cluster_counts"]
            ),
            "trend_router_leakage_control": (
                trend_router_model["leakage_control"]
            ),
            "legacy_selected_meta_variant": (
                legacy_model["selected_meta_variant"]
            ),
            "legacy_selected_meta_confidence": (
                legacy_model["selected_meta_confidence"]
            ),
            "safe_selected_meta_variant": (
                safe_model["selected_meta_variant"]
            ),
            "safe_selected_meta_confidence": (
                safe_model["selected_meta_confidence"]
            ),
            "trend_router_selected_meta_variant": (
                trend_router_model["selected_meta_variant"]
            ),
            "trend_router_selected_meta_confidence": (
                trend_router_model["selected_meta_confidence"]
            ),
        }

    summary = pd.DataFrame(rows).sort_values(["scenario", "method"])
    summary.to_csv(
        args.output / "trend_selected_router_summary.csv", index=False
    )
    manifest = {
        "protocol": "single_or_two_source_levels_to_unseen_target_level",
        "target_level_used_in_training_or_selection": False,
        "scenarios": {
            key: {
                "train_levels": list(SCENARIOS[key][0]),
                "test_level": SCENARIOS[key][1],
            }
            for key in args.scenarios
        },
        "meta_config": asdict(config),
        "trend_selected_router_config": asdict(trend_router_config),
        "trend_selected_router_protocol": {
            "selection_partition": "earliest expert partition only",
            "selection_labels": "none",
            "trend_window": 3,
            "selected_features": 3,
            "replaces_paper_fixed_router_features": [
                "TcpRtt", "SynAck", "AckDat"
            ],
            "selected_features_used_by": [
                "StandardScaler", "KMeans router"
            ],
            "selected_features_excluded_from_experts": True,
            "selected_features_frozen_for_test": True,
            "utility_and_meta_pipeline": "unchanged leakage-safe version",
        },
        "rf_clean_valid": {
            "trees": args.rf_trees,
            "features": "all reviewed Clean-Valid classifier fields",
        },
        "rf_expert_inputs": {
            "trees": args.rf_trees,
            "excluded_router_features": TIMING,
        },
        "common_target_rows_for_all_methods": True,
        "oracle_is_diagnostic_not_deployable": True,
        "audits": audits,
    }
    (args.output / "trend_selected_router_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
