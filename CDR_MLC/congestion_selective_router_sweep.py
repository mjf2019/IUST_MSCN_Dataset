"""Evaluate congestion-enriched selective routing with the original experts."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import SCENARIOS, TIMING, build_calibrated_protocol, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from congestion_selective_router_cdr_mlc import CongestionSelectiveConfig, fit_congestion_selective_router, predict_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/congestion_selective_router")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--correction-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=15)
    parser.add_argument("--min-expert-confidence-gain", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = CongestionSelectiveConfig(
        window=args.window, congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees, correction_trees=args.correction_trees,
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
        min_expert_confidence_gain=args.min_expert_confidence_gain,
        random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    rows, audits = [], {}
    for fraction in args.fractions:
        grouped = {}
        for scenario in args.scenarios:
            source, target = SCENARIOS[scenario]
            grouped.setdefault(source, []).append((scenario, target))
        root_fraction = args.output / f"cal_{int(round(fraction * 100)):02d}"
        root_fraction.mkdir(parents=True, exist_ok=True)
        for source_level, targets in grouped.items():
            development, tails, calibration = build_calibrated_protocol(data, source_level, fraction)
            model = fit_congestion_selective_router(development, config)
            eligible = development.loc[model["source_eligible_index"]]
            rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
            audit = {
                "development_rows_raw": len(development),
                "correction_training_rows": model["correction_training_rows"],
                "threshold_selection_rows": model["threshold_selection_rows"],
                "error_rate_correction_partition": model["error_rate_correction_partition"],
                "oracle_target_counts": model["oracle_target_counts"],
                "selected_threshold": model["selected_threshold"],
                "threshold_trials": model["threshold_trials"],
                "calibration": calibration,
            }
            audits[f"{fraction}:{source_level}"] = audit
            source_dir = root_fraction / f"source_{source_level.lower()}"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "congestion_selective_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            for scenario, target_level in targets:
                result = predict_all(model, tails[target_level])
                observed = result["observed"]
                truth = observed.traffic_label.to_numpy()
                predictions = {key: value for key, value in result.items() if key.startswith("CDR_")}
                predictions["RF_expert_inputs"] = predict_rf(rf, observed)
                scenario_dir = root_fraction / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
                scenario_dir.mkdir(parents=True, exist_ok=True)
                detail = {}
                for method, prediction in predictions.items():
                    score = metrics(truth, prediction, APPLICATIONS)
                    detail[method] = score
                    rows.append({
                        "adaptation_fraction": fraction, "scenario": scenario,
                        "source": source_level, "target": target_level, "method": method,
                        **{key: score[key] for key in ("n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
                    })
                frame = observed[["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]].copy()
                for method, prediction in predictions.items():
                    frame[f"prediction_{method}"] = prediction
                frame.join(result["routes"]).to_csv(scenario_dir / "predictions.csv", index=False)
                (scenario_dir / "metrics.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")

    summary = pd.DataFrame(rows).sort_values(["adaptation_fraction", "scenario", "method"])
    summary.to_csv(args.output / "congestion_selective_summary.csv", index=False)
    manifest = {
        "config": asdict(config), "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Original KMeans/expert bank is unchanged. Congestion descriptors are causal and "
            "only train the correction layer on chronological non-test partitions."
        ),
    }
    (args.output / "congestion_selective_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
