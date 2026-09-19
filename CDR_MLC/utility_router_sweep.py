"""Evaluate expected-correctness routing on all CDR-MLC scenarios."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import SCENARIOS, TIMING, build_calibrated_protocol, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from utility_router_cdr_mlc import UtilityRouterConfig, fit_utility_router, predict_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/utility_router_sweep")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--utility-trees", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = UtilityRouterConfig(
        window=args.window, congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees, utility_trees=args.utility_trees,
        max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
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
        fraction_root = args.output / f"cal_{int(round(fraction * 100)):02d}"
        fraction_root.mkdir(parents=True, exist_ok=True)
        for source_level, targets in grouped.items():
            development, tails, calibration = build_calibrated_protocol(data, source_level, fraction)
            model = fit_utility_router(development, config)
            eligible = development.loc[model["source_eligible_index"]]
            rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
            audit = {
                "development_rows_raw": len(development),
                "utility_training_rows": model["utility_training_rows"],
                "utility_correctness_counts": model["utility_correctness_counts"],
                "utility_kmeans_error_rate": model["utility_kmeans_error_rate"],
                "selected_utility_gain": model["selected_utility_gain"],
                "utility_gain_trials": model["utility_gain_trials"],
                "calibration": calibration,
            }
            audits[f"{fraction}:{source_level}"] = audit
            source_dir = fraction_root / f"source_{source_level.lower()}"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "utility_router_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            for scenario, target_level in targets:
                result = predict_all(model, tails[target_level])
                observed = result["observed"]
                truth = observed.traffic_label.to_numpy()
                predictions = {key: value for key, value in result.items() if key.startswith("CDR_")}
                predictions["RF_expert_inputs"] = predict_rf(rf, observed)
                scenario_dir = fraction_root / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
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
    summary.to_csv(args.output / "utility_router_summary.csv", index=False)
    manifest = {
        "config": asdict(config), "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Utility targets use chronological development partitions only. Test labels are "
            "used solely to report metrics and the diagnostic oracle ceiling."
        ),
    }
    (args.output / "utility_router_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
