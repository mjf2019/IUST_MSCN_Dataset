"""Evaluate leakage-controlled meta-stacked CDR-MLC in three scenarios."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import SCENARIOS, TIMING, build_calibrated_protocol, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc import MetaStackConfig, fit_meta_stacker, predict_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/meta_stacked_router")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=150)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = MetaStackConfig(
        window=args.window, congestion_window=args.congestion_window,
        congestion_features=tuple(args.congestion_features),
        expert_trees=args.expert_trees, utility_trees=args.utility_trees,
        meta_trees=args.meta_trees, random_state=args.seed,
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
            model = fit_meta_stacker(development, config)
            eligible = development.loc[model["source_eligible_index"]]
            rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
            audit = {
                "development_rows_raw": len(development),
                "partition_rows": model["partition_rows"],
                "utility_correctness_counts": model["utility_correctness_counts"],
                "selected_meta_confidence": model["selected_meta_confidence"],
                "selected_meta_variant": model["selected_meta_variant"],
                "meta_selection_trials": model["meta_selection_trials"],
                "calibration": calibration,
            }
            audits[f"{fraction}:{source_level}"] = audit
            source_dir = fraction_root / f"source_{source_level.lower()}"
            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "meta_stacker_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
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
    summary.to_csv(args.output / "meta_stacked_summary.csv", index=False)
    manifest = {
        "config": asdict(config), "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Every capture is split chronologically into disjoint expert, utility, meta, and "
            "selection partitions. Test labels are diagnostic only."
        ),
    }
    (args.output / "meta_stacked_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
