"""Evaluate conservative selective CDR-MLC routing at 0% and 20%."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import (
    SCENARIOS, TIMING, build_calibrated_protocol, fit_rf, metrics, predict_rf,
)
from selective_router_cdr_mlc import (
    SelectiveRouterConfig, fit_selective_router, predict_all,
)


def run_fraction(data, fraction, scenarios, output, config, rf_trees):
    grouped = {}
    for scenario in scenarios:
        source, target = SCENARIOS[scenario]
        grouped.setdefault(source, []).append((scenario, target))
    rows, audits = [], {}
    root = output / f"cal_{int(round(fraction * 100)):02d}"
    root.mkdir(parents=True, exist_ok=True)
    for source_level, targets in grouped.items():
        development, target_tails, calibration = build_calibrated_protocol(
            data, source_level, fraction
        )
        model = fit_selective_router(development, config)
        eligible = development.loc[model["source_eligible_index"]]
        rf = fit_rf(eligible, TIMING, config.random_state, rf_trees)
        source_dir = root / f"source_{source_level.lower()}"
        source_dir.mkdir(parents=True, exist_ok=True)
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
        audits[source_level] = audit
        (source_dir / "selective_router_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        for scenario, target_level in targets:
            result = predict_all(model, target_tails[target_level])
            observed = result["observed"]
            truth = observed.traffic_label.to_numpy()
            predictions = {
                key: value for key, value in result.items() if key.startswith("CDR_")
            }
            predictions["RF_expert_inputs"] = predict_rf(rf, observed)
            scenario_dir = root / (
                f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
            )
            scenario_dir.mkdir(parents=True, exist_ok=True)
            details = {}
            for method, prediction in predictions.items():
                value = metrics(truth, prediction, APPLICATIONS)
                details[method] = value
                rows.append({
                    "adaptation_fraction": fraction, "scenario": scenario,
                    "source": source_level, "target": target_level, "method": method,
                    **{key: value[key] for key in (
                        "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
                    )},
                })
            frame = observed[
                ["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]
            ].copy()
            for method, prediction in predictions.items():
                frame[f"prediction_{method}"] = prediction
            frame = frame.join(result["routes"])
            frame.to_csv(scenario_dir / "predictions.csv", index=False)
            (scenario_dir / "metrics.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8"
            )
    return rows, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/selective_router_sweep")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--correction-trees", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--min-expert-confidence-gain", type=float, default=.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = SelectiveRouterConfig(
        window=args.window, expert_trees=args.expert_trees,
        correction_trees=args.correction_trees, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        min_expert_confidence_gain=args.min_expert_confidence_gain,
        random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(TIMING))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for fraction in args.fractions:
        result, detail = run_fraction(
            data, fraction, args.scenarios, args.output, config, args.rf_trees
        )
        rows.extend(result)
        audits[str(fraction)] = detail
    summary = pd.DataFrame(rows).sort_values(
        ["adaptation_fraction", "scenario", "method"]
    )
    summary.to_csv(args.output / "selective_router_summary.csv", index=False)
    manifest = {
        "config": asdict(config),
        "fractions": args.fractions,
        "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Full-source KMeans is unsupervised. Correction targets and threshold selection "
            "use separate chronological source partitions. Test labels are oracle-only."
        ),
    }
    (args.output / "selective_router_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
