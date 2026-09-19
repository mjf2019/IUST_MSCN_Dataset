"""Evaluate actual, learned, and oracle CDR-MLC routing at 0% and 20%."""
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
from learned_router_cdr_mlc import (
    LearnedRouterConfig, fit_learned_router, predict_all,
)


def run_fraction(data, fraction, scenarios, output, config, rf_trees):
    grouped = {}
    for scenario in scenarios:
        source, target = SCENARIOS[scenario]
        grouped.setdefault(source, []).append((scenario, target))
    rows, audits = [], {}
    fraction_dir = output / f"cal_{int(round(fraction * 100)):02d}"
    fraction_dir.mkdir(parents=True, exist_ok=True)
    for source_level, targets in grouped.items():
        development, target_tails, calibration = build_calibrated_protocol(
            data, source_level, fraction
        )
        model = fit_learned_router(development, config)
        eligible = development.loc[model["source_eligible_index"]]
        rf = fit_rf(eligible, TIMING, config.random_state, rf_trees)
        source_dir = fraction_dir / f"source_{source_level.lower()}"
        source_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model["full_source_cluster_counts"]).to_csv(
            source_dir / "cluster_counts.csv", index=False
        )
        audit = {
            "development_rows_raw": len(development),
            "full_source_window_rows": model["full_source_rows"],
            "gate_training_rows": model["gate_training_rows"],
            "gate_target_counts": model["gate_target_counts"],
            "gate_kmeans_agreement": model["gate_kmeans_agreement"],
            "calibration": calibration,
        }
        audits[source_level] = audit
        (source_dir / "learned_router_audit.json").write_text(
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
            scenario_dir = fraction_dir / (
                f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
            )
            scenario_dir.mkdir(parents=True, exist_ok=True)
            details = {}
            for method, prediction in predictions.items():
                value = metrics(truth, prediction, APPLICATIONS)
                details[method] = value
                rows.append({
                    "adaptation_fraction": fraction,
                    "scenario": scenario,
                    "source": source_level,
                    "target": target_level,
                    "method": method,
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
    parser.add_argument("--output", type=Path, default=root / "outputs/learned_router_sweep")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, .20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--gate-trees", type=int, default=50)
    parser.add_argument("--gate-max-depth", type=int, default=8)
    parser.add_argument("--gate-min-samples-leaf", type=int, default=20)
    parser.add_argument("--gate-min-confidence", type=float, default=.45)
    parser.add_argument("--router-train-fraction", type=float, default=.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(not 0 <= value < 1 for value in args.fractions):
        parser.error("fractions must be in [0,1)")
    config = LearnedRouterConfig(
        window=args.window, expert_trees=args.expert_trees,
        gate_trees=args.gate_trees, gate_max_depth=args.gate_max_depth,
        gate_min_samples_leaf=args.gate_min_samples_leaf,
        gate_min_confidence=args.gate_min_confidence,
        router_train_fraction=args.router_train_fraction,
        random_state=args.seed,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_dataset(args.data_dir, tuple(TIMING))
    audit.to_csv(args.output / "input_audit.csv", index=False)
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
    summary.to_csv(args.output / "learned_router_summary.csv", index=False)
    manifest = {
        "config": asdict(config),
        "fractions": args.fractions,
        "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "audits": audits,
        "leakage_control": (
            "Gate targets use chronological holdout predictions from experts that did not "
            "train on those records. Test labels are used only by the oracle diagnostic."
        ),
        "limitations": [
            "KMeans is frozen after the initial chronological expert-training prefix.",
            "Experts are refit on full development data with frozen KMeans identities.",
            "The 20% calibration protocol is not unseen cross-congestion.",
        ],
    }
    (args.output / "learned_router_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
