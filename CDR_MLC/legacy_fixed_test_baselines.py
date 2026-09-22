"""Fixed-test paper baselines for the small legacy Clean-Valid dataset.

Evaluates the original fixed CDR-MLC and both RF references on exactly the same
scenario-specific immutable test tails used by the leakage-safe meta stacker.
Adaptive feature/window selection is intentionally excluded: some legacy
application captures cannot populate three nonempty validation clusters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import (
    SCENARIOS, TIMING, fit_fixed_cdr, fit_rf, metrics,
    predict_fixed_cdr, predict_rf,
)
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_fixed_test_sweep_leakage_safe import (
    build_scenario_fixed_test_protocol, frame_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Legacy_Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/legacy_fixed_test_baselines",
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"],
                        default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be in (0,1)")
    if any(value < 0 or value > 1 - args.test_fraction for value in args.fractions):
        parser.error("fractions must not overlap the fixed test tail")

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *DEFAULT_CONGESTION_FEATURES]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    rows, audits, expected_identity = [], {}, {}
    for fraction in args.fractions:
        fraction_root = args.output / f"cal_{int(round(fraction * 100)):02d}"
        fraction_root.mkdir(parents=True, exist_ok=True)
        for scenario in args.scenarios:
            source_level, target_level = SCENARIOS[scenario]
            development, fixed_test, protocol_audit = (
                build_scenario_fixed_test_protocol(
                    data, source_level, target_level, fraction,
                    args.test_fraction,
                )
            )
            identity = frame_identity(fixed_test)
            previous = expected_identity.setdefault(target_level, identity)
            if identity != previous:
                raise RuntimeError(f"fixed test identity changed for {target_level}")

            fixed = fit_fixed_cdr(
                development, args.window, args.seed, args.expert_trees
            )
            eligible = development.loc[fixed["source_eligible_index"]]
            rf_all = fit_rf(eligible, (), args.seed, args.rf_trees)
            rf_expert_inputs = fit_rf(
                eligible, TIMING, args.seed, args.rf_trees
            )

            fixed_prediction = predict_fixed_cdr(fixed, fixed_test)
            observed = fixed_test.loc[fixed_prediction.index].copy()
            evaluated_identity = frame_identity(observed)
            truth = observed.traffic_label.to_numpy()
            predictions = {
                "CDR_MLC": fixed_prediction.to_numpy(),
                "RF_all_clean_valid": predict_rf(rf_all, observed),
                "RF_expert_inputs": predict_rf(rf_expert_inputs, observed),
            }

            scenario_dir = fraction_root / (
                f"scenario_{scenario}_{source_level.lower()}_to_"
                f"{target_level.lower()}"
            )
            scenario_dir.mkdir(parents=True, exist_ok=True)
            detailed = {}
            for method, prediction in predictions.items():
                score = metrics(truth, prediction, APPLICATIONS)
                detailed[method] = score
                rows.append({
                    "adaptation_fraction": fraction,
                    "fixed_test_fraction": args.test_fraction,
                    "scenario": scenario,
                    "source": source_level,
                    "target": target_level,
                    "method": method,
                    **{key: score[key] for key in (
                        "n", "accuracy", "balanced_accuracy",
                        "macro_f1", "weighted_f1",
                    )},
                })
            pd.DataFrame({
                "timestamp": observed.timestamp.to_numpy(),
                "traffic_label": truth,
                "congestion_level": observed.congestion_level.to_numpy(),
                **{f"prediction_{key}": value
                   for key, value in predictions.items()},
            }).to_csv(scenario_dir / "predictions.csv", index=False)
            (scenario_dir / "metrics.json").write_text(
                json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
            )
            audits[f"{fraction}:{scenario}"] = {
                "adaptation_fraction": fraction,
                "scenario": scenario,
                "source": source_level,
                "target": target_level,
                "development_rows_raw": len(development),
                "eligible_development_rows": len(eligible),
                "fixed_test_rows_raw": len(fixed_test),
                "evaluated_rows": len(observed),
                "fixed_test_identity": identity,
                "evaluated_identity": evaluated_identity,
                "protocol": protocol_audit,
                "cluster_counts": fixed["cluster_counts"],
            }

    summary = pd.DataFrame(rows).sort_values(
        ["adaptation_fraction", "scenario", "method"]
    )
    summary.to_csv(args.output / "legacy_fixed_test_baseline_summary.csv",
                   index=False)
    manifest = {
        "protocol": "scenario_isolated_immutable_fixed_test",
        "fractions": args.fractions,
        "test_fraction": args.test_fraction,
        "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
        "window": args.window,
        "expert_trees": args.expert_trees,
        "rf_trees": args.rf_trees,
        "seed": args.seed,
        "audits": audits,
        "excluded_extension": {
            "method": "Adaptive_CDR_MLC source-only selector",
            "reason": (
                "At least the legacy SFTP validation capture cannot populate "
                "three nonempty clusters for any candidate configuration. "
                "Threshold relaxation cannot repair an empty cluster."
            ),
        },
    }
    (args.output / "legacy_fixed_test_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
