"""Build and evaluate the legacy Clean-Valid dataset without touching raw data.

This is an orchestration script only. It runs the existing paper comparison,
legacy meta-stacker, and strict leakage-safe meta-stacker with fixed arguments,
then combines their summary CSV files. Use --only-clean to stop after curation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(command: list[str]) -> None:
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    repository = root.parent
    parser.add_argument(
        "--source", type=Path,
        default=repository / "DATASETS/CDR-MLC/scale_1/Short",
    )
    parser.add_argument(
        "--clean-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Legacy_Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/legacy_scale_1_experiments",
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"],
                        default=["1", "2", "3"])
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-clean", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--only-clean", action="store_true")
    args = parser.parse_args()

    if args.skip_clean and args.only_clean:
        parser.error("--skip-clean and --only-clean cannot be used together")
    if any(value < 0 or value > 1 - args.test_fraction for value in args.fractions):
        parser.error("fractions must be nonnegative and must not overlap test tail")

    args.output.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    if not args.skip_clean:
        command = [
            py, str(root / "build_legacy_clean_valid.py"),
            "--source", str(args.source),
            "--output", str(args.clean_dir),
        ]
        if args.overwrite_clean:
            command.append("--overwrite")
        run(command)
    if args.only_clean:
        return
    manifest = args.clean_dir / "clean_valid_manifest.json"
    if not manifest.exists():
        parser.error(f"missing {manifest}; run without --skip-clean first")

    scenario_args = ["--scenarios", *args.scenarios]

    # Paper-aligned family: fixed CDR-MLC, RF references, Adaptive CDR-MLC,
    # and Sensitive CDR-MLC. compare_clean_valid accepts one fraction per run.
    for fraction in args.fractions:
        tag = int(round(fraction * 100))
        run([
            py, str(root / "compare_clean_valid.py"),
            "--data-dir", str(args.clean_dir),
            "--output", str(args.output / f"paper_methods_cal_{tag:02d}"),
            *scenario_args,
            "--fixed-window", str(args.window),
            "--windows", "3", "10", "20",
            "--ranking-top-k", "5",
            "--selection-seeds", str(args.seed),
            "--gating", "soft",
            "--rf-trees", str(args.rf_trees),
            "--expert-trees", str(args.expert_trees),
            "--adaptation-fraction", str(fraction),
        ])

    shared_meta = [
        "--data-dir", str(args.clean_dir),
        "--fractions", *[str(value) for value in args.fractions],
        "--test-fraction", str(args.test_fraction),
        *scenario_args,
        "--window", str(args.window),
        "--congestion-window", str(args.congestion_window),
        "--expert-trees", str(args.expert_trees),
        "--utility-trees", str(args.utility_trees),
        "--meta-trees", str(args.meta_trees),
        "--rf-trees", str(args.rf_trees),
        "--seed", str(args.seed),
    ]
    legacy_meta_output = args.output / "meta_stacked_legacy_budget_110"
    safe_meta_output = args.output / "meta_stacked_leakage_safe_budget_110"
    run([
        py, str(root / "meta_stacked_fixed_test_sweep.py"),
        *shared_meta, "--output", str(legacy_meta_output),
    ])
    run([
        py, str(root / "meta_stacked_fixed_test_sweep_leakage_safe.py"),
        *shared_meta, "--output", str(safe_meta_output),
    ])

    summaries = []
    for fraction in args.fractions:
        tag = int(round(fraction * 100))
        path = args.output / f"paper_methods_cal_{tag:02d}/comparison_summary.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "experiment_family", "paper_methods")
        frame.insert(1, "protocol", f"variable_target_tail_cal_{tag:02d}")
        frame.insert(2, "adaptation_fraction", fraction)
        summaries.append(frame)

    for family, protocol, path in [
        (
            "meta_stacker", "legacy_multilevel_fixed_test",
            legacy_meta_output / "meta_stacked_fixed_test_summary.csv",
        ),
        (
            "meta_stacker", "scenario_isolated_strict_temporal_fixed_test",
            safe_meta_output / "meta_stacked_fixed_test_summary.csv",
        ),
    ]:
        frame = pd.read_csv(path)
        frame.insert(0, "experiment_family", family)
        frame.insert(1, "protocol", protocol)
        summaries.append(frame)

    combined = pd.concat(summaries, ignore_index=True, sort=False)
    combined.to_csv(args.output / "combined_summary.csv", index=False)
    run_manifest = {
        "raw_source": str(args.source.resolve()),
        "clean_dataset": str(args.clean_dir.resolve()),
        "output": str(args.output.resolve()),
        "fractions": args.fractions,
        "scenarios": args.scenarios,
        "test_fraction": args.test_fraction,
        "budget": {
            "experts": f"3 x {args.expert_trees}",
            "utilities": f"3 x {args.utility_trees}",
            "meta": args.meta_trees,
            "total_meta_architecture_trees": (
                3 * args.expert_trees + 3 * args.utility_trees + args.meta_trees
            ),
            "rf_baseline_trees": args.rf_trees,
        },
        "important_comparability_note": (
            "Paper-method comparison uses compare_clean_valid's remaining target "
            "tail, while both meta-stacker runs use the immutable final test tail. "
            "Compare methods directly only within the same protocol."
        ),
    }
    (args.output / "legacy_experiment_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\nCombined summary:", args.output / "combined_summary.csv")


if __name__ == "__main__":
    main()
