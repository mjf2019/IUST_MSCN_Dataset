"""Evaluate CDR-MLC variants with train and test from the same congestion level.

For each requested level, every application capture is split chronologically:
the leading ``train_fraction`` is development data and the remaining tail is
an untouched test set.  Windows never cross the development/test boundary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, LEVELS, load_dataset
from compare_clean_valid import TIMING, fit_rf, metrics, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all


def chronological_same_level_split(level_data: pd.DataFrame, train_fraction: float):
    development, test, captures = [], [], []
    for capture_id, group in level_data.groupby("capture_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * train_fraction)
        if not 0 < cut < len(group):
            raise ValueError(f"{capture_id}: invalid chronological split")
        development.append(group.iloc[:cut].copy())
        test.append(group.iloc[cut:].copy())
        captures.append({
            "capture_id": capture_id,
            "total_rows": len(group),
            "development_rows": cut,
            "test_rows": len(group) - cut,
            "development_last_source_row": int(group.iloc[cut - 1].source_row),
            "test_first_source_row": int(group.iloc[cut].source_row),
        })
    return (
        pd.concat(development, ignore_index=True),
        pd.concat(test, ignore_index=True),
        captures,
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
        default=root / "outputs/same_level_meta_stacked",
    )
    parser.add_argument(
        "--levels", nargs="+", choices=LEVELS, default=LEVELS,
    )
    parser.add_argument("--train-fraction", type=float, default=.80)
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

    rows, audits = [], {}
    for level in args.levels:
        level_data = data[data.congestion_level.eq(level)].copy()
        development, test, split_audit = chronological_same_level_split(
            level_data, args.train_fraction
        )
        model = fit_meta_stacker(development, config)
        eligible = development.loc[model["source_eligible_index"]]
        rf = fit_rf(eligible, TIMING, args.seed, args.rf_trees)

        result = predict_all(model, test)
        observed = result["observed"]
        truth = observed.traffic_label.to_numpy()
        predictions = {
            key: value for key, value in result.items()
            if key.startswith("CDR_")
        }
        predictions["RF_expert_inputs"] = predict_rf(rf, observed)

        level_dir = args.output / f"level_{level.lower()}"
        level_dir.mkdir(parents=True, exist_ok=True)
        details = {}
        for method, prediction in predictions.items():
            score = metrics(truth, prediction, APPLICATIONS)
            details[method] = score
            rows.append({
                "train_level": level,
                "test_level": level,
                "train_fraction": args.train_fraction,
                "method": method,
                **{key: score[key] for key in (
                    "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "weighted_f1",
                )},
            })

        frame = observed[[
            "source_file", "source_row", "timestamp",
            "traffic_label", "congestion_level",
        ]].copy()
        for method, prediction in predictions.items():
            frame[f"prediction_{method}"] = prediction
        frame.join(result["routes"]).to_csv(
            level_dir / "predictions.csv", index=False
        )
        (level_dir / "metrics.json").write_text(
            json.dumps(details, indent=2) + "\n", encoding="utf-8"
        )
        audit = {
            "level": level,
            "raw_rows": len(level_data),
            "development_rows_raw": len(development),
            "test_rows_raw": len(test),
            "evaluated_test_rows": len(observed),
            "split_by_capture": split_audit,
            "partition_rows": model["partition_rows"],
            "utility_correctness_counts": model["utility_correctness_counts"],
            "selected_meta_variant": model["selected_meta_variant"],
            "selected_meta_confidence": model["selected_meta_confidence"],
            "meta_selection_trials": model["meta_selection_trials"],
        }
        audits[level] = audit
        (level_dir / "same_level_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )

    summary = pd.DataFrame(rows).sort_values(["train_level", "method"])
    summary.to_csv(args.output / "same_level_summary.csv", index=False)
    manifest = {
        "config": asdict(config),
        "levels": args.levels,
        "train_fraction": args.train_fraction,
        "rf_trees": args.rf_trees,
        "audits": audits,
        "leakage_control": (
            "Each application-level capture is split chronologically before "
            "window extraction. Development and same-level test tails are disjoint."
        ),
    }
    (args.output / "same_level_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
