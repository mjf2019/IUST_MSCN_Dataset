"""Leakage-safe standard Random Forest benchmark on IUST_MSCN Clean-Valid.

The target test set is the immutable chronological tail of every capture.
For a nonzero adaptation budget, only the disjoint chronological target prefix
is added to the complete source level.  All preprocessing is fitted on this
development set; target-test rows are used only by the final evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
CDR_MLC = BENCHMARKS.parent
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import (  # noqa: E402
    DEFAULT_CANDIDATES,
    FORBIDDEN,
    load_dataset,
    select_classifier_columns,
)
from compare_clean_valid import SCENARIOS  # noqa: E402
from benchmarks.console_output import (  # noqa: E402
    ResourceMonitor, print_compact_results, resource_values,
)

LEGACY_EXCLUDED = {"IdleTime", "DstWin"}


def fixed_target_tail(frame: pd.DataFrame, adaptation_fraction: float,
                      test_fraction: float):
    calibration_parts, test_parts, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        calibration_end = int(len(group) * adaptation_fraction)
        test_start = int(len(group) * (1.0 - test_fraction))
        if calibration_end > test_start:
            raise ValueError(f"{sequence_id}: adaptation prefix overlaps test tail")
        calibration_parts.append(group.iloc[:calibration_end].copy())
        test_parts.append(group.iloc[test_start:].copy())
        audit.append({
            "sequence_id": sequence_id,
            "rows": int(len(group)),
            "calibration_rows": int(calibration_end),
            "unused_rows": int(test_start - calibration_end),
            "test_rows": int(len(group) - test_start),
        })
    empty = frame.iloc[:0].copy()
    calibration = (
        pd.concat(calibration_parts, ignore_index=True)
        if adaptation_fraction > 0 else empty
    )
    return calibration, pd.concat(test_parts, ignore_index=True), audit


def fit_transformer(development: pd.DataFrame):
    numeric, _ = select_classifier_columns(development, route_features=[])
    numeric = [
        name for name in numeric
        if name not in FORBIDDEN and name not in LEGACY_EXCLUDED
    ]
    if not numeric:
        raise ValueError("no usable numeric Clean-Valid features")
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(development[numeric])
    # Scaling is retained from the supplied legacy Standard_RF notebook even
    # though tree splits themselves are invariant to monotonic scaling.
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_imputed).astype(np.float32)
    return numeric, imputer, scaler, train_x


def transform(frame: pd.DataFrame, features, imputer, scaler):
    return scaler.transform(imputer.transform(frame[features])).astype(np.float32)


def metric_values(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def run(args):
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    if any(value < 0 or value > 1 - args.test_fraction for value in args.fractions):
        raise ValueError("an adaptation fraction overlaps the fixed test tail")
    if args.trees < 1:
        raise ValueError("--trees must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    results, split_audits = [], {}

    for scenario in args.scenarios:
        source_level, target_level = SCENARIOS[scenario]
        source = data[data.congestion_level.eq(source_level)].copy()
        target = data[data.congestion_level.eq(target_level)].copy()
        for fraction in args.fractions:
            calibration, test, audit = fixed_target_tail(
                target, fraction, args.test_fraction
            )
            development = pd.concat([source, calibration], ignore_index=True)
            with ResourceMonitor("cpu") as fit_mem:
                started = time.perf_counter()
                features, imputer, scaler, train_x = fit_transformer(development)
                model = RandomForestClassifier(
                    n_estimators=args.trees,
                    criterion="gini",
                    max_depth=None,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    bootstrap=True,
                    class_weight="balanced",
                    n_jobs=args.jobs,
                    random_state=args.seed,
                )
                model.fit(train_x, development.traffic_label.astype(str).to_numpy())
                fit_seconds = time.perf_counter() - started
            with ResourceMonitor("cpu") as infer_mem:
                started = time.perf_counter()
                prediction = model.predict(
                    transform(test, features, imputer, scaler)
                )
                predict_seconds = time.perf_counter() - started
            results.append({
                "adaptation_fraction": fraction,
                "fixed_test_fraction": args.test_fraction,
                "scenario": scenario,
                "source": source_level,
                "target": target_level,
                "method": "Standard_RF",
                "seed": args.seed,
                "development_n": int(len(development)),
                "target_labeled_n": int(len(calibration)),
                "n": int(len(test)),
                **metric_values(test.traffic_label.astype(str), prediction),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_and_inference_seconds": fit_seconds + predict_seconds,
                "inference_us_per_row": 1e6 * predict_seconds / len(test),
                "throughput_rows_per_second": len(test) / predict_seconds,
                **resource_values(fit_mem, infer_mem),
            })
            key = f"scenario={scenario}:fraction={fraction:.4f}"
            split_audits[key] = {
                "captures": audit,
                "feature_count": len(features),
                "features": features,
                "calibration_test_overlap_by_capture_row": int(len(
                    set(zip(calibration.source_file, calibration.source_row))
                    & set(zip(test.source_file, test.source_row))
                )),
            }

    summary = pd.DataFrame(results).sort_values(["adaptation_fraction", "scenario"])
    summary.to_csv(args.output / "standard_rf_summary.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(split_audits, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "method": "Standard_RF",
        "data_dir": str(args.data_dir),
        "fractions": args.fractions,
        "test_fraction": args.test_fraction,
        "scenarios": args.scenarios,
        "seed": args.seed,
        "trees": args.trees,
        "max_depth": None,
        "class_weight": "balanced",
        "protocol": "source level plus disjoint target prefix; immutable target tail",
        "legacy_compatibility": [
            "100 unrestricted-depth trees by default",
            "median imputation and StandardScaler fitted on development only",
            "IdleTime and DstWin are excluded as in the supplied notebook",
            "all labels, congestion fields and identifiers are excluded from model inputs",
        ],
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(
        summary, method="RF", calibration_column="target_labeled_n",
        seconds_column=None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/iust_mscn")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .01, .05, .10, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
