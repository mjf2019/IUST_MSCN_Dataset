"""Leakage-safe AF-SingleSource adaptation for IUST_MSCN Clean-Valid.

This runner preserves the AF learning mechanism while replacing the Tor/DF
input representation with an MLP over Clean-Valid flow features. It uses the
same fixed-tail percentage protocol as the MF-CDR-MLC experiments. For the
reviewer-requested equal source-supervision comparison, all labeled source
development rows are used by default; the paper's M=25-per-class source budget
remains available through --source-budget paper25.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
CDR_MLC = BENCHMARKS.parent
MODULE_ROOT = CDR_MLC
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from adaptive_cdr_mlc import (  # noqa: E402
    DEFAULT_CANDIDATES,
    FORBIDDEN,
    load_dataset,
    select_classifier_columns,
)
from compare_clean_valid import SCENARIOS  # noqa: E402
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, load_clean_valid, record_ids,
)
from benchmarks.console_output import (  # noqa: E402
    ResourceMonitor, print_compact_results, resource_values,
)
from af_single_source import (  # noqa: E402
    AFConfig,
    embeddings,
    fit_domain_network,
    metric_row,
    seed_everything,
    select_per_class,
)


def fixed_tail(frame: pd.DataFrame, fraction: float, test_fraction: float):
    calibration, test, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        calibration_end = int(len(group) * fraction)
        test_start = int(len(group) * (1.0 - test_fraction))
        if calibration_end > test_start:
            raise ValueError(f"{sequence_id}: calibration overlaps test")
        calibration.append(group.iloc[:calibration_end])
        test.append(group.iloc[test_start:])
        audit.append({
            "sequence_id": sequence_id,
            "rows": len(group),
            "calibration_rows": calibration_end,
            "unused_rows": test_start - calibration_end,
            "test_rows": len(group) - test_start,
        })
    empty = frame.iloc[:0].copy()
    return (
        pd.concat(calibration, ignore_index=True) if fraction > 0 else empty,
        pd.concat(test, ignore_index=True),
        audit,
    )


def numeric_matrix_fit(source: pd.DataFrame, target_parts: list[pd.DataFrame]):
    numeric, _ = select_classifier_columns(source, route_features=[])
    numeric = [column for column in numeric if column not in FORBIDDEN]
    if not numeric:
        raise ValueError("no numeric Clean-Valid features available")
    imputer = SimpleImputer(strategy="median").fit(source[numeric])
    source_imputed = imputer.transform(source[numeric])
    scaler = StandardScaler().fit(source_imputed)
    matrices = [scaler.transform(source_imputed).astype(np.float32)]
    for frame in target_parts:
        matrices.append(
            scaler.transform(imputer.transform(frame[numeric])).astype(np.float32)
        )
    return numeric, matrices


def _reserve_development_tail(frame, fraction, train_fraction=.80):
    within = fraction / train_fraction
    source_parts, calibration_parts, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        count = int(np.floor(len(group) * within))
        if count < 1 or count >= len(group):
            raise ValueError(f"{sequence_id}: insufficient S7 calibration rows")
        source_parts.append(group.iloc[:-count].copy())
        calibration_parts.append(group.iloc[-count:].copy())
        audit.append({"sequence_id": sequence_id, "calibration_rows": count})
    return (
        pd.concat(source_parts, ignore_index=True),
        pd.concat(calibration_parts, ignore_index=True),
        audit,
    )


def run(args):
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    if any(f < 0 or f > 1 - args.test_fraction for f in args.fractions):
        raise ValueError("adaptation fractions overlap the fixed test tail")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_clean_valid(args.data_dir)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_config = replace(
        AFConfig(), folds=1, seed=args.seed,
        pretrain_epochs=args.epochs, batch_size=args.batch_size,
    )
    rows, split_audits = [], {}
    method_name = (
        "AF-MLP-FullSource" if args.source_budget == "all"
        else "AF-MLP-Paper25"
    )

    for development, common_test, definition in evaluations(
        data, args.scenarios, train_fraction=1.0 - args.test_fraction,
        target_test_fraction=args.test_fraction,
    ):
        protocol = definition["protocol"]
        scenario = protocol.removeprefix("S")
        for fraction in args.fractions:
            if fraction == 0:
                source, calibration, audit = (
                    development.copy(), development.iloc[:0].copy(), []
                )
            elif protocol == "S7":
                source, calibration, audit = _reserve_development_tail(
                    development, fraction, 1.0 - args.test_fraction
                )
            else:
                source = development.copy()
                target = data[
                    data.congestion_level.astype(str).eq(str(definition["target"]))
                ].copy()
                calibration, _, audit = fixed_tail(
                    target, fraction, args.test_fraction
                )

            calibration_ids = record_ids(calibration)
            keep = [
                identity not in calibration_ids
                for identity in zip(
                    common_test.source_file.astype(str),
                    common_test.source_row.astype(int),
                )
            ]
            test = common_test.loc[keep].copy().reset_index(drop=True)
            audit_key = f"{protocol}:fraction={fraction:.4f}"
            split_audits[audit_key] = {
                "definition": definition, "captures": audit,
                "development_test_overlap": len(record_ids(development) & record_ids(test)),
                "calibration_test_overlap": len(record_ids(calibration) & record_ids(test)),
            }
            if record_ids(calibration) & record_ids(test):
                raise RuntimeError(f"{protocol}: calibration/test overlap")
            if fraction == 0:
                rows.append({
                    "adaptation_fraction": fraction,
                    "fixed_test_fraction": args.test_fraction,
                    "protocol": protocol, "scenario": scenario,
                    "source": definition["source"], "target": definition["target"],
                    "method": method_name, "seed": args.seed, "n": len(test),
                    "source_budget": args.source_budget,
                    "source_labeled_rows": len(source),
                    "target_labeled_rows": 0, "target_rows_per_class_min": 0,
                    "status": "N/A: AF target k-NN requires labeled target samples",
                    "accuracy": np.nan, "balanced_accuracy": np.nan,
                    "macro_f1": np.nan, "weighted_f1": np.nan,
                    "fit_seconds": np.nan, "predict_seconds": np.nan,
                    "inference_us_per_row": np.nan,
                    "throughput_rows_per_second": np.nan,
                    "peak_ram_mb": np.nan, "peak_gpu_mb": np.nan,
                    "fit_and_inference_seconds": np.nan,
                })
                continue
            if calibration.empty or test.empty:
                raise ValueError(f"{protocol}: empty calibration or test partition")

            labels = LabelEncoder().fit(source.traffic_label.astype(str))
            unseen = sorted(
                (set(calibration.traffic_label.astype(str)) |
                 set(test.traffic_label.astype(str))) - set(labels.classes_)
            )
            if unseen:
                raise ValueError(f"{protocol}: unseen target classes: {unseen}")
            y_source = labels.transform(source.traffic_label.astype(str))
            y_calibration = labels.transform(calibration.traffic_label.astype(str))
            y_test = labels.transform(test.traffic_label.astype(str))
            feature_names, matrices = numeric_matrix_fit(source, [calibration, test])
            x_source, x_calibration, x_test = matrices

            seed_everything(args.seed)
            rng = np.random.default_rng(args.seed)
            if args.source_budget == "all":
                source_idx = np.arange(len(y_source), dtype=int)
            else:
                source_idx = select_per_class(
                    y_source, base_config.source_per_class, rng
                )
            per_class = pd.Series(y_calibration).value_counts()
            if len(per_class) != len(labels.classes_) or per_class.min() < 1:
                raise ValueError(f"{protocol}: calibration does not cover every class")
            k = int(per_class.min())
            with ResourceMonitor(device) as fit_mem:
                started = time.perf_counter()
                extractor = fit_domain_network(
                    x_source[source_idx], y_source[source_idx],
                    x_calibration, "tabular", base_config, device,
                )
                calibration_z = embeddings(extractor, x_calibration, device)
                classifier = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
                classifier.fit(calibration_z, y_calibration)
                fit_seconds = time.perf_counter() - started
            with ResourceMonitor(device) as infer_mem:
                started = time.perf_counter()
                test_z = embeddings(extractor, x_test, device)
                predicted = classifier.predict(test_z)
                predict_seconds = time.perf_counter() - started
            rows.append({
                "adaptation_fraction": fraction,
                "fixed_test_fraction": args.test_fraction,
                "protocol": protocol, "scenario": scenario,
                "source": definition["source"], "target": definition["target"],
                "method": method_name, "seed": args.seed, "n": len(test),
                "source_budget": args.source_budget,
                "source_labeled_rows": len(source_idx),
                "target_labeled_rows": len(calibration),
                "target_rows_per_class_min": k,
                "M": ("all" if args.source_budget == "all" else 25), "k": k,
                "status": "ok", **metric_row(y_test, predicted),
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "fit_and_inference_seconds": fit_seconds + predict_seconds,
                "inference_us_per_row": 1e6 * predict_seconds / len(test),
                "throughput_rows_per_second": len(test) / predict_seconds,
                **resource_values(fit_mem, infer_mem),
            })
            split_audits[audit_key + ":model"] = {
                "feature_count": len(feature_names), "features": feature_names,
                "source_selected_rows": source.iloc[source_idx].source_row.astype(int).tolist(),
                "calibration_test_overlap": 0,
            }

    result = pd.DataFrame(rows).sort_values(["adaptation_fraction", "scenario"])
    result.to_csv(args.output / "af_iust_mscn_summary.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(split_audits, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_manifest.json").write_text(json.dumps({
        "method": method_name, "source_budget": args.source_budget,
        "fractions": args.fractions, "test_fraction": args.test_fraction,
        "scenarios": args.scenarios, "seed": args.seed, "device": str(device),
        "source_supervision": "all labeled development rows" if args.source_budget == "all"
                              else "M=25 labeled source rows per class",
        "protocol": "common leakage-safe S1-S7; disjoint target calibration/test",
    }, indent=2) + "\n", encoding="utf-8")
    print_compact_results(
        result, method="AF", calibration_column="target_labeled_rows",
        seconds_column=None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=MODULE_ROOT / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/iust_mscn")
    parser.add_argument("--fractions", nargs="+", type=float, default=[.01])
    parser.add_argument(
        "--source-budget", choices=("all", "paper25"), default="all",
        help="all labeled source rows (fairness default) or AF paper M=25 per class",
    )
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
