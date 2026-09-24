"""Leakage-safe AF-SingleSource adaptation for IUST_MSCN Clean-Valid.

This runner preserves the AF learning mechanism while replacing the Tor/DF
input representation with an MLP over Clean-Valid flow features.  It uses the
same fixed-tail percentage protocol as the S-Meta experiments so results are
directly comparable.  Output method name: AF-MLP-adapted.
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
from benchmarks.console_output import print_compact_results  # noqa: E402
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


def run(args):
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    if any(f < 0 or f > 1 - args.test_fraction for f in args.fractions):
        raise ValueError("adaptation fractions overlap the fixed test tail")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_config = replace(
        AFConfig(), folds=1, seed=args.seed,
        pretrain_epochs=args.epochs, batch_size=args.batch_size,
    )
    rows, split_audits = [], {}

    for scenario in args.scenarios:
        source_level, target_level = SCENARIOS[scenario]
        source = data[data.congestion_level.eq(source_level)].copy()
        target_all = data[data.congestion_level.eq(target_level)].copy()

        for fraction in args.fractions:
            calibration, test, audit = fixed_tail(target_all, fraction, args.test_fraction)
            audit_key = f"scenario={scenario}:fraction={fraction:.4f}"
            split_audits[audit_key] = audit
            if fraction == 0:
                rows.append({
                    "adaptation_fraction": fraction,
                    "fixed_test_fraction": args.test_fraction,
                    "scenario": scenario,
                    "source": source_level, "target": target_level,
                    "method": "AF-MLP-adapted", "n": len(test),
                    "target_labeled_rows": 0, "target_rows_per_class_min": 0,
                    "status": "N/A: AF target k-NN requires labeled target samples",
                    "accuracy": np.nan, "balanced_accuracy": np.nan,
                    "macro_f1": np.nan, "weighted_f1": np.nan,
                    "fit_and_inference_seconds": np.nan,
                })
                continue

            labels = LabelEncoder().fit(source.traffic_label.astype(str))
            unseen = sorted(set(calibration.traffic_label.astype(str)) - set(labels.classes_))
            if unseen:
                raise ValueError(f"target contains unseen traffic labels: {unseen}")
            y_source = labels.transform(source.traffic_label.astype(str))
            y_calibration = labels.transform(calibration.traffic_label.astype(str))
            y_test = labels.transform(test.traffic_label.astype(str))
            feature_names, matrices = numeric_matrix_fit(source, [calibration, test])
            x_source, x_calibration, x_test = matrices

            seed_everything(args.seed)
            rng = np.random.default_rng(args.seed)
            source_idx = select_per_class(
                y_source, base_config.source_per_class, rng
            )
            per_class = pd.Series(y_calibration).value_counts()
            if per_class.empty or per_class.min() < 1:
                raise ValueError(f"{audit_key}: missing calibration class")
            # AF uses k=N, where N is target training traces per class.  With
            # unequal IUST capture sizes we use the minimum class budget.
            k = int(per_class.min())
            started = time.perf_counter()
            extractor = fit_domain_network(
                x_source[source_idx], y_source[source_idx], x_calibration,
                "tabular", base_config, device,
            )
            calibration_z = embeddings(extractor, x_calibration, device)
            test_z = embeddings(extractor, x_test, device)
            classifier = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            classifier.fit(calibration_z, y_calibration)
            predicted = classifier.predict(test_z)
            score = metric_row(y_test, predicted)
            rows.append({
                "adaptation_fraction": fraction, "fixed_test_fraction": args.test_fraction,
                "scenario": scenario, "source": source_level, "target": target_level,
                "method": "AF-MLP-adapted", "n": len(test),
                "target_labeled_rows": len(calibration),
                "target_rows_per_class_min": k, "M": 25, "k": k,
                "status": "ok", **score,
                "fit_and_inference_seconds": time.perf_counter() - started,
            })
            split_audits[audit_key + ":model"] = {
                "feature_count": len(feature_names), "features": feature_names,
                "source_selected_rows": source.iloc[source_idx].source_row.astype(int).tolist(),
                "calibration_source_rows": calibration.source_row.astype(int).tolist(),
                "test_source_rows": test.source_row.astype(int).tolist(),
                "calibration_test_overlap_by_capture_row": int(len(set(zip(
                    calibration.source_file, calibration.source_row
                )) & set(zip(test.source_file, test.source_row)))),
            }

    result = pd.DataFrame(rows).sort_values(["adaptation_fraction", "scenario"])
    result.to_csv(args.output / "af_iust_mscn_summary.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(split_audits, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "method": "AF-MLP-adapted",
        "paper_faithful": [
            "DANN with GRL", "M=25 source rows per class", "30 epochs by default",
            "lambda=1", "learning rate=1e-5", "512-dimensional embedding",
            "the identical target calibration samples serve as unlabeled DANN input and labeled k-NN input",
            "k equals the minimum target samples per class",
        ],
        "dataset_adaptations": [
            "MLP replaces the DF packet-direction CNN",
            "fractional chronological calibration replaces N={1,5,10,15,20}",
            "a fixed 20% chronological target tail replaces T=70 per class",
        ],
        "data_dir": str(args.data_dir), "fractions": args.fractions,
        "test_fraction": args.test_fraction, "scenarios": args.scenarios,
        "seed": args.seed, "device": str(device),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(
        result, method="AF", calibration_column="target_labeled_rows",
        seconds_column="fit_and_inference_seconds",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=MODULE_ROOT / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/iust_mscn")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .01, .05, .10, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
