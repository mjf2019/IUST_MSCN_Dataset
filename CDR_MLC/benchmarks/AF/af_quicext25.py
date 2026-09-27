"""AF-MLP with an explicit per-class target-label budget on QUICEXT-25."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
for path in (HERE, CDR_MLC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from af_single_source import (  # noqa: E402
    AFConfig, embeddings, fit_domain_network, seed_everything, select_per_class,
)
from benchmarks.console_output import ResourceMonitor, resource_values  # noqa: E402
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, labels, load_clean_valid, matrices,
    record_ids, save_run, scores,
)


def labeled_prefix_per_class(frame: pd.DataFrame, fraction: float):
    """Reserve the earliest declared fraction of every target class."""
    calibration, test, audit = [], [], []
    for label, group in frame.groupby("traffic_label", sort=True):
        group = group.sort_values(["timestamp", "source_file", "source_row"], kind="stable")
        count = max(1, int(np.floor(len(group) * fraction)))
        if count >= len(group):
            raise ValueError(f"target class {label} is too small for AF calibration")
        calibration.append(group.iloc[:count].copy())
        test.append(group.iloc[count:].copy())
        audit.append({
            "traffic_label": str(label), "rows": len(group),
            "labeled_rows": count, "test_rows": len(group) - count,
        })
    return (
        pd.concat(calibration, ignore_index=True),
        pd.concat(test, ignore_index=True).sort_values(
            ["timestamp", "source_file", "source_row"], kind="stable"
        ).reset_index(drop=True),
        audit,
    )


def run(args):
    if not 0 < args.target_label_fraction < 1:
        raise ValueError("--target-label-fraction must be in (0,1)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = replace(
        AFConfig(), folds=1, seed=args.seed,
        pretrain_epochs=args.epochs, batch_size=args.batch_size,
    )
    data, input_audit = load_clean_valid(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}

    for development, target, definition in evaluations(data, args.scenarios, .80):
        calibration, test, budget_audit = labeled_prefix_per_class(
            target, args.target_label_fraction
        )
        features, arrays = matrices(development, [calibration, test])
        source_x, calibration_x, test_x = arrays
        source_y, calibration_y, test_y = (
            labels(development), labels(calibration), labels(test)
        )
        seed_everything(args.seed)
        rng = np.random.default_rng(args.seed)
        source_index = select_per_class(source_y, config.source_per_class, rng)
        per_class = pd.Series(calibration_y).value_counts()
        if len(per_class) != len(np.unique(source_y)):
            raise ValueError("AF calibration does not contain every fixed class")
        k = int(per_class.min())
        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            extractor = fit_domain_network(
                source_x[source_index], source_y[source_index], calibration_x,
                "tabular", config, device,
            )
            calibration_z = embeddings(extractor, calibration_x, device)
            classifier = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            classifier.fit(calibration_z, calibration_y)
            fit_seconds = time.perf_counter() - started
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = classifier.predict(embeddings(extractor, test_x, device))
            predict_seconds = time.perf_counter() - started
        rows.append({
            **definition, "seed": args.seed, "method": "AF-MLP-adapted",
            "n": len(test), "features": len(features),
            "target_labeled_rows": len(calibration),
            "target_label_fraction": args.target_label_fraction,
            "target_rows_per_class_min": k, **scores(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            **resource_values(fit_mem, infer_mem),
        })
        audits[definition["protocol"]] = {
            "budget": budget_audit, "feature_count": len(features),
            "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "calibration_test_overlap": len(record_ids(calibration) & record_ids(test)),
        }
    return save_run(args.output, "AF", rows, audits, {
        "dataset": "CESNET-QUICEXT-25", "scenarios": args.scenarios,
        "target_label_fraction": args.target_label_fraction,
        "seed": args.seed, "device": str(device),
        "protocol": (
            "earliest per-class labeled target prefix; remaining target rows immutable"
        ),
        "comparability_note": "AF is supervised target adaptation; all other QUIC runs are zero-shot",
    })


def parse_args():
    dataset = CDR_MLC / "DATASETS/CESNET-QUICEXT-25/processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--output", type=Path, default=CDR_MLC / "outputs/quicext25_af")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--target-label-fraction", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
