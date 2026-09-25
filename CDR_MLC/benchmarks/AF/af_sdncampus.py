"""AF-MLP on an external dataset with a labeled budget drawn from train only."""
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
    evaluations, labels, load_clean_valid, matrices, record_ids, save_run, scores,
)


def training_budget_split(development, full_fraction, train_fraction=.80):
    """Take the latest labeled budget from train; never touch fixed test."""
    if not 0 < full_fraction < train_fraction:
        raise ValueError("--target-label-fraction must be in (0, 0.80)")
    source, calibration, audit = [], [], []
    within_train_fraction = full_fraction / train_fraction
    for sequence_id, group in development.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        count = max(1, int(np.floor(len(group) * within_train_fraction)))
        if count >= len(group):
            raise ValueError(f"{sequence_id}: AF budget consumes source train")
        source.append(group.iloc[:-count].copy())
        calibration.append(group.iloc[-count:].copy())
        audit.append({
            "sequence_id": sequence_id, "development_rows": len(group),
            "source_rows": len(group) - count, "labeled_calibration_rows": count,
        })
    return pd.concat(source, ignore_index=True), pd.concat(calibration, ignore_index=True), audit


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = replace(
        AFConfig(), folds=1, seed=args.seed,
        pretrain_epochs=args.epochs, batch_size=args.batch_size,
    )
    data, input_audit = load_clean_valid(args.data)
    dataset_name = str(input_audit.iloc[0]["dataset"])
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for development, test, definition in evaluations(data, ["7"], .80):
        source, calibration, budget_audit = training_budget_split(
            development, args.target_label_fraction
        )
        features, arrays = matrices(source, [calibration, test])
        source_x, calibration_x, test_x = arrays
        source_y, calibration_y, test_y = (
            labels(source), labels(calibration), labels(test)
        )
        seed_everything(args.seed)
        rng = np.random.default_rng(args.seed)
        source_index = select_per_class(source_y, config.source_per_class, rng)
        per_class = pd.Series(calibration_y).value_counts()
        if len(per_class) != len(np.unique(source_y)):
            raise ValueError("AF calibration does not contain every class")
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
            "budget": budget_audit, "features": features,
            "source_calibration_overlap": len(record_ids(source) & record_ids(calibration)),
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
        }
    return save_run(args.output, "AF", rows, audits, {
        "dataset": dataset_name, "split": "fixed 80/20: ISCX stratified; SDNCampus ordered",
        "target_label_fraction_of_full_dataset": args.target_label_fraction,
        "budget_source": "disjoint latest prefix inside the 80% training partition",
        "fixed_test_used_for_adaptation": False,
        "test_context_eligibility_window": 20,
        "seed": args.seed, "device": str(device),
    })


def parse_args():
    dataset = CDR_MLC.parent / "AMCAL/SDNCampus_TEST/Dataset/SDNCampus_original.csv"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=dataset)
    p.add_argument("--output", type=Path, default=CDR_MLC / "outputs/sdncampus_af")
    p.add_argument("--target-label-fraction", type=float, default=.01)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
