"""Run AF with a 1% labeled-target budget on all seven IUST protocols.

AF is first trained on the complete labeled development partition.  Its
reported target budget is used only for drift adaptation.  Calibration rows
are always disjoint from scored rows.  S1--S6 reserve an ordered target prefix;
S7 reserves the last 1% of the full stream from its 80% development partition.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from benchmark_iust_mscn import fixed_tail, numeric_matrix_fit  # noqa: E402
from af_single_source import (  # noqa: E402
    AFConfig, embeddings, fit_domain_network, metric_row, seed_everything,
)
from benchmarks.console_output import (  # noqa: E402
    ResourceMonitor, print_compact_results, resource_values,
)
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, load_clean_valid, record_ids,
)


def reserve_from_development(frame, full_fraction, train_fraction=.80):
    """Reserve a causal calibration tail from S7 development only."""
    within = full_fraction / train_fraction
    source_parts, calibration_parts, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        count = int(np.floor(len(group) * within))
        if count < 1 or count >= len(group):
            raise ValueError(f"{sequence_id}: insufficient S7 calibration rows")
        source_parts.append(group.iloc[:-count].copy())
        calibration_parts.append(group.iloc[-count:].copy())
        audit.append({
            "sequence_id": sequence_id, "development_rows": len(group),
            "source_rows": len(group) - count, "calibration_rows": count,
        })
    return (
        pd.concat(source_parts, ignore_index=True),
        pd.concat(calibration_parts, ignore_index=True),
        audit,
    )


def run(args):
    if not 0 < args.target_label_fraction < .80:
        raise ValueError("--target-label-fraction must be in (0,.80)")
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_clean_valid(args.data_dir)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = replace(
        AFConfig(), folds=1, seed=args.seed,
        pretrain_epochs=args.epochs, batch_size=args.batch_size,
    )
    rows, audits = [], {}

    for development, common_test, definition in evaluations(
        data, args.scenarios, train_fraction=.80, target_test_fraction=.20
    ):
        protocol = definition["protocol"]
        if protocol == "S7":
            source, calibration, split_audit = reserve_from_development(
                development, args.target_label_fraction, .80
            )
            test = common_test.copy()
        else:
            target_all = data[
                data.congestion_level.astype(str).eq(str(definition["target"]))
            ].copy()
            calibration, _, split_audit = fixed_tail(
                target_all, args.target_label_fraction, .20
            )
            source = development.copy()
            # S4--S6 score the held-out target level.  Remove the labeled
            # prefix from the common test so calibration is never scored.
            calibration_ids = record_ids(calibration)
            keep = [
                identity not in calibration_ids
                for identity in zip(
                    common_test.source_file.astype(str),
                    common_test.source_row.astype(int),
                )
            ]
            test = common_test.loc[keep].copy().reset_index(drop=True)

        overlap = record_ids(calibration) & record_ids(test)
        if overlap:
            raise RuntimeError(f"{protocol}: calibration/test overlap")
        if calibration.empty or test.empty:
            raise ValueError(f"{protocol}: empty calibration or test partition")

        encoder = LabelEncoder().fit(source.traffic_label.astype(str))
        unseen = sorted(
            (set(calibration.traffic_label.astype(str)) |
             set(test.traffic_label.astype(str))) - set(encoder.classes_)
        )
        if unseen:
            raise ValueError(f"{protocol}: unseen target classes: {unseen}")
        y_source = encoder.transform(source.traffic_label.astype(str))
        y_calibration = encoder.transform(calibration.traffic_label.astype(str))
        y_test = encoder.transform(test.traffic_label.astype(str))
        features, (x_source, x_calibration, x_test) = numeric_matrix_fit(
            source, [calibration, test]
        )
        per_class = pd.Series(y_calibration).value_counts()
        if len(per_class) != len(encoder.classes_) or per_class.min() < 1:
            raise ValueError(f"{protocol}: calibration does not cover every class")
        k = int(per_class.min())

        seed_everything(args.seed)
        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            extractor = fit_domain_network(
                x_source, y_source, x_calibration, "tabular", config, device
            )
            calibration_z = embeddings(extractor, x_calibration, device)
            classifier = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            classifier.fit(calibration_z, y_calibration)
            fit_seconds = time.perf_counter() - started
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = classifier.predict(embeddings(extractor, x_test, device))
            predict_seconds = time.perf_counter() - started

        rows.append({
            "protocol": protocol, "scenario": protocol.removeprefix("S"),
            "source": definition["source"], "target": definition["target"],
            "method": "AF-MLP-FullSource", "seed": args.seed,
            "adaptation_fraction": args.target_label_fraction,
            "source_labeled_rows": len(source),
            "target_labeled_rows": len(calibration),
            "target_rows_per_class_min": k, "n": len(test),
            **metric_row(y_test, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            **resource_values(fit_mem, infer_mem),
        })
        audits[protocol] = {
            "definition": definition, "split": split_audit,
            "features": features, "feature_count": len(features),
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "calibration_test_overlap": 0,
            "calibration_identity": sorted(record_ids(calibration)),
            "test_identity": sorted(record_ids(test)),
        }

    summary = pd.DataFrame(rows).sort_values("scenario")
    summary.to_csv(args.output / "results.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_manifest.json").write_text(json.dumps({
        "method": "AF-MLP-FullSource", "protocols": args.scenarios,
        "source_supervision": "all labeled development rows",
        "target_adaptation_fraction": args.target_label_fraction,
        "target_adaptation_and_test_are_disjoint": True,
        "seed": args.seed, "device": str(device),
    }, indent=2) + "\n", encoding="utf-8")
    print_compact_results(
        summary, method="AF", calibration_column="target_labeled_rows",
        seconds_column=None,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/table5_s1_s7")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--target-label-fraction", type=float, default=.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
