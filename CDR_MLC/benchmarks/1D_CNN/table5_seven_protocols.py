"""Run the standard 1D-CNN on the seven common IUST_MSCN protocols.

This table runner is zero-shot: no labeled target rows are added.  It reuses
the published benchmark implementation and the common leakage-safe S1--S7
partitions used by the other Table-5 methods.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from standard_1d_cnn_iust_mscn import (  # noqa: E402
    chronological_source_validation, encoded_labels, fit_model,
    fitted_matrices, metric_values, predict,
)
from benchmarks.console_output import (  # noqa: E402
    ResourceMonitor, print_compact_results, resource_values,
)
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, load_clean_valid, record_ids,
)


def run(args):
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in (0,1)")
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_clean_valid(args.data_dir)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows, audits = [], {}

    for development, test, definition in evaluations(
        data, args.scenarios, train_fraction=.80, target_test_fraction=.20
    ):
        protocol = definition["protocol"]
        train, validation = chronological_source_validation(
            development, args.validation_fraction
        )
        features, (train_x, validation_x, test_x) = fitted_matrices(
            train, [validation, test]
        )
        train_y = encoded_labels(train)
        validation_y = encoded_labels(validation)
        test_y = encoded_labels(test)

        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            model, epochs_completed, best_validation_loss = fit_model(
                train_x, train_y, validation_x, validation_y, args, device
            )
            fit_seconds = time.perf_counter() - started
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = predict(model, test_x, args.batch_size, device, args.seed)
            predict_seconds = time.perf_counter() - started

        rows.append({
            "protocol": protocol, "scenario": protocol.removeprefix("S"),
            "source": definition["source"], "target": definition["target"],
            "method": "Standard_1D_CNN", "seed": args.seed,
            "train_n": len(train), "validation_n": len(validation),
            "target_labeled_n": 0, "n": len(test),
            "epochs_completed": epochs_completed,
            "best_validation_loss": best_validation_loss,
            **metric_values(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            **resource_values(fit_mem, infer_mem),
        })
        audits[protocol] = {
            "definition": definition,
            "feature_count": len(features), "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "train_test_overlap": len(record_ids(train) & record_ids(test)),
        }

    summary = pd.DataFrame(rows).sort_values("scenario")
    summary.to_csv(args.output / "results.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_manifest.json").write_text(json.dumps({
        "method": "Standard_1D_CNN", "protocols": args.scenarios,
        "target_label_budget": 0, "seed": args.seed, "device": str(device),
        "shared_test_protocol": "common leakage-safe S1-S7",
    }, indent=2) + "\n", encoding="utf-8")
    print_compact_results(
        summary, method="1D-CNN", calibration_column="target_labeled_n",
        seconds_column=None, extra_columns={"Ep": "epochs_completed"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/table5_s1_s7")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
