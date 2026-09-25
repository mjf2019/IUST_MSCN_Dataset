"""Standard 1D-CNN on a fixed external-dataset ordered 80/20 protocol."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
for path in (HERE, CDR_MLC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from standard_1d_cnn_iust_mscn import fit_model, predict  # noqa: E402
from benchmarks.console_output import ResourceMonitor, resource_values  # noqa: E402
from benchmarks.deep_common import (  # noqa: E402
    chronological_validation, evaluations, labels, load_clean_valid,
    matrices, record_ids, save_run, scores,
)


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_clean_valid(args.data)
    dataset_name = str(input_audit.iloc[0]["dataset"])
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for development, test, definition in evaluations(data, ["7"], .80):
        train, valid = chronological_validation(development, args.validation_fraction)
        features, arrays = matrices(train, [valid, test])
        train_x, valid_x, test_x = arrays
        train_y, valid_y, test_y = labels(train), labels(valid), labels(test)
        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            model, epochs, best_loss = fit_model(
                train_x, train_y, valid_x, valid_y, args, device
            )
            fit_seconds = time.perf_counter() - started
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = predict(model, test_x, args.batch_size, device, args.seed)
            predict_seconds = time.perf_counter() - started
        rows.append({
            **definition, "seed": args.seed, "method": "Standard_1D_CNN",
            "train_n": len(train), "validation_n": len(valid), "n": len(test),
            "features": len(features), "epochs": epochs,
            "best_validation_loss": best_loss, **scores(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
            **resource_values(fit_mem, infer_mem),
        })
        audits[definition["protocol"]] = {
            "development_n": len(development), "train_n": len(train),
            "validation_n": len(valid), "test_n": len(test),
            "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
        }
    return save_run(args.output, "1D-CNN", rows, audits, {
        "dataset": dataset_name, "split": "fixed 80/20: ISCX stratified; SDNCampus ordered",
        "validation_fraction_of_training": args.validation_fraction,
        "test_context_eligibility_window": 20,
        "seed": args.seed, "device": str(device),
    })


def parse_args():
    dataset = CDR_MLC.parent / "AMCAL/SDNCampus_TEST/Dataset/SDNCampus_original.csv"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=dataset)
    p.add_argument("--output", type=Path, default=CDR_MLC / "outputs/sdncampus_1dcnn")
    p.add_argument("--validation-fraction", type=float, default=.10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=.001)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
