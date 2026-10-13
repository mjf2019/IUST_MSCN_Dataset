"""Standard 1D-CNN on the fixed 20-class QUICEXT-25 S1--S7 protocols."""

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
    PROTOCOL_IDS, chronological_validation, evaluations, labels,
    load_clean_valid, matrices, record_ids, save_run, scores,
)


def run(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_clean_valid(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for development, test, definition in evaluations(data, args.scenarios, .80):
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
            "feature_count": len(features), "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return save_run(args.output, "1D-CNN", rows, audits, {
        "dataset": "CESNET-QUICEXT-25", "scenarios": args.scenarios,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed, "device": str(device),
        "protocol": "source-only training and complete immutable target test",
    })


def parse_args():
    dataset = CDR_MLC / "DATASETS/CESNET-QUICEXT-25/processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--output", type=Path, default=CDR_MLC / "outputs/quicext25_1dcnn")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
