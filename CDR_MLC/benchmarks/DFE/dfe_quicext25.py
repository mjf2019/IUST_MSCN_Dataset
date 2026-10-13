"""DFE-adapted on the fixed 20-class QUICEXT-25 S1--S7 protocols."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
for path in (HERE, CDR_MLC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dfe_iust_mscn import (  # noqa: E402
    DFEConfig, FlowImageTransform, choose_k, chronological_source_split,
    seed_all, select_templates, template_predict, train_backbone,
)
from benchmarks.console_output import ResourceMonitor, resource_values  # noqa: E402
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, labels, load_clean_valid, record_ids,
    save_run, scores,
)
from quicext25_common import numeric_model_features  # noqa: E402


class QUICFlowImageTransform(FlowImageTransform):
    """DFE 9x9 mapping fitted only on model-safe QUIC numeric fields."""

    def fit(self, frame):
        self.columns = sorted(numeric_model_features(frame))[:self.fields]
        if not self.columns:
            raise ValueError("no numeric QUICEXT input fields")
        imputed = self.imputer.fit_transform(frame[self.columns])
        self.scaler.fit(imputed)
        return self


def run(args):
    config = DFEConfig(epochs=args.epochs, seed=args.seed)
    seed_all(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_clean_valid(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits, trained = [], {}, {}

    for development, test, definition in evaluations(data, args.scenarios, .80):
        cache_key = definition["development_identity"]
        if cache_key not in trained:
            train, valid = chronological_source_split(development)
            transform = QUICFlowImageTransform(config.input_fields).fit(train)
            train_x, valid_x = transform.transform(train), transform.transform(valid)
            train_y, valid_y = labels(train), labels(valid)
            with ResourceMonitor(device) as fit_mem:
                started = time.perf_counter()
                model, training_audit = train_backbone(
                    train_x, train_y, valid_x, valid_y, config, device
                )
                template_x, template_y = select_templates(
                    train_x, train_y, config.templates_per_class, config.seed
                )
                k, k_trials = choose_k(
                    model, template_x, template_y, valid_x, valid_y, device
                )
                fit_seconds = time.perf_counter() - started
            trained[cache_key] = {
                "model": model, "transform": transform,
                "template_x": template_x, "template_y": template_y, "k": k,
                "fit_seconds": fit_seconds,
                "fit_peak_ram_mb": fit_mem.peak_ram_mb,
                "fit_peak_gpu_mb": fit_mem.peak_gpu_mb,
                "train_n": len(train), "valid_n": len(valid),
            }
            audits[f"development:{cache_key}"] = {
                **training_audit, "selected_k": k, "k_trials": k_trials,
                "features": transform.columns, "train_n": len(train),
                "validation_n": len(valid),
            }
        fitted = trained[cache_key]
        test_x, test_y = fitted["transform"].transform(test), labels(test)
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = template_predict(
                fitted["model"], fitted["template_x"], fitted["template_y"],
                test_x, fitted["k"], device,
            )
            predict_seconds = time.perf_counter() - started
        rows.append({
            **definition, "seed": args.seed, "method": "DFE-adapted",
            "train_n": fitted["train_n"], "validation_n": fitted["valid_n"],
            "n": len(test), "features": len(fitted["transform"].columns),
            "k": fitted["k"], **scores(test_y, prediction),
            "fit_seconds": fitted["fit_seconds"],
            "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "peak_ram_mb": max(fitted["fit_peak_ram_mb"], infer_mem.peak_ram_mb),
            "peak_gpu_mb": max(fitted["fit_peak_gpu_mb"], infer_mem.peak_gpu_mb),
        })
        audits[definition["protocol"]] = {
            "development_n": len(development), "test_n": len(test),
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "development_identity": cache_key,
        }
    return save_run(args.output, "DFE", rows, audits, {
        "dataset": "CESNET-QUICEXT-25", "config": asdict(config),
        "scenarios": args.scenarios, "seed": args.seed, "device": str(device),
        "adaptation_budget": 0,
        "protocol": "source templates only and complete immutable target test",
    })


def parse_args():
    dataset = CDR_MLC / "DATASETS/CESNET-QUICEXT-25/processed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset)
    parser.add_argument("--output", type=Path, default=CDR_MLC / "outputs/quicext25_dfe")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
