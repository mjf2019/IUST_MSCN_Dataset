"""DFE-adapted on a fixed external-dataset ordered 80/20 protocol."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

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
    evaluations, labels, load_clean_valid, record_ids, save_run, scores,
)


def run(args):
    config = DFEConfig(epochs=args.epochs, seed=args.seed)
    seed_all(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_clean_valid(args.data)
    dataset_name = str(input_audit.iloc[0]["dataset"])
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}
    for development, test, definition in evaluations(data, ["7"], .80):
        train, valid = chronological_source_split(development)
        transform = FlowImageTransform(config.input_fields).fit(train)
        train_x, valid_x, test_x = (
            transform.transform(train), transform.transform(valid),
            transform.transform(test),
        )
        train_y, valid_y, test_y = labels(train), labels(valid), labels(test)
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
        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = template_predict(
                model, template_x, template_y, test_x, k, device
            )
            predict_seconds = time.perf_counter() - started
        rows.append({
            **definition, "seed": args.seed, "method": "DFE-adapted",
            "train_n": len(train), "validation_n": len(valid), "n": len(test),
            "features": len(transform.columns), "k": k,
            **scores(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "peak_ram_mb": max(fit_mem.peak_ram_mb, infer_mem.peak_ram_mb),
            "peak_gpu_mb": max(fit_mem.peak_gpu_mb, infer_mem.peak_gpu_mb),
        })
        audits[definition["protocol"]] = {
            **training_audit, "selected_k": k, "k_trials": k_trials,
            "features": transform.columns,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
        }
    return save_run(args.output, "DFE", rows, audits, {
        "dataset": dataset_name, "config": asdict(config),
        "split": "ordered 80/20 per application sequence",
        "test_context_eligibility_window": 20,
        "adaptation_budget": 0, "seed": args.seed, "device": str(device),
    })


def parse_args():
    dataset = CDR_MLC.parent / "AMCAL/SDNCampus_TEST/Dataset/SDNCampus_original.csv"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=dataset)
    p.add_argument("--output", type=Path, default=CDR_MLC / "outputs/sdncampus_dfe")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
