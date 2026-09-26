"""Run the adapted DFE benchmark on the seven common IUST_MSCN protocols.

The DFE backbone and template classifier are imported unchanged from the
standalone DFE implementation.  This Table-5 runner uses zero target labels and
the common leakage-safe S1--S7 partitions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import APPLICATIONS  # noqa: E402
from dfe_iust_mscn import (  # noqa: E402
    DFEConfig, FlowImageTransform, chronological_source_split, choose_k,
    metrics, seed_all, select_templates, template_predict, train_backbone,
)
from benchmarks.console_output import ResourceMonitor, print_compact_results  # noqa: E402
from benchmarks.deep_common import (  # noqa: E402
    PROTOCOL_IDS, evaluations, load_clean_valid, record_ids,
)


def run(args):
    config = DFEConfig(epochs=args.epochs, seed=args.seed)
    seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_clean_valid(args.data_dir)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    encoder = LabelEncoder().fit(APPLICATIONS)
    rows, audits = [], {}

    for development, test, definition in evaluations(
        data, args.scenarios, train_fraction=.80, target_test_fraction=.20
    ):
        protocol = definition["protocol"]
        train, validation = chronological_source_split(development)
        transform = FlowImageTransform(config.input_fields).fit(train)
        train_x = transform.transform(train)
        validation_x = transform.transform(validation)
        test_x = transform.transform(test)
        train_y = encoder.transform(train.traffic_label.astype(str))
        validation_y = encoder.transform(validation.traffic_label.astype(str))
        test_y = encoder.transform(test.traffic_label.astype(str))

        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            model, training_audit = train_backbone(
                train_x, train_y, validation_x, validation_y, config, device
            )
            template_x, template_y = select_templates(
                train_x, train_y, config.templates_per_class, config.seed
            )
            k, k_trials = choose_k(
                model, template_x, template_y, validation_x, validation_y, device
            )
            fit_seconds = time.perf_counter() - started

        with ResourceMonitor(device) as infer_mem:
            started = time.perf_counter()
            prediction = template_predict(
                model, template_x, template_y, test_x, k, device
            )
            predict_seconds = time.perf_counter() - started

        rows.append({
            "protocol": protocol, "scenario": protocol.removeprefix("S"),
            "source": definition["source"], "target": definition["target"],
            "method": "DFE-adapted", "seed": args.seed,
            "target_labeled_n": 0, "train_n": len(train),
            "validation_n": len(validation), "n": len(test),
            "templates_total": len(template_y), "k": k,
            **training_audit, **metrics(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "peak_ram_mb": max(fit_mem.peak_ram_mb, infer_mem.peak_ram_mb),
            "peak_gpu_mb": max(fit_mem.peak_gpu_mb, infer_mem.peak_gpu_mb),
        })
        audits[protocol] = {
            "definition": definition, "features": transform.columns,
            "selected_k": k, "k_trials": k_trials,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "train_test_overlap": len(record_ids(train) & record_ids(test)),
        }

    summary = pd.DataFrame(rows).sort_values("scenario")
    summary.to_csv(args.output / "results.csv", index=False)
    (args.output / "audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "run_manifest.json").write_text(json.dumps({
        "method": "DFE-adapted", "config": asdict(config),
        "protocols": args.scenarios, "target_label_budget": 0,
        "seed": args.seed, "device": str(device),
        "shared_test_protocol": "common leakage-safe S1-S7",
    }, indent=2) + "\n", encoding="utf-8")
    print_compact_results(
        summary, method="DFE", calibration_column="target_labeled_n",
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
