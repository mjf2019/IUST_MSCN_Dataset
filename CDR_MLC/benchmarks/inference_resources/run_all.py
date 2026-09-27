"""Run every saved inference artifact in a fresh CPU-only subprocess."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

METHOD_CODES = {
    "rf": "RF",
    "original-cdr": "O-CDR",
    "MF-Sequential": "MF-Seq",
    "MF-Parallel": "MF-Par",
    "MF-Pipelined": "MF-Pipe",
    "1d-cnn": "1D-CNN",
    "ft-transformer": "FT-T",
    "graphsage": "G-SAGE",
    "scarf": "SCARF",
    "dfe": "DFE",
    "af": "AF",
}


def _fmt(series, digits):
    return series.map(lambda value: f"{float(value):.{digits}f}")


def print_compact_results(combined, mode):
    codes = combined["method"].map(
        lambda name: METHOD_CODES.get(str(name), str(name))
    )
    if mode == "streaming":
        display = pd.DataFrame({
            "M": codes,
            "N": combined["evaluated_rows"].astype(int),
            "Acc": _fmt(combined["accuracy"], 4),
            "MF1": _fmt(combined["macro_f1"], 4),
            "P50us": _fmt(combined["latency_record_p50_us"], 2),
            "P95us": _fmt(combined["latency_record_p95_us"], 2),
            "P99us": _fmt(combined["latency_record_p99_us"], 2),
            "R/s": _fmt(
                combined["throughput_evaluated_rows_per_second"], 2
            ),
            "RAM": _fmt(combined["peak_rss_delta_mb"], 2),
            "Size": _fmt(combined["model_bytes"] / (1024.0 ** 2), 2),
        })
        legend = [
            ("M", "Method"),
            ("N", "Evaluated records"),
            ("Acc", "Accuracy"),
            ("MF1", "Macro F1-score"),
            ("P50us", "Median per-record latency (us)"),
            ("P95us", "95th-percentile per-record latency (us)"),
            ("P99us", "99th-percentile per-record latency (us)"),
            ("R/s", "Evaluated records per second"),
            ("RAM", "Peak RSS increase (MiB)"),
            ("Size", "Serialized model size (MiB)"),
        ]
    else:
        display = pd.DataFrame({
            "M": codes,
            "N": combined["evaluated_rows"].astype(int),
            "Acc": _fmt(combined["accuracy"], 4),
            "MF1": _fmt(combined["macro_f1"], 4),
            "P50s": _fmt(combined["latency_batch_p50_seconds"], 4),
            "P95s": _fmt(combined["latency_batch_p95_seconds"], 4),
            "us/R": _fmt(combined["mean_us_per_evaluated_row"], 2),
            "R/s": _fmt(
                combined["throughput_evaluated_rows_per_second"], 2
            ),
            "RAM": _fmt(combined["peak_rss_delta_mb"], 2),
            "Size": _fmt(combined["model_bytes"] / (1024.0 ** 2), 2),
        })
        legend = [
            ("M", "Method"),
            ("N", "Evaluated records"),
            ("Acc", "Accuracy"),
            ("MF1", "Macro F1-score"),
            ("P50s", "Median batch latency (s)"),
            ("P95s", "95th-percentile batch latency (s)"),
            ("us/R", "Mean microseconds per evaluated record"),
            ("R/s", "Evaluated records per second"),
            ("RAM", "Peak RSS increase (MiB)"),
            ("Size", "Serialized model size (MiB)"),
        ]

    seen = set()
    for original, code in zip(combined["method"].astype(str), codes):
        if code not in seen:
            legend.append((code, original))
            seen.add(code)
    print("\nColumn abbreviations")
    print(pd.DataFrame(legend, columns=["Key", "Meaning"]).to_string(index=False))
    print("\nResults")
    print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=HERE / "artifacts/models")
    parser.add_argument(
        "--sample", type=Path,
        default=HERE / "artifacts/medium_inference_sample.pkl",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--mode", choices=("streaming", "batch"), default="streaming")
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument(
        "--forest-jobs", type=int, default=-1,
        help="sklearn forest n_jobs; -1 uses all available logical CPUs",
    )
    parser.add_argument(
        "--mf-branch-workers", type=int, default=3,
        help="concurrent MF expert/utility branch workers",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    index = json.loads((args.models / "index.json").read_text(encoding="utf-8"))
    entries = index["methods"]
    if args.methods:
        requested = set(args.methods)
        entries = [entry for entry in entries if entry["method"] in requested]
        missing = requested - {entry["method"] for entry in entries}
        if missing:
            raise ValueError(f"models not found in index: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(args.cpu_threads)

    frames = []
    for entry in entries:
        method = entry["method"]
        destination = args.output / method
        command = [
            sys.executable, str(HERE / "benchmark_method.py"),
            "--model", entry["artifact"],
            "--sample", str(args.sample),
            "--output", str(destination),
            "--mode", args.mode,
            "--cpu-threads", str(args.cpu_threads),
            "--forest-jobs", str(args.forest_jobs),
            "--mf-branch-workers", str(args.mf_branch_workers),
            "--warmup-runs", str(args.warmup_runs),
            "--repeats", str(args.repeats),
        ]
        print(f"RUN {method} ({args.mode})", flush=True)
        completed = subprocess.run(
            command, env=environment, capture_output=True, text=True
        )
        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout, file=sys.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(
                completed.returncode, command
            )
        frames.append(pd.read_csv(destination / "summary.csv"))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output / "inference_resource_summary.csv", index=False)
    print_compact_results(combined, args.mode)


if __name__ == "__main__":
    main()
