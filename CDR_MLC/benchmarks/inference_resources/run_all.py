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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=HERE / "artifacts/models")
    parser.add_argument(
        "--sample", type=Path,
        default=HERE / "artifacts/medium_inference_sample.pkl",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
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
            "--cpu-threads", str(args.cpu_threads),
            "--warmup-runs", str(args.warmup_runs),
            "--repeats", str(args.repeats),
        ]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)
        frames.append(pd.read_csv(destination / "summary.csv"))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output / "inference_resource_summary.csv", index=False)
    print("\nCombined CPU-only inference results")
    columns = [
        "method", "evaluated_rows", "accuracy", "macro_f1",
        "latency_batch_p50_seconds", "latency_batch_p95_seconds",
        "mean_us_per_evaluated_row", "throughput_evaluated_rows_per_second",
        "peak_rss_delta_mb", "model_bytes",
    ]
    print(combined[columns].to_string(index=False))


if __name__ == "__main__":
    main()
