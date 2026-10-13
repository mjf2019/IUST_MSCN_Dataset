"""Run the clustering comparison over multiple causal TTFEF windows.

The child benchmark remains responsible for all leakage controls, balanced
sampling, clustering metrics, and per-window artifacts. This runner adds
explicit start/completion logs and produces one combined CSV.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Sweep clustering algorithms over TTFEF window sizes"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 15, 20],
    )
    parser.add_argument("--development-fraction", type=float, default=0.80)
    parser.add_argument("--samples-per-level", type=int, default=5000)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument(
        "--dbscan-eps-grid",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00],
    )
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--spectral-neighbors", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tune-mbk",
        action="store_true",
        help="Optional label-free MBK tuning; omit for the paper-fixed MBK",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/clustering_window_sweep",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    windows = list(dict.fromkeys(args.windows))
    if not windows or any(window < 1 for window in windows):
        raise ValueError("--windows must contain positive unique integers")
    benchmark = Path(__file__).with_name(
        "clustering_algorithm_comparison.py"
    )
    if not benchmark.exists():
        raise FileNotFoundError(benchmark)

    args.output.mkdir(parents=True, exist_ok=True)
    combined, runtime_rows = [], []
    total_start = perf_counter()

    for position, window in enumerate(windows, start=1):
        child_output = args.output / f"window_{window}"
        command = [
            sys.executable,
            str(benchmark),
            "--data-dir", str(args.data_dir),
            "--development-fraction", str(args.development_fraction),
            "--window", str(window),
            "--samples-per-level", str(args.samples_per_level),
            "--silhouette-sample-size",
            str(args.silhouette_sample_size),
            "--dbscan-min-samples", str(args.dbscan_min_samples),
            "--spectral-neighbors", str(args.spectral_neighbors),
            "--threads", str(args.threads),
            "--seed", str(args.seed),
            "--output", str(child_output),
            "--dbscan-eps-grid",
            *[str(value) for value in args.dbscan_eps_grid],
        ]
        if args.tune_mbk:
            command.append("--tune-mbk")

        print(
            f"\n[{position}/{len(windows)}] START window={window}",
            flush=True,
        )
        started = perf_counter()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            elapsed = perf_counter() - started
            print(
                f"[{position}/{len(windows)}] FAILED window={window} "
                f"after {elapsed:.2f}s (exit={error.returncode})",
                flush=True,
            )
            raise
        elapsed = perf_counter() - started
        result_path = child_output / "clustering_comparison.csv"
        if not result_path.exists():
            raise FileNotFoundError(
                f"window {window} completed without {result_path}"
            )
        frame = pd.read_csv(result_path)
        frame.insert(0, "window", window)
        combined.append(frame)
        runtime_rows.append({
            "window": int(window),
            "elapsed_seconds": float(elapsed),
            "output": str(child_output),
            "status": "completed",
        })
        print(
            f"[{position}/{len(windows)}] COMPLETED window={window} "
            f"in {elapsed:.2f}s -> {child_output}",
            flush=True,
        )

    combined_frame = pd.concat(combined, ignore_index=True)
    combined_path = args.output / "clustering_window_comparison.csv"
    combined_frame.to_csv(combined_path, index=False)
    runtime_frame = pd.DataFrame(runtime_rows)
    runtime_frame.to_csv(
        args.output / "window_runtime.csv", index=False
    )

    total_elapsed = perf_counter() - total_start
    print(
        f"\nALL WINDOWS COMPLETED: {len(windows)} windows in "
        f"{total_elapsed:.2f}s",
        flush=True,
    )
    print(f"Combined results: {combined_path}", flush=True)

    display = combined_frame[[
        "window", "abbreviation", "n", "clusters",
        "noise_fraction", "silhouette", "davies_bouldin",
        "adjusted_rand", "normalized_mutual_info",
        "mapped_accuracy", "fit_seconds",
        "native_out_of_sample_predict",
    ]].copy()
    display.columns = [
        "W", "M", "N", "K", "Noise", "Sil", "DB",
        "ARI", "NMI", "MAcc", "FitS", "Pred",
    ]
    for column in ["Noise", "Sil", "DB", "ARI", "NMI", "MAcc"]:
        display[column] = display[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4f}"
        )
    display["FitS"] = display["FitS"].map(
        lambda value: f"{value:.2f}"
    )
    display["Pred"] = display["Pred"].map(
        {True: "Yes", False: "No"}
    )
    print("\nCombined window results")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
