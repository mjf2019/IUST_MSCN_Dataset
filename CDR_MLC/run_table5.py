#!/usr/bin/env python3
"""Run the five Table 5 scenarios with the leakage-safe CDR-MLC model."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cdr_mlc import CDRMLC, CDRMLCConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def scenario_paths(data_root: Path) -> dict[str, tuple[Path, Path]]:
    short = data_root / "Short"
    long = data_root / "Long"
    short_all = next(
        (path for path in (short / "CDR-MLC-Shuffle.csv", short / "CDR_MLC_Shuffle.csv") if path.exists()),
        short / "CDR-MLC-Shuffle.csv",
    )
    long_all = next(
        (path for path in (long / "CDR-MLC-Shuffle.csv", long / "CDR_MLC_Shuffle.csv") if path.exists()),
        long / "CDR-MLC-Shuffle.csv",
    )
    return {
        "scenario_1": (short / "level_1.csv", short / "level_2.csv"),
        "scenario_2": (short / "level_1.csv", short / "level_3.csv"),
        "scenario_3": (short / "level_2.csv", short / "level_3.csv"),
        "scenario_4": (short_all, long_all),
        "scenario_5": (long_all, short_all),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "DATASETS/CDR-MLC/scale_0.001",
        help="Directory containing Short/ and Long/",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=[f"scenario_{index}" for index in range(1, 6)],
        default=[f"scenario_{index}" for index in range(1, 6)],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/runs/table5",
        help="Generated files go here; results/runs is ignored by Git.",
    )
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Table 5 dataset is missing: {path}")
    header = pd.read_csv(path, nrows=0)
    if "label" not in header.columns:
        raise ValueError(f"Required target column 'label' is missing from {path}")
    ignored = {"IdleTime"}
    columns = [column for column in header.columns if column not in ignored]
    dtypes = {column: np.float32 for column in columns if column != "label"}
    try:
        frame = pd.read_csv(path, usecols=columns, dtype=dtypes)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric feature or invalid value in {path}: {error}") from error
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"NaN or infinite numeric value found in {path}")
    return frame


def run_once(
    scenario: str,
    train_path: Path,
    test_path: Path,
    seed: int,
    window_size: int,
    n_estimators: int,
) -> dict[str, object]:
    print(f"[{scenario} seed={seed}] loading training data: {train_path}", flush=True)
    train = load_frame(train_path)
    config = CDRMLCConfig(
        window_size=window_size,
        n_estimators=n_estimators,
        random_state=seed,
    )
    model = CDRMLC(config)

    print(f"[{scenario} seed={seed}] fitting {len(train):,} samples", flush=True)
    fit_started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - fit_started
    train_samples = len(train)
    del train
    gc.collect()

    # Load the test set only after fitting, avoiding simultaneous train/test
    # DataFrames in memory for the full-scale experiments.
    print(f"[{scenario} seed={seed}] loading test data: {test_path}", flush=True)
    test = load_frame(test_path)
    y_test = test.pop("label").to_numpy()
    print(f"[{scenario} seed={seed}] predicting {len(test):,} samples", flush=True)
    predict_started = time.perf_counter()
    predictions, routes = model.predict_with_routes(test)
    predict_seconds = time.perf_counter() - predict_started

    return {
        "scenario": scenario,
        "seed": seed,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_samples": train_samples,
        "test_samples": len(test),
        "config": asdict(config),
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        },
        "timing": {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "microseconds_per_sample": predict_seconds * 1_000_000 / len(test),
            "samples_per_second": len(test) / predict_seconds,
        },
        "route_counts": {
            str(cluster): int(count)
            for cluster, count in zip(*np.unique(routes, return_counts=True))
        },
        "model": model.metadata(),
    }


def main() -> None:
    args = parse_args()
    paths = scenario_paths(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in args.scenarios:
        train_path, test_path = paths[scenario]
        for seed in args.seeds:
            result = run_once(
                scenario,
                train_path,
                test_path,
                seed,
                args.window_size,
                args.n_estimators,
            )
            results.append(result)
            output = args.output_dir / f"{scenario}_seed_{seed}.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                scenario,
                f"seed={seed}",
                f"accuracy={result['metrics']['accuracy']:.6f}",
                f"f1={result['metrics']['f1_weighted']:.6f}",
            )

    rows = []
    for result in results:
        rows.append(
            {
                "scenario": result["scenario"],
                "seed": result["seed"],
                **result["metrics"],
                **result["timing"],
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "runs.csv", index=False)


if __name__ == "__main__":
    main()
