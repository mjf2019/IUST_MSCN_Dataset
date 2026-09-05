#!/usr/bin/env python3
"""Run the five Table 5 scenarios with the leakage-safe CDR-MLC model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cdr_mlc import CDRMLC, CDRMLCConfig


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
        default=Path("DATASETS/CDR-MLC/scale_1"),
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
    parser.add_argument("--output-dir", type=Path, default=Path("CDR_MLC/results/table5"))
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Table 5 dataset is missing: {path}")
    return pd.read_csv(path)


def run_once(
    scenario: str,
    train_path: Path,
    test_path: Path,
    seed: int,
    window_size: int,
    n_estimators: int,
) -> dict[str, object]:
    train = load_frame(train_path)
    test = load_frame(test_path)
    y_test = test["label"].to_numpy()
    config = CDRMLCConfig(
        window_size=window_size,
        n_estimators=n_estimators,
        random_state=seed,
    )
    model = CDRMLC(config)

    fit_started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    predictions, routes = model.predict_with_routes(test.drop(columns=["label"]))
    predict_seconds = time.perf_counter() - predict_started

    return {
        "scenario": scenario,
        "seed": seed,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_samples": len(train),
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
