"""Quick causal warm-start ablation for MF-CDR-MLC.

The warm-start variant uses every causally available record in each sequence
partition. Partial windows are applied consistently during expert/router,
utility, meta-fusion, selection, and inference stages. No padding, future
records, or cross-partition state is used.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import compare_clean_valid as fixed_module
import congestion_feature_cdr_mlc as context_module
import congestion_selective_router_cdr_mlc as enhanced_module
import learned_router_cdr_mlc as learned_module
from adaptive_cdr_mlc import APPLICATIONS, STATS, load_dataset
from compare_clean_valid import TIMING, metrics
from confirmatory_rf_cdr_mlc import build_evaluations
from congestion_feature_cdr_mlc import (
    DEFAULT_CONGESTION_FEATURES,
    _context_schema,
    _rolling_slope,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    fit_meta_stacker,
    predict_all,
)


def warm_trend_frame(frame, features, window: int, engine: str = "vectorized"):
    """Causal trailing TTFEF with expanding prefixes before a full window."""
    features = list(features)
    columns = [f"{feature}_{stat}" for feature in features for stat in STATS]
    chunks = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        numeric = group[features].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric).all(axis=1) & numeric.ge(0).all(axis=1)
        segments = (~valid).cumsum()
        for _, segment in group.loc[valid].groupby(segments[valid], sort=False):
            values = segment[features].apply(pd.to_numeric, errors="coerce")
            output = {}
            for feature in features:
                rolling = values[feature].rolling(window, min_periods=1)
                for stat in STATS:
                    output[f"{feature}_{stat}"] = (
                        rolling.std(ddof=0)
                        if stat == "std"
                        else getattr(rolling, stat)()
                    )
            matrix = pd.DataFrame(output, index=segment.index)
            matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
            if len(matrix):
                joined = segment.loc[matrix.index].copy()
                joined[columns] = matrix[columns]
                chunks.append(joined)
    if not chunks:
        return frame.iloc[:0].assign(
            **{column: pd.Series(dtype=float) for column in columns}
        )
    return pd.concat(chunks).sort_index(kind="stable")


def warm_congestion_feature_frame(frame, config):
    """Causal congestion descriptors with expanding prefix windows."""
    available, columns = _context_schema(frame, config)
    chunks = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        numeric = group[available].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric).all(axis=1) & numeric.ge(0).all(axis=1)
        segments = (~valid).cumsum()
        for _, segment in group.loc[valid].groupby(segments[valid], sort=False):
            values = segment[available].apply(pd.to_numeric, errors="coerce")
            output = {}
            for feature in available:
                series = values[feature]
                rolling = series.rolling(config.window, min_periods=1)
                median = rolling.median()
                mean = rolling.mean()
                std = rolling.std(ddof=0)
                minimum = rolling.min()
                maximum = rolling.max()
                q25 = rolling.quantile(0.25)
                q75 = rolling.quantile(0.75)
                first = rolling.apply(lambda values: values[0], raw=True)
                scale = median.abs() + config.epsilon
                output[f"router_{feature}_log_median"] = np.log1p(
                    median.clip(lower=0)
                )
                output[f"router_{feature}_cv"] = (
                    std / (mean.abs() + config.epsilon)
                )
                output[f"router_{feature}_iqr_ratio"] = (q75 - q25) / scale
                output[f"router_{feature}_range_ratio"] = (
                    maximum - minimum
                ) / scale
                output[f"router_{feature}_delta_ratio"] = (
                    series - first
                ) / scale
                output[f"router_{feature}_slope_ratio"] = (
                    rolling.apply(_rolling_slope, raw=True) / scale
                )
            matrix = pd.DataFrame(output, index=segment.index)
            matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
            if len(matrix):
                joined = segment.loc[matrix.index].copy()
                joined[columns] = matrix[columns]
                chunks.append(joined)
    if not chunks:
        empty = frame.iloc[:0].copy()
        return empty.assign(
            **{column: pd.Series(dtype=float) for column in columns}
        ), available
    return pd.concat(chunks).sort_index(kind="stable"), available


@contextmanager
def warm_start_features():
    """Temporarily install warm-start feature builders in the MF call graph."""
    original_fixed = fixed_module.trend_frame
    original_learned = learned_module.trend_frame
    original_context = enhanced_module.congestion_feature_frame
    fixed_module.trend_frame = warm_trend_frame
    learned_module.trend_frame = warm_trend_frame
    enhanced_module.congestion_feature_frame = warm_congestion_feature_frame
    try:
        yield
    finally:
        fixed_module.trend_frame = original_fixed
        learned_module.trend_frame = original_learned
        enhanced_module.congestion_feature_frame = original_context


def fit_and_predict(development, test, config, warm_start: bool):
    start = perf_counter()
    if warm_start:
        with warm_start_features():
            model = fit_meta_stacker(development, config)
            fit_seconds = perf_counter() - start
            predict_start = perf_counter()
            result = predict_all(model, test)
    else:
        model = fit_meta_stacker(development, config)
        fit_seconds = perf_counter() - start
        predict_start = perf_counter()
        result = predict_all(model, test)
    predict_seconds = perf_counter() - predict_start
    return model, result, fit_seconds, predict_seconds


def score_row(mode, scope, observed, prediction, denominator, fit_s, infer_s):
    report = metrics(
        observed.traffic_label.to_numpy(),
        np.asarray(prediction),
        APPLICATIONS,
    )
    return {
        "Mode": mode,
        "Scope": scope,
        "N": report["n"],
        "Cov": report["n"] / denominator if denominator else np.nan,
        "Acc": report["accuracy"],
        "BAcc": report["balanced_accuracy"],
        "MF1": report["macro_f1"],
        "WF1": report["weighted_f1"],
        "FitS": fit_s,
        "InfS": infer_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--scenario", choices=["1", "2", "3"], default="1")
    parser.add_argument("--target-test-fraction", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/causal_warm_start_ablation",
    )
    args = parser.parse_args()
    if not 0 < args.target_test_fraction <= 1:
        parser.error("--target-test-fraction must be in (0,1]")

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([
        *TIMING, *DEFAULT_CONGESTION_FEATURES
    ]))
    data, audit = load_dataset(args.data_dir, candidates)
    audit.to_csv(args.output / "input_audit.csv", index=False)
    development, test, definition = build_evaluations(
        data,
        [args.scenario],
        [],
        0.80,
        args.target_test_fraction,
    )[0]
    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(DEFAULT_CONGESTION_FEATURES),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()

    print(f"FIT full-window: {definition['name']}", flush=True)
    full_model, full, full_fit, full_infer = fit_and_predict(
        development, test, config, False
    )
    print(f"FIT causal warm-start: {definition['name']}", flush=True)
    warm_model, warm, warm_fit, warm_infer = fit_and_predict(
        development, test, config, True
    )

    full_observed = full["observed"]
    warm_observed = warm["observed"]
    full_prediction = pd.Series(
        full["CDR_MLC_meta_stacker"], index=full_observed.index
    )
    warm_prediction = pd.Series(
        warm["CDR_MLC_meta_stacker"], index=warm_observed.index
    )
    common = full_observed.index.intersection(warm_observed.index)
    early = warm_observed.index.difference(full_observed.index)
    denominator = len(warm_observed)

    rows = [
        score_row(
            "Full", "Common", full_observed.loc[common],
            full_prediction.loc[common], denominator, full_fit, full_infer,
        ),
        score_row(
            "Warm", "Common", warm_observed.loc[common],
            warm_prediction.loc[common], denominator, warm_fit, warm_infer,
        ),
        score_row(
            "Warm", "All", warm_observed,
            warm_prediction, denominator, warm_fit, warm_infer,
        ),
    ]
    if len(early):
        rows.append(score_row(
            "Warm", "Cold", warm_observed.loc[early],
            warm_prediction.loc[early], denominator, warm_fit, warm_infer,
        ))

    results = pd.DataFrame(rows)
    results.to_csv(args.output / "warm_start_comparison.csv", index=False)
    manifest = {
        "scenario": definition,
        "config": asdict(config),
        "full_rows": len(full_observed),
        "warm_rows": len(warm_observed),
        "common_rows": len(common),
        "cold_start_rows": len(early),
        "warm_start_policy": {
            "causal": True,
            "padding": False,
            "cross_sequence_state": False,
            "partial_ttfef_windows": True,
            "partial_congestion_windows": True,
        },
        "full_partition_rows": full_model["partition_rows"],
        "warm_partition_rows": warm_model["partition_rows"],
    }
    (args.output / "warm_start_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nColumn abbreviations")
    print("  Mode  Full=complete windows; Warm=causal expanding windows")
    print(" Scope  Common=shared rows; All=all warm rows; Cold=added prefix rows")
    print("     N  evaluated rows")
    print("   Cov  fraction of all warm-start rows")
    print("   Acc  accuracy")
    print("  BAcc  balanced accuracy")
    print("   MF1  macro-F1")
    print("   WF1  weighted-F1")
    print("  FitS  training seconds")
    print("  InfS  inference seconds")
    print("\nResults")
    display = results.copy()
    for column in ("Cov", "Acc", "BAcc", "MF1", "WF1"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    for column in ("FitS", "InfS"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
