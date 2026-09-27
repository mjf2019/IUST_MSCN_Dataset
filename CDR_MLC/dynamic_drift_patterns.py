"""Evaluate RF, CDR-MLC and MF-CDR-MLC on three dynamic drift streams.

The common development partition is the ordered prefix of every original
sequence at all three congestion levels.  Dynamic streams are constructed
only from the disjoint ordered tails.  D1 represents gradual drift, D2
recurring drift, and D3 mixed gradual/sudden drift.  Models remain frozen
throughout every stream.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, LEVELS, load_dataset
from compare_clean_valid import TIMING, fit_rf, metrics, predict_fixed_cdr, predict_rf
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from confirmatory_rf_cdr_mlc import fit_original_cdr
from meta_stacked_cdr_mlc_leakage_safe import MetaStackConfig, fit_meta_stacker, predict_all


METHODS = ("RF", "CDR-MLC", "MF-CDR-MLC")
SCENARIOS = {
    "D1": {
        "name": "Gradual",
        "blocks": [
            ("G1-100L", {"Low": 1.0}),
            ("G2-80L20H", {"Low": 0.8, "High": 0.2}),
            ("G3-60L40H", {"Low": 0.6, "High": 0.4}),
            ("G4-40L60H", {"Low": 0.4, "High": 0.6}),
            ("G5-20L80H", {"Low": 0.2, "High": 0.8}),
            ("G6-100H", {"High": 1.0}),
        ],
    },
    "D2": {
        "name": "Recurring",
        "blocks": [
            ("R1-Low", {"Low": 1.0}),
            ("R2-High", {"High": 1.0}),
            ("R3-Low", {"Low": 1.0}),
        ],
    },
    "D3": {
        "name": "Mixed",
        "blocks": [
            ("M1-100L", {"Low": 1.0}),
            ("M2-80L20M", {"Low": 0.8, "Medium": 0.2}),
            ("M3-50L50M", {"Low": 0.5, "Medium": 0.5}),
            ("M4-20L80M", {"Low": 0.2, "Medium": 0.8}),
            ("M5-High", {"High": 1.0}),
            ("M6-Medium", {"Medium": 1.0}),
        ],
    },
}


def ordered_development_tail(data: pd.DataFrame, train_fraction: float):
    development, tails = [], []
    for sequence_id, group in data.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * train_fraction)
        if not 0 < cut < len(group):
            raise ValueError(f"{sequence_id}: insufficient rows for ordered split")
        development.append(group.iloc[:cut].copy())
        tails.append(group.iloc[cut:].copy())
    development = pd.concat(development, ignore_index=True)
    tails = pd.concat(tails, ignore_index=True)
    dev_ids = set(zip(development.source_file, development.source_row))
    tail_ids = set(zip(tails.source_file, tails.source_row))
    if dev_ids & tail_ids:
        raise RuntimeError("development/test-tail overlap detected")
    return development, tails


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    levels = list(weights)
    raw = np.asarray([weights[level] for level in levels], dtype=float)
    raw = raw / raw.sum() * total
    counts = np.floor(raw).astype(int)
    for position in np.argsort(-(raw - counts))[: total - counts.sum()]:
        counts[position] += 1
    return dict(zip(levels, counts.tolist()))


def _interleave(chunks: dict[str, pd.DataFrame], rng: np.random.Generator):
    queues = {key: deque(frame.to_dict("records")) for key, frame in chunks.items()}
    schedule = [key for key, queue in queues.items() for _ in range(len(queue))]
    rng.shuffle(schedule)
    return [queues[key].popleft() for key in schedule]


def build_stream(
    tails: pd.DataFrame,
    scenario: str,
    block_rows: int,
    seed: int,
):
    if block_rows < len(APPLICATIONS) or block_rows % len(APPLICATIONS):
        raise ValueError("--block-rows must be a positive multiple of application count")
    definition = SCENARIOS[scenario]
    pools = {}
    cursors = defaultdict(int)
    for level in LEVELS:
        for application in APPLICATIONS:
            pool = tails[
                tails.congestion_level.eq(level)
                & tails.traffic_label.eq(application)
            ].sort_values(["timestamp", "source_row"], kind="stable")
            pools[level, application] = pool.reset_index(drop=True)

    rng = np.random.default_rng(seed + 1000 * int(scenario[1:]))
    per_application = block_rows // len(APPLICATIONS)
    stream_rows = []
    app_clock = defaultdict(int)
    segment_audit = []

    for segment_order, (segment, weights) in enumerate(definition["blocks"], 1):
        application_rows = {}
        level_counts = defaultdict(int)
        for application in APPLICATIONS:
            allocation = _allocate(per_application, weights)
            chunks = {}
            for level, count in allocation.items():
                key = (level, application)
                start, stop = cursors[key], cursors[key] + count
                if stop > len(pools[key]):
                    raise ValueError(
                        f"{scenario}/{segment}: insufficient {level}/{application} "
                        f"tail rows; need {stop}, have {len(pools[key])}. "
                        "Reduce --block-rows."
                    )
                chunks[level] = pools[key].iloc[start:stop].copy()
                cursors[key] = stop
                level_counts[level] += count
            application_rows[application] = deque(_interleave(chunks, rng))

        # Round-robin applications while preserving each application's internal order.
        for _ in range(per_application):
            for application in APPLICATIONS:
                row = application_rows[application].popleft()
                row["dynamic_scenario"] = scenario
                row["drift_type"] = definition["name"]
                row["drift_segment"] = segment
                row["segment_order"] = segment_order
                row["original_sequence_id"] = row["sequence_id"]
                row["sequence_id"] = f"{scenario}::{application}"
                row["timestamp"] = app_clock[application]
                app_clock[application] += 1
                stream_rows.append(row)
        segment_audit.append({
            "scenario": scenario,
            "drift_type": definition["name"],
            "segment": segment,
            "segment_order": segment_order,
            "rows": block_rows,
            "level_counts": dict(level_counts),
        })

    stream = pd.DataFrame(stream_rows).reset_index(drop=True)
    identities = list(zip(stream.source_file, stream.source_row))
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"{scenario}: a held-out record was reused within the stream")
    return stream, segment_audit


def fit_models(development: pd.DataFrame, config: MetaStackConfig, rf_trees: int):
    start = perf_counter()
    mf = fit_meta_stacker(development, config)
    mf_fit = perf_counter() - start

    start = perf_counter()
    original = fit_original_cdr(development, config)
    original_fit = perf_counter() - start
    if list(original["source_eligible_index"]) != list(mf["source_eligible_index"]):
        raise RuntimeError("CDR-MLC and MF-CDR-MLC eligibility differs")

    eligible = development.loc[mf["source_eligible_index"]]
    start = perf_counter()
    rf = fit_rf(eligible, (), config.random_state, rf_trees)
    rf_fit = perf_counter() - start
    return {"RF": rf, "CDR-MLC": original, "MF-CDR-MLC": mf}, {
        "RF": rf_fit, "CDR-MLC": original_fit, "MF-CDR-MLC": mf_fit
    }


def evaluate_stream(models, fit_seconds, stream: pd.DataFrame, scenario: str, seed: int):
    start = perf_counter()
    mf_result = predict_all(models["MF-CDR-MLC"], stream)
    mf_seconds = perf_counter() - start
    observed = mf_result["observed"]

    start = perf_counter()
    original_all = predict_fixed_cdr(models["CDR-MLC"], stream)
    original_seconds = perf_counter() - start
    original = original_all.loc[observed.index].to_numpy()
    embedded = mf_result["CDR_MLC_actual_router"]
    if not np.array_equal(original, embedded):
        raise RuntimeError(f"{scenario}: independent and embedded CDR-MLC differ")

    start = perf_counter()
    rf_all = predict_rf(models["RF"], stream)
    rf_seconds = perf_counter() - start
    rf = pd.Series(rf_all, index=stream.index).loc[observed.index].to_numpy()

    predictions = {
        "RF": rf,
        "CDR-MLC": original,
        "MF-CDR-MLC": mf_result["CDR_MLC_meta_stacker"],
    }
    prediction_seconds = {
        "RF": rf_seconds,
        "CDR-MLC": original_seconds,
        "MF-CDR-MLC": mf_seconds,
    }
    truth = observed.traffic_label.to_numpy()
    overall, segments = [], []
    for method in METHODS:
        score = metrics(truth, predictions[method], APPLICATIONS)
        overall.append({
            "scenario": scenario,
            "drift_type": SCENARIOS[scenario]["name"],
            "seed": seed,
            "method": method,
            **{key: score[key] for key in (
                "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
            )},
            "fit_seconds": fit_seconds[method],
            "predict_seconds": prediction_seconds[method],
        })
        prediction = pd.Series(predictions[method], index=observed.index)
        for (order, segment), part in observed.groupby(
            ["segment_order", "drift_segment"], sort=True
        ):
            segment_score = metrics(
                part.traffic_label.to_numpy(), prediction.loc[part.index].to_numpy(),
                APPLICATIONS,
            )
            segments.append({
                "scenario": scenario,
                "drift_type": SCENARIOS[scenario]["name"],
                "segment_order": int(order),
                "segment": segment,
                "seed": seed,
                "method": method,
                **{key: segment_score[key] for key in (
                    "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
                )},
            })
    return overall, segments, observed


def print_scenario(overall, segments):
    scenario = overall[0]["scenario"]
    drift_type = overall[0]["drift_type"]
    print(f"\nCOMPLETED {scenario} ({drift_type})", flush=True)
    print("Overall: Acc=accuracy, MF1=macro-F1", flush=True)
    print(pd.DataFrame(overall)[["method", "n", "accuracy", "macro_f1"]]
          .rename(columns={"method": "M", "accuracy": "Acc", "macro_f1": "MF1"})
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("Segments", flush=True)
    print(pd.DataFrame(segments)[
        ["segment_order", "segment", "method", "n", "accuracy", "macro_f1"]
    ].rename(columns={
        "segment_order": "Ord", "segment": "Seg", "method": "M",
        "accuracy": "Acc", "macro_f1": "MF1",
    }).to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/dynamic_drift_patterns")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--block-rows", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be in (0,1)")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct integers")

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    development, tails = ordered_development_tail(data, args.train_fraction)

    all_overall, all_segments, audits = [], [], []
    for seed in args.seeds:
        config = MetaStackConfig(
            window=args.window,
            congestion_window=args.congestion_window,
            congestion_features=tuple(args.congestion_features),
            expert_trees=args.expert_trees,
            utility_trees=args.utility_trees,
            meta_trees=args.meta_trees,
            random_state=seed,
        ).validate()
        print(f"FIT common development models seed={seed}", flush=True)
        models, fit_seconds = fit_models(development, config, args.rf_trees)
        for scenario in args.scenarios:
            print(f"START {scenario} ({SCENARIOS[scenario]['name']}) seed={seed}", flush=True)
            stream, segment_audit = build_stream(tails, scenario, args.block_rows, seed)
            overall, segments, observed = evaluate_stream(
                models, fit_seconds, stream, scenario, seed
            )
            all_overall.extend(overall)
            all_segments.extend(segments)
            audits.append({
                "scenario": scenario,
                "seed": seed,
                "raw_stream_rows": len(stream),
                "evaluated_rows": len(observed),
                "segments": segment_audit,
            })
            print_scenario(overall, segments)

    overall_frame = pd.DataFrame(all_overall)
    segment_frame = pd.DataFrame(all_segments)
    overall_frame.to_csv(args.output / "dynamic_drift_overall.csv", index=False)
    segment_frame.to_csv(args.output / "dynamic_drift_segments.csv", index=False)
    summary = overall_frame.groupby(
        ["scenario", "drift_type", "method"]
    )[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output / "dynamic_drift_summary.csv", index=False)
    manifest = {
        "arguments": vars(args) | {"data_dir": str(args.data_dir), "output": str(args.output)},
        "methods": list(METHODS),
        "scenario_definitions": SCENARIOS,
        "development_rows": len(development),
        "held_out_tail_rows": len(tails),
        "leakage_controls": [
            "ordered development prefixes and test tails are disjoint",
            "streams use held-out tails only",
            "models are frozen before stream evaluation",
            "TTFEF uses synthetic per-application causal stream order",
            "all methods use identical evaluated records",
        ],
        "configuration": asdict(config),
        "runs": audits,
    }
    (args.output / "dynamic_drift_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"\nALL COMPLETED: {len(audits)} scenario/seed runs", flush=True)


if __name__ == "__main__":
    main()
