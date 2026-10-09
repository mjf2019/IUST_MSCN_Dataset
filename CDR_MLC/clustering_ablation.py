"""Paired full MF-CDR-MLC versus a label-free no-clustering expert bank.

No_Clustering fits neither MBK nor its scaler, trains three experts on stable
random development partitions, and removes centroid distances and route one-hot
features. Context descriptors, utility, fusion, fallback, chronological splits,
tree budgets and development-only selection are retained. No test tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import TIMING, metrics
from confirmatory_rf_cdr_mlc import build_evaluations
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from congestion_selective_router_cdr_mlc import _enhanced_outputs
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig, fit_meta_stacker, four_way_split, predict_all,
)
from mixed_level_protocols_leakage_safe import PROTOCOLS, frame_identity
from dynamic_drift_patterns import SCENARIOS as DYNAMIC_SCENARIOS, ordered_development_tail, build_stream


METHODS = ("MF_Full", "MF_No_Clustering")
SCORES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def check_pair(models, development, test):
    full, control = models
    if not full["use_clustering"] or control["use_clustering"]:
        raise RuntimeError("Pair must contain full and no-clustering variants")
    if control["router"] is not None or control["scaler"] is not None:
        raise RuntimeError("No-clustering variant must not fit MBK or its scaler")
    for key in ("numeric", "categorical", "source_eligible_index", "partition_rows", "congestion_config"):
        if full[key] != control[key]:
            raise RuntimeError(f"Pair changed {key}")
    for name, frame in [*four_way_split(development, full["meta_stack_config"]).items(), ("test", test)]:
        a = _enhanced_outputs(full, frame, full["congestion_config"])
        b = _enhanced_outputs(control, frame, control["congestion_config"])
        if frame_identity(a[0]) != frame_identity(b[0]) or a[-1] != b[-1]:
            raise RuntimeError(f"{name}: eligible records or context schema differ")
        count = len(a[-1])
        if not count:
            raise RuntimeError("Congestion descriptors must remain enabled")
        np.testing.assert_array_equal(a[-2][:, -count:], b[-2][:, -count:])
        if b[1].shape[1] != 0 or a[-2].shape[1] - b[-2].shape[1] != 6:
            raise RuntimeError("Must remove all three distances and three route indicators")
    for model in models:
        if len(model["experts"]) != 3:
            raise RuntimeError("Both variants must retain three experts")
        if any(e.n_estimators != model["meta_stack_config"].expert_trees
               for e in model["experts"].values()):
            raise RuntimeError("Expert tree budget changed")


def evaluate_pair(development, test, config):
    ids = lambda f: set(zip(f.source_file, f.source_row))
    if ids(development) & ids(test):
        raise RuntimeError("Development and test records overlap")
    models = [fit_meta_stacker(development, replace(config, use_clustering=value))
              for value in (True, False)]
    check_pair(models, development, test)
    results = [predict_all(model, test) for model in models]
    observed = results[0]["observed"]
    if frame_identity(observed) != frame_identity(results[1]["observed"]):
        raise RuntimeError("Scored records differ")
    prediction_columns = ["timestamp", "source_file", "source_row",
                          "traffic_label", "congestion_level"]
    prediction_columns += [name for name in ("dynamic_scenario", "drift_segment", "segment_order")
                           if name in observed]
    predictions = observed[prediction_columns].copy()
    rows, audit = [], {}
    for method, model, result in zip(METHODS, models, results):
        prediction = result["CDR_MLC_meta_stacker"]
        score = metrics(observed.traffic_label.to_numpy(), prediction, APPLICATIONS)
        meta_dimension = int(model["meta_models"][model["selected_meta_variant"]].n_features_in_)
        rows.append({"method": method, "n": score["n"],
                     "raw_expert_features": len(model["numeric"] + model["categorical"]),
                     "encoded_expert_dimension": int(model["experts"][0].n_features_in_),
                     "meta_input_dimension": meta_dimension,
                     **{key: score[key] for key in SCORES}})
        predictions[f"prediction_{method}"] = prediction
        audit[method] = {
            "use_clustering": model["use_clustering"],
            "router_algorithm": model["router_algorithm"],
            "router_audit": model["router_audit"],
            "mbk_label_aware": model["mbk_label_aware"],
            "initial_class_coverage_audit": model["class_coverage_audit"],
            "refit_class_coverage_audit": model.get("refit_class_coverage_audit"),
            "initial_expert_partition_counts": model["cluster_counts"],
            "expert_partition_counts": model.get("full_source_cluster_counts", model["cluster_counts"]),
            "expert_input_columns": model["numeric"] + model["categorical"],
            "congestion_columns": model["congestion_columns"],
            "meta_input_dimension": meta_dimension,
            "partition_rows": model["partition_rows"],
            "random_partition": model.get("random_partition"),
            "selected_meta_variant": model["selected_meta_variant"],
            "selected_meta_confidence": model["selected_meta_confidence"],
            "selection_trials": model["meta_selection_trials"],
            "leakage_control": model["leakage_control"],
        }
    audit.update({"development_identity": frame_identity(development),
                  "test_identity": frame_identity(test),
                  "evaluated_identity": frame_identity(observed),
                  "same_scored_rows_and_context_values_verified": True,
                  "no_MBK_scaler_distances_or_route_features_verified": True})
    return rows, predictions, audit


def summarize(rows):
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(["protocol", "method"], sort=False)[list(SCORES)].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    deltas = []
    for (protocol, seed), group in frame.groupby(["protocol", "seed"], sort=False):
        indexed = group.set_index("method")
        deltas.append({"protocol": protocol, "seed": seed,
                       **{f"{key}_delta_no_clustering_minus_full": float(
                           indexed.loc[METHODS[1], key] - indexed.loc[METHODS[0], key]
                       ) for key in SCORES}})
    delta = pd.DataFrame(deltas)
    means = frame.groupby(["seed", "method"], sort=False)[list(SCORES)].mean().reset_index()
    means["protocols"] = frame.groupby(["seed", "method"], sort=False).size().to_numpy()
    mean_delta = delta.groupby("seed", sort=False).mean(numeric_only=True).reset_index()
    return frame, aggregate.reset_index(), delta, means, mean_delta


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/clustering_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--scenarios", nargs="*", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--protocols", nargs="*", choices=list(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--dynamic-scenarios", nargs="*", choices=list(DYNAMIC_SCENARIOS), default=[])
    parser.add_argument("--block-rows", type=int, default=1000,
                        help="Records per dynamic-stream segment")
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--target-test-fraction", type=float, default=1.0)
    parser.add_argument("--mbk-n-init", type=int, default=10,
                        help="MBK initialization candidates; ignored by No_Clustering")
    parser.add_argument("--mbk-batch-size", type=int, default=1024,
                        help="MBK minibatch size; ignored by No_Clustering")
    parser.add_argument("--mbk-max-iter", type=int, default=100,
                        help="MBK maximum iterations; ignored by No_Clustering")
    parser.add_argument("--mbk-label-aware", action="store_true",
                        help="Constrain training expert partitions to contain every observed application class")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--expert-refit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    if not args.scenarios and not args.protocols and not args.dynamic_scenarios:
        parser.error("Select at least one scenario, protocol or dynamic scenario")
    if args.dynamic_scenarios and (args.block_rows < len(APPLICATIONS) or args.block_rows % len(APPLICATIONS)):
        parser.error("--block-rows must be a positive multiple of application count")
    if not 0 < args.train_fraction < 1 or not 0 < args.target_test_fraction <= 1:
        parser.error("Invalid train/test fractions")
    if min(args.mbk_n_init, args.mbk_batch_size, args.mbk_max_iter) < 1:
        parser.error("--mbk-n-init, --mbk-batch-size and --mbk-max-iter must be positive")
    if min(args.expert_trees, args.utility_trees, args.meta_trees) < 1:
        parser.error("Tree counts must be positive")
    config = MetaStackConfig(window=args.window, congestion_window=args.congestion_window,
        expert_trees=args.expert_trees, utility_trees=args.utility_trees,
        meta_trees=args.meta_trees, refit_experts=args.expert_refit,
        mbk_n_init=args.mbk_n_init, mbk_batch_size=args.mbk_batch_size,
        mbk_max_iter=args.mbk_max_iter,
        mbk_label_aware=args.mbk_label_aware).validate()
    data, input_audit = load_dataset(args.data_dir, tuple(dict.fromkeys(
        [*TIMING, *DEFAULT_CONGESTION_FEATURES])))
    evaluations = build_evaluations(data, args.scenarios, args.protocols,
                                    args.train_fraction, args.target_test_fraction)
    if args.dynamic_scenarios:
        development, tails = ordered_development_tail(data, args.train_fraction)
        evaluations.extend((development, tails, {
            "name": name, "dynamic": True, "drift_type": DYNAMIC_SCENARIOS[name]["name"],
            "block_rows": args.block_rows,
        }) for name in args.dynamic_scenarios)
    manifest = json.loads(json.dumps({
        "config": asdict(config), "seeds": args.seeds,
        "scenarios": args.scenarios, "protocols": args.protocols,
        "dynamic_scenarios": args.dynamic_scenarios, "block_rows": args.block_rows,
        "dynamic_definitions": {name: DYNAMIC_SCENARIOS[name] for name in args.dynamic_scenarios},
        "train_fraction": args.train_fraction,
        "target_test_fraction": args.target_test_fraction,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(args.data_dir.glob("*.flow"))},
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.glob("*.py"))},
        "intervention": "replace MBK expert partitions with independent uniform RNG partitions; remove all geometry features",
        "retained": "three experts, context, utility, meta-fusion, fallback, partitions, seed and tree budgets",
        "eligibility": "same complete TTFEF/context windows in both variants",
        "selection_policy": "same development-only search, independently selected per variant",
        "delta_direction": "no clustering minus full; negative favors clustering",
    }))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            parser.error("Resume requires the same inputs, code, and configuration")
    elif manifest_path.exists():
        parser.error("Output already exists; use --resume or a new --output folder")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    all_rows = []
    for development, test, definition in evaluations:
        name = definition["name"]
        for seed in args.seeds:
            run_test, run_definition = test, definition
            if definition.get("dynamic"):
                run_test, segments = build_stream(test, name, args.block_rows, seed)
                run_definition = {**definition, "segments": segments,
                                  "raw_stream_rows": len(run_test)}
            destination = args.output / name / f"seed_{seed}"
            checkpoint = destination / "pair.json"
            if args.resume and checkpoint.exists():
                rows = json.loads(checkpoint.read_text())["rows"]
                print(f"Resume: {name}, seed {seed}", flush=True)
            else:
                print(f"Running: {name}, seed {seed} (full and no clustering)", flush=True)
                rows, predictions, audit = evaluate_pair(development, run_test, replace(config, random_state=seed))
                rows = [{"protocol": name, "seed": seed, **r} for r in rows]
                destination.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(destination / "predictions.csv", index=False)
                temporary = destination / "pair.tmp"
                temporary.write_text(json.dumps({"rows": rows, "definition": run_definition, "audit": audit}, indent=2) + "\n", encoding="utf-8")
                temporary.replace(checkpoint)
            all_rows.extend(rows)
            frame, aggregate, delta, means, mean_delta = summarize(all_rows)
            for filename, output in (("per_seed_metrics.csv", frame), ("summary.csv", aggregate),
                ("paired_deltas.csv", delta), ("protocol_means.csv", means), ("mean_paired_deltas.csv", mean_delta)):
                output.to_csv(args.output / filename, index=False)
            print(frame.tail(2).to_string(index=False), flush=True)
    print("\nUnweighted means across completed protocols:")
    print(means.to_string(index=False))
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
