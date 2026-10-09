"""Paired MF-CDR-MLC ablation: 32 versus 35 raw expert features.

Both variants retain the complete congestion-context path and use the same
production training/selection procedure. No hyperparameter is chosen on test.
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


METHODS = ("MF_32_Features", "MF_35_Features")
SCORES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def check_pair(models, development, test):
    """Fail rather than report a comparison with changed geometry or context."""
    left, right = models
    left_columns = left["numeric"] + left["categorical"]
    right_columns = right["numeric"] + right["categorical"]
    if len(left_columns) != 32 or len(right_columns) != 35:
        raise ValueError(
            f"Expected 32/35 raw expert columns, found {len(left_columns)}/"
            f"{len(right_columns)}. Check the Clean_Valid schema and constant fields."
        )
    if set(right_columns) - set(left_columns) != set(TIMING):
        raise RuntimeError("Expert inputs differ by fields other than the three timing fields")
    if set(left_columns) - set(right_columns):
        raise RuntimeError("The 32-feature inputs are not a subset of the 35-feature inputs")
    if left["partition_rows"] != right["partition_rows"]:
        raise RuntimeError("Training partition sizes differ")
    if left["source_eligible_index"] != right["source_eligible_index"]:
        raise RuntimeError("Expert training rows differ")
    for attribute in ("mean_", "scale_", "var_"):
        np.testing.assert_array_equal(
            getattr(left["scaler"], attribute), getattr(right["scaler"], attribute)
        )
    np.testing.assert_array_equal(
        left["router"].cluster_centers_, right["router"].cluster_centers_
    )
    if left["congestion_config"] != right["congestion_config"]:
        raise RuntimeError("Congestion-context configurations differ")
    # Check later training partitions as well as test; expert probabilities may
    # change, but routing geometry and appended congestion values must not.
    split = four_way_split(development, left["meta_stack_config"])
    for name, frame in [(k, split[k]) for k in ("utility", "meta", "selection")] + [("test", test)]:
        a = _enhanced_outputs(left, frame, left["congestion_config"])
        b = _enhanced_outputs(right, frame, right["congestion_config"])
        if frame_identity(a[0]) != frame_identity(b[0]) or a[-1] != b[-1]:
            raise RuntimeError(f"{name}: eligible rows or context columns differ")
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        count = len(a[-1])
        if not count:
            raise RuntimeError("Congestion context must remain enabled")
        np.testing.assert_array_equal(a[-2][:, -count:], b[-2][:, -count:])


def evaluate_pair(development, test, config):
    ids = lambda f: set(zip(f.source_file, f.source_row))
    if ids(development) & ids(test):
        raise RuntimeError("Development and test records overlap")
    models = [fit_meta_stacker(
        development, replace(config, include_timing_in_experts=include)
    ) for include in (False, True)]
    check_pair(models, development, test)
    results = [predict_all(model, test) for model in models]
    observed = results[0]["observed"]
    if frame_identity(observed) != frame_identity(results[1]["observed"]):
        raise RuntimeError("Evaluated test rows differ")
    predictions = observed[["timestamp", "source_file", "source_row",
                            "traffic_label", "congestion_level"]].copy()
    rows, audit = [], {}
    for method, model, result in zip(METHODS, models, results):
        prediction = result["CDR_MLC_meta_stacker"]
        score = metrics(observed.traffic_label.to_numpy(), prediction, APPLICATIONS)
        rows.append({"method": method, "n": score["n"],
                     **{key: score[key] for key in SCORES}})
        predictions[f"prediction_{method}"] = prediction
        audit[method] = {
            "expert_input_columns": model["numeric"] + model["categorical"],
            "encoded_expert_dimension": int(model["experts"][0].n_features_in_),
            "partition_rows": model["partition_rows"],
            "selected_meta_variant": model["selected_meta_variant"],
            "selected_meta_confidence": model["selected_meta_confidence"],
            "selection_trials": model["meta_selection_trials"],
            "leakage_control": model["leakage_control"],
        }
    audit.update({"development_identity": frame_identity(development),
                  "test_identity": frame_identity(test),
                  "evaluated_identity": frame_identity(observed),
                  "same_router_scaler_and_congestion_values_verified": True})
    return rows, predictions, audit


def summarize(rows):
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby(["protocol", "method"], sort=False)[list(SCORES)].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    delta = []
    for (protocol, seed), group in frame.groupby(["protocol", "seed"], sort=False):
        indexed = group.set_index("method")
        delta.append({"protocol": protocol, "seed": seed,
                      **{f"{key}_delta_35_minus_32": float(
                          indexed.loc[METHODS[1], key] - indexed.loc[METHODS[0], key]
                      ) for key in SCORES}})
    return frame, aggregate.reset_index(), pd.DataFrame(delta)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/expert_input_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--scenarios", nargs="*", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--protocols", nargs="*", choices=list(PROTOCOLS), default=list(PROTOCOLS))
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--target-test-fraction", type=float, default=1.0)
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
    if not args.scenarios and not args.protocols:
        parser.error("Select at least one scenario or protocol")
    if not 0 < args.train_fraction < 1 or not 0 < args.target_test_fraction <= 1:
        parser.error("Invalid train/test fractions")
    if min(args.expert_trees, args.utility_trees, args.meta_trees) < 1:
        parser.error("Tree counts must be positive")
    config = MetaStackConfig(
        window=args.window, congestion_window=args.congestion_window,
        expert_trees=args.expert_trees, utility_trees=args.utility_trees,
        meta_trees=args.meta_trees, refit_experts=args.expert_refit,
    ).validate()
    data, input_audit = load_dataset(args.data_dir, tuple(dict.fromkeys(
        [*TIMING, *DEFAULT_CONGESTION_FEATURES]
    )))
    evaluations = build_evaluations(data, args.scenarios, args.protocols,
                                    args.train_fraction, args.target_test_fraction)
    # Hash actual capture contents, not just row identities, for safe resumption.
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.data_dir.glob("*.flow"))}
    manifest = {
        "config": asdict(config), "seeds": args.seeds,
        "scenarios": args.scenarios, "protocols": args.protocols,
        "train_fraction": args.train_fraction,
        "target_test_fraction": args.target_test_fraction,
        "input_sha256": hashes,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(root.glob("*.py"))},
        "only_intervention": "include AckDat, TcpRtt, SynAck in RF expert inputs",
        "selection_policy": "same development-only candidate search for both variants",
        "delta_direction": "35 features minus 32 features; positive favors 35",
    }
    # Normalize tuple fields to their on-disk JSON representation.
    manifest = json.loads(json.dumps(manifest))
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
            destination = args.output / name / f"seed_{seed}"
            checkpoint = destination / "pair.json"
            if args.resume and checkpoint.exists():
                rows = json.loads(checkpoint.read_text())["rows"]
                print(f"Resume: {name}, seed {seed}", flush=True)
            else:
                print(f"Running: {name}, seed {seed} (32 and 35 features)", flush=True)
                rows, predictions, audit = evaluate_pair(
                    development, test, replace(config, random_state=seed)
                )
                rows = [{"protocol": name, "seed": seed, **r} for r in rows]
                destination.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(destination / "predictions.csv", index=False)
                # The checkpoint is written last; interrupted pairs are rerun.
                payload = {"rows": rows, "definition": definition, "audit": audit}
                temporary = destination / "pair.tmp"
                temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temporary.replace(checkpoint)
            all_rows.extend(rows)
            frame, aggregate, delta = summarize(all_rows)
            frame.to_csv(args.output / "per_seed_metrics.csv", index=False)
            aggregate.to_csv(args.output / "summary.csv", index=False)
            delta.to_csv(args.output / "paired_deltas.csv", index=False)
            print(frame.tail(2).to_string(index=False), flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
