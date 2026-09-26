"""Fast, leakage-safe staged hyperparameter search for MF-CDR-MLC.

Stage 1 searches window pairs with a small fixed tree budget. Stage 2 evaluates
only the best window pairs with a short list of tree budgets. Selection uses an
internal development/validation split. The fixed outer test is evaluated once,
and only when --final-evaluate is supplied.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from compare_clean_valid import metrics
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    fit_meta_stacker,
    predict_all,
)
from sdncampus_rf_cdr_mf_comparison import (
    PAPER_TIMING,
    _set_application_ontology,
    infer_dataset_name,
    load_sdncampus,
    split_80_20,
)


def parse_tree_budget(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tree budget must be expert,utility,meta"
        ) from error
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError(
            "tree budget must contain three positive integers"
        )
    return parts


def common_scoring_index(frame: pd.DataFrame, context: int) -> pd.Index:
    selected = []
    for _, group in frame.groupby("sequence_id", sort=False):
        ordered = group.sort_values(["timestamp", "source_row"], kind="stable")
        if len(ordered) >= context:
            selected.extend(ordered.index[context - 1:].tolist())
    if not selected:
        raise ValueError("validation has no common context-eligible rows")
    return pd.Index(selected)


def score_trial(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    scoring_index: pd.Index,
    labels: list[str],
    window: int,
    congestion_window: int,
    trees: tuple[int, int, int],
    seed: int,
) -> dict:
    expert_trees, utility_trees, meta_trees = trees
    config = MetaStackConfig(
        window=window,
        congestion_window=congestion_window,
        congestion_features=PAPER_TIMING,
        expert_trees=expert_trees,
        utility_trees=utility_trees,
        meta_trees=meta_trees,
        random_state=seed,
    ).validate()

    started = perf_counter()
    model = fit_meta_stacker(train, config)
    result = predict_all(model, validation)
    seconds = perf_counter() - started

    observed = result["observed"]
    prediction = np.asarray(result["CDR_MLC_meta_stacker"])
    keep = observed.index.isin(scoring_index)
    if int(keep.sum()) != len(scoring_index):
        missing = len(scoring_index) - int(keep.sum())
        raise RuntimeError(
            f"trial omitted {missing} common validation rows"
        )
    truth = observed.loc[keep, "traffic_label"].to_numpy()
    scored_prediction = prediction[keep]
    values = metrics(truth, scored_prediction, labels)
    return {
        "window": window,
        "congestion_window": congestion_window,
        "expert_trees": expert_trees,
        "utility_trees": utility_trees,
        "meta_trees": meta_trees,
        "validation_n": values["n"],
        "accuracy": values["accuracy"],
        "balanced_accuracy": values["balanced_accuracy"],
        "macro_f1": values["macro_f1"],
        "weighted_f1": values["weighted_f1"],
        "fit_predict_seconds": seconds,
    }


def ranking_key(row: dict):
    return (
        row["macro_f1"],
        row["balanced_accuracy"],
        row["accuracy"],
        -row["fit_predict_seconds"],
    )


def compact(frame: pd.DataFrame, limit: int = 20) -> None:
    columns = [
        "stage", "window", "congestion_window",
        "expert_trees", "utility_trees", "meta_trees",
        "validation_n", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "fit_predict_seconds",
    ]
    shown = frame.loc[:, columns].head(limit).copy()
    shown.columns = [
        "Stg", "W", "CW", "ET", "UT", "MT",
        "N", "Acc", "BAcc", "MF1", "WF1", "Sec",
    ]
    for name in ("Acc", "BAcc", "MF1", "WF1"):
        shown[name] = shown[name].map(lambda value: f"{value:.4f}")
    shown["Sec"] = shown["Sec"].map(lambda value: f"{value:.2f}")
    print(shown.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--validation-fraction", type=float, default=.20)
    parser.add_argument(
        "--split-mode", choices=("auto", "stratified", "ordered"),
        default="auto",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1042)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, nargs="+", default=[3, 5, 10, 15])
    parser.add_argument(
        "--congestion-windows", type=int, nargs="+",
        default=[10, 15, 19, 20, 25, 32],
    )
    parser.add_argument(
        "--stage1-trees", type=parse_tree_budget, default=(20, 10, 20),
        metavar="E,U,M",
    )
    parser.add_argument(
        "--tree-budgets", type=parse_tree_budget, nargs="+",
        default=[
            (20, 10, 20), (50, 40, 50),
            (80, 60, 80), (100, 80, 100),
        ],
        metavar="E,U,M",
    )
    parser.add_argument("--top-window-pairs", type=int, default=3)
    parser.add_argument("--final-evaluate", action="store_true")
    args = parser.parse_args()

    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0,1)")
    if min(args.windows + args.congestion_windows) < 2:
        parser.error("all windows must be at least 2")
    if args.top_window_pairs < 1:
        parser.error("--top-window-pairs must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_sdncampus(args.data)
    labels = sorted(data.traffic_label.astype(str).unique().tolist())
    _set_application_ontology(labels)

    development, fixed_test, outer_audit = split_80_20(
        data, args.train_fraction, args.split_mode, args.split_seed
    )
    validation_train_fraction = 1.0 - args.validation_fraction
    search_train, search_validation, validation_audit = split_80_20(
        development,
        validation_train_fraction,
        outer_audit[0]["split_mode"],
        args.validation_seed,
    )

    max_context = max(max(args.windows), max(args.congestion_windows))
    scoring_index = common_scoring_index(search_validation, max_context)
    cache: dict[tuple[int, int, int, int, int], dict] = {}
    rows: list[dict] = []

    def evaluate(stage, window, congestion_window, trees):
        key = (window, congestion_window, *trees)
        if key not in cache:
            cache[key] = score_trial(
                search_train, search_validation, scoring_index, labels,
                window, congestion_window, trees, args.seed,
            )
        row = {"stage": stage, **cache[key]}
        rows.append(row)
        return row

    stage1 = []
    for window in sorted(set(args.windows)):
        for congestion_window in sorted(set(args.congestion_windows)):
            stage1.append(evaluate(
                "window", window, congestion_window, args.stage1_trees
            ))
    stage1.sort(key=ranking_key, reverse=True)
    selected_pairs = []
    for row in stage1:
        pair = (row["window"], row["congestion_window"])
        if pair not in selected_pairs:
            selected_pairs.append(pair)
        if len(selected_pairs) >= args.top_window_pairs:
            break

    stage2 = []
    for window, congestion_window in selected_pairs:
        for trees in args.tree_budgets:
            stage2.append(evaluate(
                "trees", window, congestion_window, trees
            ))
    candidates = stage2 if stage2 else stage1
    best = max(candidates, key=ranking_key)

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    results.to_csv(args.output / "search_results.csv", index=False)

    summary = {
        "dataset": infer_dataset_name(args.data),
        "data": str(args.data),
        "selection_metric": "validation macro_f1; balanced_accuracy tie-break",
        "test_used_during_search": False,
        "outer_split_mode": outer_audit[0]["split_mode"],
        "outer_split_seed": args.split_seed,
        "validation_split_seed": args.validation_seed,
        "common_validation_rows": len(scoring_index),
        "stage1_trials": len(stage1),
        "stage2_trials": len(stage2),
        "selected_window_pairs": selected_pairs,
        "best": best,
        "input_audit": input_audit,
        "outer_split_audit": outer_audit,
        "validation_split_audit": validation_audit,
    }

    if args.final_evaluate:
        best_config = MetaStackConfig(
            window=int(best["window"]),
            congestion_window=int(best["congestion_window"]),
            congestion_features=PAPER_TIMING,
            expert_trees=int(best["expert_trees"]),
            utility_trees=int(best["utility_trees"]),
            meta_trees=int(best["meta_trees"]),
            random_state=args.seed,
        ).validate()
        started = perf_counter()
        final_model = fit_meta_stacker(development, best_config)
        final_result = predict_all(final_model, fixed_test)
        final_seconds = perf_counter() - started
        observed = final_result["observed"]
        final_values = metrics(
            observed.traffic_label.to_numpy(),
            final_result["CDR_MLC_meta_stacker"],
            labels,
        )
        final_row = {
            "protocol": (
                f"{infer_dataset_name(args.data)}-"
                f"{outer_audit[0]['split_mode'].title()}-80-20"
            ),
            "method": "MF_CDR_MLC",
            "seed": args.seed,
            **{
                key: final_values[key] for key in (
                    "n", "accuracy", "balanced_accuracy",
                    "macro_f1", "weighted_f1",
                )
            },
            "fit_predict_seconds": final_seconds,
        }
        pd.DataFrame([final_row]).to_csv(
            args.output / "final_test_results.csv", index=False
        )
        summary["final_test"] = final_row

    (args.output / "best_config.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    compact(results)
    print("\nBest validation configuration")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
