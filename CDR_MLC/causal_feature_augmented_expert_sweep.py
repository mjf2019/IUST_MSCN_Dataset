"""Lightweight causal expert-augmentation comparison.

Fits only two 110-tree CDR-MLC architectures per run:
1) causal SA-Meta baseline, and
2) the same architecture with causal expert feature augmentation.
FA-Oracle is diagnostic output from model 2 and does not fit another model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, load_dataset
from compare_clean_valid import TIMING, metrics
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from ood_residual_trend_meta_stacker import (
    OODResidualConfig,
    fit_ood_residual_meta_stacker,
    predict_all,
)


SCENARIOS = {
    "1": (("Low",), "Medium"),
    "2": (("Low",), "High"),
    "3": (("Medium",), "High"),
    "LM-H": (("Low", "Medium"), "High"),
    "LH-M": (("Low", "High"), "Medium"),
    "MH-L": (("Medium", "High"), "Low"),
}


def fixed_test_and_adaptation(target, adaptation_fraction, test_fraction):
    adaptation, test, audit = [], [], []
    for sequence_id, group in target.groupby("sequence_id", sort=False):
        group = group.sort_values(
            ["timestamp", "source_row"], kind="stable"
        )
        count = len(group)
        adaptation_end = int(count * adaptation_fraction)
        test_start = int(count * (1.0 - test_fraction))
        if not 0 < test_start < count:
            raise ValueError(
                f"{sequence_id}: invalid fixed test split"
            )
        if adaptation_end > test_start:
            raise ValueError(
                f"{sequence_id}: adaptation overlaps test"
            )
        if adaptation_end:
            adaptation.append(group.iloc[:adaptation_end].copy())
        test.append(group.iloc[test_start:].copy())
        audit.append({
            "sequence_id": str(sequence_id),
            "adaptation_rows": adaptation_end,
            "unused_rows": test_start - adaptation_end,
            "test_rows": count - test_start,
        })
    empty = target.iloc[:0].copy()
    return (
        pd.concat(adaptation, ignore_index=True)
        if adaptation else empty,
        pd.concat(test, ignore_index=True),
        audit,
    )


def frame_identity(frame):
    ordered = frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    )[["sequence_id", "source_file", "source_row"]]
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def route_audit(result):
    routes = result["routes"].copy()
    routes["sequence_id"] = (
        result["observed"].loc[routes.index, "sequence_id"].astype(str)
    )
    rows = []
    for sequence_id, group in routes.groupby(
        "sequence_id", sort=False
    ):
        selected = group[
            "shift_adaptive_selected_policy"
        ].value_counts()
        rows.append({
            "sequence_id": str(sequence_id),
            "rows": int(len(group)),
            "causal_score_min": float(
                group["causal_shift_score"].min()
            ),
            "causal_score_median": float(
                group["causal_shift_score"].median()
            ),
            "causal_score_max": float(
                group["causal_shift_score"].max()
            ),
            "cb_rows": int(selected.get("CB-Meta", 0)),
            "dg_rows": int(selected.get("DG-Meta", 0)),
        })
    return rows


def series(result, key):
    return pd.Series(
        result[key], index=result["observed"].index
    )


def make_config(args, augmented):
    return OODResidualConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        router_candidates=tuple(args.congestion_features),
        context_features=tuple(args.congestion_features),
        selected_feature_count=3,
        trend_window=3,
        expert_trees=args.expert_trees,
        expert_feature_count=None,
        expert_causal_augmentation=augmented,
        expert_augmentation_window=args.expert_augmentation_window,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        shift_history_window=args.shift_window,
        random_state=args.seed,
    ).validate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/causal_feature_augmented_experts",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS),
        default=["1", "2"],
    )
    parser.add_argument(
        "--adaptation-fractions", nargs="+", type=float,
        default=[0.0],
    )
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--shift-window", type=int, default=10)
    parser.add_argument(
        "--expert-augmentation-window", type=int, default=10
    )
    parser.add_argument(
        "--congestion-features", nargs="+",
        default=list(DEFAULT_CONGESTION_FEATURES),
    )
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fractions = tuple(dict.fromkeys(args.adaptation_fractions))
    if any(
        value < 0 or value + args.test_fraction > 1
        for value in fractions
    ):
        raise ValueError("invalid adaptation/test fractions")

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([
        *TIMING, *args.congestion_features,
    ]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    baseline_config = make_config(args, augmented=False)
    augmented_config = make_config(args, augmented=True)
    summary_rows = []
    run_audits = {}

    for scenario in args.scenarios:
        source_levels, target_level = SCENARIOS[scenario]
        source = data[
            data.congestion_level.isin(source_levels)
        ].copy()
        raw_target = data[
            data.congestion_level.eq(target_level)
        ].copy()
        if source.empty or raw_target.empty:
            raise ValueError(f"{scenario}: empty source or target")

        test_identity = None
        for fraction in fractions:
            adaptation, target, split_audit = (
                fixed_test_and_adaptation(
                    raw_target, fraction, args.test_fraction
                )
            )
            identity = frame_identity(target)
            if test_identity is None:
                test_identity = identity
            elif identity != test_identity:
                raise RuntimeError("fixed test changed across fractions")
            development = pd.concat(
                [source, adaptation], ignore_index=True
            )

            baseline_model = fit_ood_residual_meta_stacker(
                development, baseline_config
            )
            augmented_model = fit_ood_residual_meta_stacker(
                development, augmented_config
            )
            baseline = predict_all(baseline_model, target)
            augmented = predict_all(augmented_model, target)

            prediction_series = {
                "SA-Meta": series(
                    baseline,
                    "CDR_MLC_shift_adaptive_meta_stacker",
                ),
                "FA-SA-Meta": series(
                    augmented,
                    "CDR_MLC_shift_adaptive_meta_stacker",
                ),
                "FA-Oracle": series(
                    augmented, "CDR_MLC_oracle_router"
                ),
            }
            common = sorted(set.intersection(*[
                set(value.index)
                for value in prediction_series.values()
            ]))
            observed = target.loc[common]
            truth = observed.traffic_label.to_numpy()

            run_key = f"{scenario}_af_{int(round(100*fraction)):02d}"
            detailed = {}
            for method, prediction in prediction_series.items():
                score = metrics(
                    truth,
                    prediction.loc[common].to_numpy(),
                    APPLICATIONS,
                )
                detailed[method] = score
                summary_rows.append({
                    "adaptation_fraction": fraction,
                    "scenario": scenario,
                    "source": "+".join(source_levels),
                    "target": target_level,
                    "n": score["n"],
                    "method": method,
                    "accuracy": score["accuracy"],
                    "balanced_accuracy": score["balanced_accuracy"],
                    "macro_f1": score["macro_f1"],
                    "weighted_f1": score["weighted_f1"],
                })

            run_dir = args.output / f"scenario_{run_key.lower()}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.json").write_text(
                json.dumps(detailed, indent=2) + "\n",
                encoding="utf-8",
            )
            run_audits[run_key] = {
                "source_levels": list(source_levels),
                "target_level": target_level,
                "adaptation_fraction": fraction,
                "development_rows": len(development),
                "test_rows": len(observed),
                "fixed_test_identity": identity,
                "target_split": split_audit,
                "baseline_routes": route_audit(baseline),
                "augmented_routes": route_audit(augmented),
                "augmented_feature_count_per_expert": {
                    str(cluster): len(columns)
                    for cluster, columns in (
                        augmented_model["expert_numeric"].items()
                    )
                },
                "augmented_features_per_expert": {
                    str(cluster): list(columns)
                    for cluster, columns in (
                        augmented_model["expert_numeric"].items()
                    )
                },
                "leakage_control": augmented_model[
                    "leakage_control"
                ],
            }

    summary = pd.DataFrame(summary_rows).sort_values([
        "scenario", "adaptation_fraction", "method"
    ])
    summary.to_csv(
        args.output / "causal_feature_augmented_summary.csv",
        index=False,
    )
    manifest = {
        "methods": ["SA-Meta", "FA-SA-Meta", "FA-Oracle"],
        "fitted_architectures_per_run": 2,
        "trees_per_architecture": (
            3 * args.expert_trees
            + 3 * args.utility_trees
            + args.meta_trees
        ),
        "baseline_config": asdict(baseline_config),
        "augmented_config": asdict(augmented_config),
        "augmentation": {
            "raw_features_preserved": True,
            "causal_per_sequence": True,
            "future_rows_used": False,
            "derived_per_numeric_feature": [
                "delta1", "relative_to_trailing_median",
                "trailing_robust_z"
            ],
        },
        "target_test_used_for_training_or_selection": False,
        "oracle_is_diagnostic_only": True,
        "runs": run_audits,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    display = summary[[
        "adaptation_fraction", "scenario", "source", "target",
        "n", "method", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1",
    ]].copy()
    display.columns = [
        "AF", "Scn", "Src", "Tgt", "N", "Method",
        "Acc", "BAcc", "MF1", "WF1",
    ]
    rendered = display.to_string(
        index=False,
        justify="left",
        formatters={
            "AF": lambda value: f"{value:.2f}",
            "Acc": lambda value: f"{value:.4f}",
            "BAcc": lambda value: f"{value:.4f}",
            "MF1": lambda value: f"{value:.4f}",
            "WF1": lambda value: f"{value:.4f}",
        },
    )
    lrm = "\u200e"
    print("\n".join(lrm + line for line in rendered.splitlines()))


if __name__ == "__main__":
    main()
