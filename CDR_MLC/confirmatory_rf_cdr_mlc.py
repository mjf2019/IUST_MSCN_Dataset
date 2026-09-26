"""Multi-seed confirmatory comparison of RF, Original CDR-MLC, and MF-CDR-MLC.

The experiment covers three complete-source-to-complete-target congestion
transfers and four mixed-level protocols.  All methods receive identical
development/test partitions within a protocol and seed.  The test partitions
are fixed across seeds.  Original CDR-MLC uses the same frozen early router and
full-development expert refit used by the actual-router branch of MF-CDR-MLC;
the script verifies prediction equivalence on every run.

Outputs
-------
confirmatory_runs.csv
    One row per protocol, seed, and method.
confirmatory_summary.csv
    Mean, sample standard deviation, and 95% t confidence intervals.
confirmatory_pairwise_tests.csv
    Paired Wilcoxon tests, Holm-adjusted p-values, and rank-biserial effects.
confirmatory_runtime.csv
    Training/inference wall time and inference throughput for every run.
confirmatory_manifest.json
    Configuration, partition identities, composition, and leakage controls.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t, wilcoxon

from adaptive_cdr_mlc import APPLICATIONS, load_dataset, trend_frame
from compare_clean_valid import (
    SCENARIOS,
    TIMING,
    fit_fixed_cdr,
    fit_rf,
    metrics,
    predict_fixed_cdr,
    predict_rf,
)
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from learned_router_cdr_mlc import (
    LearnedRouterConfig,
    _refit_experts_with_frozen_router,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    fit_meta_stacker,
    four_way_split,
    predict_all,
)
from mixed_level_protocols_leakage_safe import (
    PROTOCOLS,
    build_protocol,
    composition,
    frame_identity,
)
from benchmarks.console_output import (
    ResourceMonitor, print_compact_results, resource_values,
)


METHODS = ("RF_Clean_Valid", "Original_CDR_MLC", "MF_CDR_MLC")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
SUMMARY_VALUES = (
    *METRICS,
    "fit_seconds",
    "predict_seconds",
    "throughput_input_rows_per_second",
    "inference_microseconds_per_input_row",
    "peak_ram_mb",
    "peak_gpu_mb",
    "ttfef_us_per_row",
)
PAIRWISE = (
    ("MF_CDR_MLC", "RF_Clean_Valid"),
    ("MF_CDR_MLC", "Original_CDR_MLC"),
    ("Original_CDR_MLC", "RF_Clean_Valid"),
)


def _timed(function, *args, device="cpu", **kwargs):
    gc.collect()
    with ResourceMonitor(device) as monitor:
        start = perf_counter()
        value = function(*args, **kwargs)
        seconds = perf_counter() - start
    return value, seconds, monitor


def _record_ids(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame.source_file.astype(str), frame.source_row.astype(int)))


def _full_target_protocol(
    data: pd.DataFrame, scenario: str, target_test_fraction: float
):
    source, target = SCENARIOS[scenario]
    development = (
        data[data.congestion_level.eq(source)].copy().reset_index(drop=True)
    )
    target_all = data[data.congestion_level.eq(target)].copy().reset_index(drop=True)
    if not 0 < target_test_fraction <= 1:
        raise ValueError("target_test_fraction must be in (0,1]")
    if np.isclose(target_test_fraction, 1.0):
        test = target_all
    else:
        parts = []
        for sequence_id, group in target_all.groupby("sequence_id", sort=False):
            group = group.sort_values(["timestamp", "source_row"], kind="stable")
            start = int(len(group) * (1.0 - target_test_fraction))
            if not 0 < start < len(group):
                raise ValueError(f"{sequence_id}: insufficient ordered target tail")
            parts.append(group.iloc[start:].copy())
        test = pd.concat(parts, ignore_index=True)
    if development.empty or test.empty:
        raise ValueError(f"scenario {scenario}: empty development or test set")
    overlap = _record_ids(development) & _record_ids(test)
    if overlap:
        raise RuntimeError(
            f"scenario {scenario}: {len(overlap)} development/test overlaps"
        )
    return development, test, {
        "name": f"Scenario-{scenario}-{source}-to-{target}",
        "evaluation_type": "full_target_scenario",
        "scenario": scenario,
        "source": source,
        "target": target,
        "kind": "complete_source_level_to_ordered_target_tail",
        "target_level_used_in_training_or_selection": False,
        "target_rows_before_tail": len(target_all),
        "target_test_fraction": target_test_fraction,
    }


def _mixed_protocol(data: pd.DataFrame, name: str, train_fraction: float):
    development, test, definition = build_protocol(data, name, train_fraction)
    specification = PROTOCOLS[name]
    if name == "ALL-80-20":
        source = target = "Low+Medium+High"
    else:
        source = "+".join(specification["train_levels"])
        target = specification["test_level"]
    return development, test, {
        "name": name,
        "evaluation_type": "mixed_level",
        "scenario": "",
        "source": source,
        "target": target,
        **definition,
    }


def build_evaluations(
    data: pd.DataFrame, scenarios, protocols, train_fraction,
    target_test_fraction,
):
    evaluations = []
    for scenario in scenarios:
        evaluations.append(
            _full_target_protocol(data, scenario, target_test_fraction)
        )
    for protocol in protocols:
        evaluations.append(_mixed_protocol(data, protocol, train_fraction))
    names = [definition["name"] for _, _, definition in evaluations]
    if len(names) != len(set(names)):
        raise RuntimeError("evaluation names are not unique")
    return evaluations


def fit_original_cdr(development: pd.DataFrame, config: MetaStackConfig):
    """Fit the actual-router baseline embedded in the full MF architecture."""
    split = four_way_split(development, config)
    initial = fit_fixed_cdr(
        split["expert"],
        config.window,
        config.random_state,
        config.expert_trees,
    )
    helper = LearnedRouterConfig(
        window=config.window,
        expert_trees=config.expert_trees,
        random_state=config.random_state,
    )
    return _refit_experts_with_frozen_router(initial, development, helper)


def _assert_common_rows(observed: pd.DataFrame, original_prediction: pd.Series):
    missing = observed.index.difference(original_prediction.index)
    if len(missing):
        raise RuntimeError(
            f"Original CDR-MLC is missing {len(missing)} MF evaluation rows"
        )


def evaluate_once(
    development: pd.DataFrame,
    test: pd.DataFrame,
    definition: dict,
    config: MetaStackConfig,
    rf_trees: int,
):
    mf_model, mf_fit_seconds, mf_fit_mem = _timed(fit_meta_stacker, development, config)
    mf_result, mf_predict_seconds, mf_predict_mem = _timed(predict_all, mf_model, test)
    observed = mf_result["observed"]
    truth = observed.traffic_label.to_numpy()

    original_model, original_fit_seconds, original_fit_mem = _timed(
        fit_original_cdr, development, config
    )
    original_all, original_predict_seconds, original_predict_mem = _timed(
        predict_fixed_cdr, original_model, test
    )
    _assert_common_rows(observed, original_all)
    original_prediction = original_all.loc[observed.index].to_numpy()
    embedded_original = mf_result["CDR_MLC_actual_router"]
    if not np.array_equal(original_prediction, embedded_original):
        mismatches = int(np.sum(original_prediction != embedded_original))
        raise RuntimeError(
            f"{definition['name']}: independently fitted Original CDR-MLC "
            f"differs from MF actual-router branch on {mismatches} rows"
        )

    if list(original_model["source_eligible_index"]) != list(
        mf_model["source_eligible_index"]
    ):
        raise RuntimeError(
            f"{definition['name']}: RF eligibility differs across model fits"
        )
    eligible = development.loc[mf_model["source_eligible_index"]]
    rf_model, rf_fit_seconds, rf_fit_mem = _timed(
        fit_rf, eligible, (), config.random_state, rf_trees
    )
    rf_all, rf_predict_seconds, rf_predict_mem = _timed(predict_rf, rf_model, test)
    rf_prediction = pd.Series(rf_all, index=test.index).loc[
        observed.index
    ].to_numpy()

    predictions = {
        "RF_Clean_Valid": rf_prediction,
        "Original_CDR_MLC": original_prediction,
        "MF_CDR_MLC": mf_result["CDR_MLC_meta_stacker"],
    }
    timing = {
        "RF_Clean_Valid": (rf_fit_seconds, rf_predict_seconds),
        "Original_CDR_MLC": (
            original_fit_seconds, original_predict_seconds
        ),
        "MF_CDR_MLC": (mf_fit_seconds, mf_predict_seconds),
    }
    _, ttfef_seconds, _ = _timed(
        trend_frame, test, TIMING, config.window
    )
    ttfef_us = 1e6 * ttfef_seconds / max(len(test), 1)
    memory = {
        "RF_Clean_Valid": resource_values(rf_fit_mem, rf_predict_mem),
        "Original_CDR_MLC": resource_values(
            original_fit_mem, original_predict_mem
        ),
        "MF_CDR_MLC": resource_values(mf_fit_mem, mf_predict_mem),
    }

    rows = []
    for method in METHODS:
        score = metrics(truth, predictions[method], APPLICATIONS)
        fit_seconds, predict_seconds = timing[method]
        input_rows = len(test)
        rows.append({
            "protocol": definition["name"],
            "evaluation_type": definition["evaluation_type"],
            "scenario": definition["scenario"],
            "source": definition["source"],
            "target": definition["target"],
            "seed": config.random_state,
            "method": method,
            **{key: score[key] for key in ("n", *METRICS)},
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_input_rows": input_rows,
            "throughput_input_rows_per_second": (
                input_rows / predict_seconds if predict_seconds > 0 else np.inf
            ),
            "inference_microseconds_per_input_row": (
                1e6 * predict_seconds / input_rows
            ),
            "peak_ram_mb": memory[method]["peak_ram_mb"],
            "peak_gpu_mb": memory[method]["peak_gpu_mb"],
            "ttfef_us_per_row": (
                np.nan if method == "RF_Clean_Valid" else ttfef_us
            ),
        })
    audit = {
        "protocol": definition,
        "development_rows_raw": len(development),
        "test_rows_raw": len(test),
        "evaluated_rows": len(observed),
        "development_identity": frame_identity(development),
        "test_identity": frame_identity(test),
        "evaluated_identity": frame_identity(observed),
        "development_composition": composition(development),
        "test_composition": composition(test),
        "original_matches_embedded_actual_router": True,
        "same_evaluated_rows_for_all_methods": True,
        "partition_rows": mf_model["partition_rows"],
        "leakage_control": mf_model["leakage_control"],
    }
    return rows, audit


def _mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, np.nan, np.nan, np.nan
    standard_deviation = float(values.std(ddof=1))
    half_width = float(
        t.ppf((1.0 + confidence) / 2.0, n - 1)
        * standard_deviation / np.sqrt(n)
    )
    return mean, standard_deviation, mean - half_width, mean + half_width


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, evaluation_type, method), group in runs.groupby(
        ["protocol", "evaluation_type", "method"], sort=False
    ):
        row = {
            "protocol": protocol,
            "evaluation_type": evaluation_type,
            "method": method,
            "runs": int(group.seed.nunique()),
            "n_test": int(group.n.iloc[0]),
        }
        for column in SUMMARY_VALUES:
            mean, standard_deviation, lower, upper = _mean_ci(group[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = standard_deviation
            row[f"{column}_ci95_low"] = lower
            row[f"{column}_ci95_high"] = upper
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["evaluation_type", "protocol", "method"])


def _rank_biserial(differences) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[~np.isclose(differences, 0.0)]
    if len(differences) == 0:
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / ranks.sum())


def _wilcoxon(differences):
    differences = np.asarray(differences, dtype=float)
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        differences,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _holm_adjust(pvalues) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        candidate = (count - position) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def pairwise_tests(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol, group in runs.groupby("protocol", sort=False):
        for metric in ("accuracy", "macro_f1"):
            pivot = group.pivot(index="seed", columns="method", values=metric)
            for candidate, baseline in PAIRWISE:
                paired = pivot[[candidate, baseline]].dropna()
                differences = paired[candidate] - paired[baseline]
                statistic, pvalue = _wilcoxon(differences)
                mean, std, lower, upper = _mean_ci(differences)
                tolerance = 1e-12
                rows.append({
                    "protocol": protocol,
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "pairs": len(paired),
                    "candidate_mean": float(paired[candidate].mean()),
                    "baseline_mean": float(paired[baseline].mean()),
                    "mean_difference": mean,
                    "difference_std": std,
                    "difference_ci95_low": lower,
                    "difference_ci95_high": upper,
                    "wins": int((differences > tolerance).sum()),
                    "ties": int((np.abs(differences) <= tolerance).sum()),
                    "losses": int((differences < -tolerance).sum()),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_pvalue": pvalue,
                    "rank_biserial_effect": _rank_biserial(differences),
                })
    result = pd.DataFrame(rows)
    result["wilcoxon_pvalue_holm"] = np.nan
    for metric, indices in result.groupby("metric").groups.items():
        result.loc[indices, "wilcoxon_pvalue_holm"] = _holm_adjust(
            result.loc[indices, "wilcoxon_pvalue"].to_numpy()
        )
    result["significant_holm_0_05"] = (
        result.wilcoxon_pvalue_holm < 0.05
    )
    return result.sort_values(["metric", "protocol", "candidate", "baseline"])


def _configuration_hash(arguments: dict) -> str:
    payload = json.dumps(arguments, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/confirmatory_rf_cdr_mlc",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82, 92, 102, 112, 122, 132],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
    )
    parser.add_argument(
        "--protocols",
        nargs="*",
        choices=list(PROTOCOLS),
        default=list(PROTOCOLS),
        help="mixed-level protocols; pass --protocols with no values to skip them",
    )
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument(
        "--target-test-fraction", type=float, default=0.20,
        help="ordered tail of each target sequence used in Scenarios 1--3",
    )
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument(
        "--congestion-features",
        nargs="+",
        default=list(DEFAULT_CONGESTION_FEATURES),
    )
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    args = parser.parse_args()

    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain distinct integers")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be in (0,1)")
    if not 0 < args.target_test_fraction <= 1:
        parser.error("--target-test-fraction must be in (0,1]")
    if not args.scenarios and not args.protocols:
        parser.error("at least one scenario or protocol is required")

    args.output.mkdir(parents=True, exist_ok=True)
    candidates = tuple(dict.fromkeys([*TIMING, *args.congestion_features]))
    data, input_audit = load_dataset(args.data_dir, candidates)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    evaluations = build_evaluations(
        data, args.scenarios, args.protocols, args.train_fraction,
        args.target_test_fraction,
    )

    run_rows = []
    protocol_audits = {}
    total = len(evaluations) * len(args.seeds)
    completed = 0
    for development, test, definition in evaluations:
        reference_audit = None
        for seed in args.seeds:
            completed += 1
            print(
                f"[{completed}/{total}] {definition['name']} seed={seed}",
                flush=True,
            )
            config = MetaStackConfig(
                window=args.window,
                congestion_window=args.congestion_window,
                congestion_features=tuple(args.congestion_features),
                expert_trees=args.expert_trees,
                utility_trees=args.utility_trees,
                meta_trees=args.meta_trees,
                random_state=seed,
            ).validate()
            rows, audit = evaluate_once(
                development, test, definition, config, args.rf_trees
            )
            run_rows.extend(rows)
            stable_audit = {
                key: value for key, value in audit.items()
                if key not in ("partition_rows", "leakage_control")
            }
            if reference_audit is None:
                reference_audit = stable_audit
                protocol_audits[definition["name"]] = audit
            elif stable_audit != reference_audit:
                raise RuntimeError(
                    f"{definition['name']}: partition identity changed across seeds"
                )

    runs = pd.DataFrame(run_rows).sort_values(
        ["evaluation_type", "protocol", "seed", "method"]
    )
    expected = len(evaluations) * len(args.seeds) * len(METHODS)
    if len(runs) != expected:
        raise RuntimeError(f"expected {expected} run rows, observed {len(runs)}")
    runs.to_csv(args.output / "confirmatory_runs.csv", index=False)
    runs[[
        "protocol",
        "evaluation_type",
        "seed",
        "method",
        "n",
        "predict_input_rows",
        "fit_seconds",
        "predict_seconds",
        "throughput_input_rows_per_second",
        "inference_microseconds_per_input_row",
        "peak_ram_mb",
        "peak_gpu_mb",
        "ttfef_us_per_row",
    ]].to_csv(args.output / "confirmatory_runtime.csv", index=False)

    summary = summarize_runs(runs)
    summary.to_csv(args.output / "confirmatory_summary.csv", index=False)
    tests = pairwise_tests(runs)
    tests.to_csv(args.output / "confirmatory_pairwise_tests.csv", index=False)

    arguments = {
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "protocols": args.protocols,
        "train_fraction": args.train_fraction,
        "target_test_fraction": args.target_test_fraction,
        "window": args.window,
        "congestion_window": args.congestion_window,
        "congestion_features": args.congestion_features,
        "expert_trees": args.expert_trees,
        "utility_trees": args.utility_trees,
        "meta_trees": args.meta_trees,
        "rf_trees": args.rf_trees,
    }
    manifest = {
        "arguments": arguments,
        "configuration_sha256": _configuration_hash(arguments),
        "methods": list(METHODS),
        "same_partition_within_protocol_for_all_methods_and_seeds": True,
        "full_target_scenarios_use_no_target_training_rows": True,
        "all_level_split_is_chronological_within_each_capture": True,
        "original_is_verified_against_mf_actual_router_every_run": True,
        "confidence_interval": "two-sided 95% Student-t interval across seeds",
        "statistical_test": (
            "two-sided paired Wilcoxon signed-rank test across seeds; "
            "Holm adjustment is applied separately to accuracy and macro-F1"
        ),
        "effect_size": (
            "matched-pairs rank-biserial correlation; positive favors candidate"
        ),
        "timing": (
            "wall-clock seconds; inference throughput uses all raw input rows, "
            "while predictive metrics use identical causally eligible rows"
        ),
        "protocol_audits": protocol_audits,
    }
    (args.output / "confirmatory_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    short_protocol = {
        "LM-H": "S4", "LH-M": "S5", "MH-L": "S6", "ALL-80-20": "S7"
    }
    console = summary.copy()
    console["protocol"] = console["protocol"].map(
        lambda value: (
            "S" + value.split("-")[1]
            if str(value).startswith("Scenario-")
            else short_protocol.get(str(value), str(value))
        )
    )
    console["method"] = console["method"].map({
        "RF_Clean_Valid": "RF",
        "Original_CDR_MLC": "O-CDR",
        "MF_CDR_MLC": "MF",
    })
    console = console.rename(columns={
        "n_test": "n",
        "accuracy_mean": "accuracy",
        "balanced_accuracy_mean": "balanced_accuracy",
        "macro_f1_mean": "macro_f1",
        "weighted_f1_mean": "weighted_f1",
        "fit_seconds_mean": "fit_seconds",
        "predict_seconds_mean": "predict_seconds",
        "inference_microseconds_per_input_row_mean":
            "inference_microseconds_per_input_row",
        "throughput_input_rows_per_second_mean":
            "throughput_input_rows_per_second",
        "peak_ram_mb_mean": "peak_ram_mb",
        "peak_gpu_mb_mean": "peak_gpu_mb",
        "ttfef_us_per_row_mean": "ttfef_us_per_row",
    })
    print_compact_results(console)
    print(
        "\nPaired significance tests are saved in "
        "confirmatory_pairwise_tests.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
