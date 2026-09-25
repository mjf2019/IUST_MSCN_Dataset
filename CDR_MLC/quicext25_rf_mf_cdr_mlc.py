"""Run RF, original CDR-MLC, and MF-CDR-MLC on QUICEXT-25 S1--S7.

The 20-class ontology, record partitions, and evaluated rows are identical for
all reported methods.  QUIC context measurements are exposed to the existing
CDR implementation through explicit compatibility aliases documented in the
run manifest; no TCP measurement is claimed to exist in this dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import adaptive_cdr_mlc as adaptive
import compare_clean_valid as compare
import congestion_feature_cdr_mlc as congestion
import learned_router_cdr_mlc as learned
import meta_stacked_cdr_mlc_leakage_safe as meta_stack
import utility_router_cdr_mlc as utility
from benchmarks.console_output import (
    ResourceMonitor,
    print_compact_results,
    resource_values,
)
from quicext25_common import (
    CONTEXT_ALIASES,
    PROTOCOL_IDS,
    build_protocol,
    load_class_spec,
    load_months,
    numeric_model_features,
    record_identity,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    dataset = root / "DATASETS/CESNET-QUICEXT-25"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=dataset / "processed")
    parser.add_argument("--class-spec", type=Path, default=dataset / "quicext25_classes.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/quicext25_rf_mf")
    parser.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS, default=list(PROTOCOL_IDS))
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def configure_fixed_classes(classes: tuple[str, ...]) -> None:
    """Set the existing generic RF/CDR modules to one immutable ontology."""
    fixed = list(classes)
    # These modules imported APPLICATIONS by value, so every owning module is
    # assigned explicitly before any model is fitted or prediction is made.
    for module in (adaptive, compare, congestion, learned, utility, meta_stack):
        module.APPLICATIONS = fixed


def score(truth, prediction, classes: tuple[str, ...]) -> dict:
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(
            truth, prediction, labels=classes, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            truth, prediction, labels=classes, average="weighted", zero_division=0
        )),
        "classification_report": classification_report(
            truth, prediction, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": list(classes),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=classes
        ).tolist(),
    }


def efficiency(seconds: float, input_rows: int) -> dict:
    return {
        "predict_seconds": seconds,
        "inference_input_rows": int(input_rows),
        "inference_us_per_row": seconds * 1e6 / max(input_rows, 1),
        "throughput_rows_per_second": input_rows / max(seconds, 1e-12),
    }


def add_result(
    rows: list[dict], detailed: dict, definition: dict, method: str,
    truth, prediction, classes, fit_seconds, prediction_seconds,
    input_rows, resources, ttfef_us_per_row=np.nan,
) -> None:
    result = score(truth, prediction, classes)
    detailed[method] = result
    rows.append({
        "protocol": definition["protocol"],
        "source": definition["source"],
        "target": definition["target"],
        "method": method,
        "class_count": len(classes),
        "test_coverage": definition["test_coverage"],
        "fit_seconds": fit_seconds,
        "ttfef_us_per_row": ttfef_us_per_row,
        **efficiency(prediction_seconds, input_rows),
        **resources,
        **{key: result[key] for key in (
            "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
        )},
        "status": "ok",
    })


def timed_context_features(frame, mf_model, window: int) -> tuple[float, float]:
    started = time.perf_counter()
    adaptive.trend_frame(frame, compare.TIMING, window)
    congestion.congestion_feature_frame(frame, mf_model["congestion_config"])
    elapsed = time.perf_counter() - started
    return elapsed, elapsed * 1e6 / max(len(frame), 1)


def run_scenario(
    development: pd.DataFrame,
    test: pd.DataFrame,
    definition: dict,
    classes: tuple[str, ...],
    args: argparse.Namespace,
    output: Path,
) -> tuple[list[dict], dict]:
    scenario_dir = output / definition["protocol"].lower()
    scenario_dir.mkdir(parents=True, exist_ok=True)

    config = meta_stack.MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(CONTEXT_ALIASES),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()

    with ResourceMonitor() as mf_fit_resource:
        started = time.perf_counter()
        mf_model = meta_stack.fit_meta_stacker(development, config)
        mf_fit_seconds = time.perf_counter() - started

    # Train the standalone original CDR-MLC, rather than treating the internal
    # actual-router diagnostic of MF-CDR-MLC as an independent baseline.
    with ResourceMonitor() as original_fit_resource:
        started = time.perf_counter()
        original_model = compare.fit_fixed_cdr(
            development, args.window, args.seed, args.expert_trees
        )
        original_fit_seconds = time.perf_counter() - started

    # RF uses the same expert input space and the same window-eligible source
    # rows as MF-CDR-MLC; aliases are excluded while their original QUIC source
    # columns remain present.
    eligible_development = development.loc[mf_model["source_eligible_index"]]
    expected_features = numeric_model_features(eligible_development)
    with ResourceMonitor() as rf_fit_resource:
        started = time.perf_counter()
        rf_model = compare.fit_rf(
            eligible_development, compare.TIMING, args.seed, args.rf_trees
        )
        rf_fit_seconds = time.perf_counter() - started
    if set(rf_model["numeric"]) != set(expected_features):
        missing = sorted(set(expected_features) - set(rf_model["numeric"]))
        extra = sorted(set(rf_model["numeric"]) - set(expected_features))
        raise RuntimeError(f"RF feature audit failed: missing={missing}, extra={extra}")

    with ResourceMonitor() as mf_predict_resource:
        started = time.perf_counter()
        mf_result = meta_stack.predict_all(mf_model, test)
        mf_predict_seconds = time.perf_counter() - started
    mf_observed = mf_result["observed"]
    mf_series = {
        method: pd.Series(values, index=mf_observed.index)
        for method, values in mf_result.items()
        if method.startswith("CDR_")
    }

    with ResourceMonitor() as original_predict_resource:
        started = time.perf_counter()
        original_prediction = compare.predict_fixed_cdr(original_model, test)
        original_predict_seconds = time.perf_counter() - started

    common = sorted(set(mf_observed.index) & set(original_prediction.index))
    if not common:
        raise ValueError(f"{definition['protocol']}: no common evaluated rows")
    observed = test.loc[common].copy()
    truth = observed.traffic_label.to_numpy()
    evaluated_identity = record_identity(observed)

    with ResourceMonitor() as rf_predict_resource:
        started = time.perf_counter()
        rf_prediction = compare.predict_rf(rf_model, observed)
        rf_predict_seconds = time.perf_counter() - started

    _, mf_ttfef_us = timed_context_features(test, mf_model, args.window)
    started = time.perf_counter()
    adaptive.trend_frame(test, compare.TIMING, args.window)
    original_ttfef_us = (
        (time.perf_counter() - started) * 1e6 / max(len(test), 1)
    )

    rows, detailed = [], {}
    add_result(
        rows, detailed, definition, "RF_Clean_Valid", truth, rf_prediction,
        classes, rf_fit_seconds, rf_predict_seconds, len(observed),
        resource_values(rf_fit_resource, rf_predict_resource),
    )
    add_result(
        rows, detailed, definition, "Original_CDR_MLC", truth,
        original_prediction.loc[common].to_numpy(), classes,
        original_fit_seconds, original_predict_seconds, len(test),
        resource_values(original_fit_resource, original_predict_resource),
        original_ttfef_us,
    )
    method_order = (
        "CDR_MLC_actual_router",
        "CDR_MLC_utility_router",
        "CDR_MLC_meta_stacker",
        "CDR_MLC_oracle_router",
    )
    for method in method_order:
        add_result(
            rows, detailed, definition, method, truth,
            mf_series[method].loc[common].to_numpy(), classes,
            mf_fit_seconds, mf_predict_seconds, len(test),
            resource_values(mf_fit_resource, mf_predict_resource),
            mf_ttfef_us,
        )

    predictions = observed[[
        "record_id", "period", "timestamp", "traffic_label"
    ]].copy()
    predictions["prediction_RF_Clean_Valid"] = rf_prediction
    predictions["prediction_Original_CDR_MLC"] = original_prediction.loc[common].to_numpy()
    for method in method_order:
        predictions[f"prediction_{method}"] = mf_series[method].loc[common].to_numpy()
    routes = mf_result["routes"].loc[common].reset_index(drop=True)
    pd.concat([predictions.reset_index(drop=True), routes], axis=1).to_parquet(
        scenario_dir / "predictions.parquet", index=False
    )
    (scenario_dir / "metrics.json").write_text(
        json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        **definition,
        "evaluated_rows": len(observed),
        "evaluated_identity": evaluated_identity,
        "class_counts_development": {
            label: int((development.traffic_label == label).sum()) for label in classes
        },
        "class_counts_test": {
            label: int((test.traffic_label == label).sum()) for label in classes
        },
        "class_counts_evaluated": {
            label: int((observed.traffic_label == label).sum()) for label in classes
        },
        "mf_config": asdict(config),
        "mf_partition_rows": mf_model["partition_rows"],
        "mf_selected_meta_variant": mf_model["selected_meta_variant"],
        "mf_selected_meta_confidence": mf_model["selected_meta_confidence"],
        "original_router_audit": original_model["router_audit"],
        "rf_features": rf_model["numeric"],
    }
    (scenario_dir / "split_model_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return rows, audit


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = load_class_spec(args.class_spec)
    classes = tuple(spec["classes"])
    configure_fixed_classes(classes)
    months, input_audit = load_months(args.data_dir)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)

    all_rows, audits = [], {}
    for scenario in args.scenarios:
        development, test, definition = build_protocol(months, scenario, classes)
        rows, audit = run_scenario(
            development, test, definition, classes, args, args.output
        )
        all_rows.extend(rows)
        audits[definition["protocol"]] = audit
        summary = pd.DataFrame(all_rows).sort_values(["protocol", "method"])
        summary.to_csv(args.output / "results.csv", index=False)
        print_compact_results(summary)

    manifest = {
        "dataset": "CESNET-QUICEXT-25",
        "seed": args.seed,
        "scenarios": args.scenarios,
        "fixed_classes": list(classes),
        "class_order": "lexical and immutable",
        "transport_context_mapping": CONTEXT_ALIASES,
        "context_interpretation": (
            "Protocol-aware QUIC timing/interaction proxies; not identical TCP measurements"
        ),
        "window": args.window,
        "congestion_window": args.congestion_window,
        "expert_trees": args.expert_trees,
        "utility_trees": args.utility_trees,
        "meta_trees": args.meta_trees,
        "rf_trees": args.rf_trees,
        "oracle_is_diagnostic_only": True,
        "audits": audits,
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
