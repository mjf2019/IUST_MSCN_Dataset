"""Oracle-routing upper bound for CDR-MLC at 0% and 20% calibration.

The oracle evaluates every expert for each test record and uses the true
application label only to select an expert that predicts correctly when one
exists. This deliberate label leakage is diagnostic only: it measures the
maximum gain available from routing and is not a deployable result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, aligned_probabilities, load_dataset, trend_frame
from compare_clean_valid import (
    SCENARIOS, TIMING, build_calibrated_protocol, fit_fixed_cdr,
    fit_rf, metrics, predict_fixed_cdr, predict_rf,
)


def oracle_predict(model: dict, target: pd.DataFrame):
    """Return actual routing, oracle routing and per-row routing diagnostics."""
    view = trend_frame(target, TIMING, model["window"])
    raw = target.loc[view.index]
    z = model["scaler"].transform(view[model["trend_columns"]])
    actual_route = model["router"].predict(z)
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(raw[columns])

    probability = []
    prediction = []
    for cluster in range(3):
        expert = model["experts"][cluster]
        cluster_probability = aligned_probabilities(expert, x, APPLICATIONS)
        probability.append(cluster_probability)
        prediction.append(np.asarray(APPLICATIONS)[cluster_probability.argmax(axis=1)])
    probability = np.stack(probability, axis=1)  # rows x experts x classes
    prediction = np.stack(prediction, axis=1)    # rows x experts
    truth = raw.traffic_label.to_numpy()
    true_index = np.array([APPLICATIONS.index(label) for label in truth])
    correct = prediction == truth[:, None]

    # If one or more experts are correct, select the correct expert assigning
    # the largest probability to the true class. Otherwise select the expert
    # with the largest true-class probability; its final prediction remains an
    # error, preserving the classifier ceiling rather than forcing truth.
    true_probability = probability[
        np.arange(len(raw))[:, None], np.arange(3)[None, :], true_index[:, None]
    ]
    oracle_score = np.where(correct, true_probability + 2.0, true_probability)
    oracle_route = oracle_score.argmax(axis=1)
    oracle_prediction = prediction[np.arange(len(raw)), oracle_route]
    actual_prediction = prediction[np.arange(len(raw)), actual_route]
    report = pd.DataFrame({
        "actual_route": actual_route,
        "oracle_route": oracle_route,
        "actual_route_is_oracle": actual_route == oracle_route,
        "any_expert_correct": correct.any(axis=1),
        "number_of_correct_experts": correct.sum(axis=1),
    }, index=raw.index)
    for cluster in range(3):
        report[f"expert_{cluster}_prediction"] = prediction[:, cluster]
        report[f"expert_{cluster}_true_probability"] = true_probability[:, cluster]
    return (
        pd.Series(actual_prediction, index=raw.index, name="CDR_MLC"),
        pd.Series(oracle_prediction, index=raw.index, name="Oracle_CDR_MLC"),
        report,
    )


def run_fraction(data, fraction: float, scenarios, output: Path,
                 window: int, seed: int, expert_trees: int, rf_trees: int):
    fraction_name = f"cal_{int(round(fraction * 100)):02d}"
    root = output / fraction_name
    root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for scenario in scenarios:
        source, target = SCENARIOS[scenario]
        grouped.setdefault(source, []).append((scenario, target))
    summary, protocol_audit = [], {}
    for source_level, targets in grouped.items():
        development, target_tails, calibration_audit = build_calibrated_protocol(
            data, source_level, fraction
        )
        fixed = fit_fixed_cdr(development, window, seed, expert_trees)
        eligible_development = development.loc[fixed["source_eligible_index"]]
        rf = fit_rf(eligible_development, TIMING, seed, rf_trees)
        source_dir = root / f"source_{source_level.lower()}"
        source_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fixed["cluster_counts"]).to_csv(
            source_dir / "cluster_class_counts.csv", index=False
        )
        protocol_audit[source_level] = {
            "development_rows_raw": len(development),
            "development_rows_window_eligible": len(eligible_development),
            "calibration": calibration_audit,
        }
        for scenario, target_level in targets:
            target = target_tails[target_level]
            actual, oracle, route_report = oracle_predict(fixed, target)
            common = actual.index.tolist()
            observed = target.loc[common]
            truth = observed.traffic_label.to_numpy()
            predictions = {
                "CDR_MLC_actual_router": actual.to_numpy(),
                "CDR_MLC_oracle_router": oracle.to_numpy(),
                "RF_expert_inputs": predict_rf(rf, observed),
            }
            scenario_dir = root / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            detailed = {}
            for method, values in predictions.items():
                result = metrics(truth, values, APPLICATIONS)
                detailed[method] = result
                summary.append({
                    "adaptation_fraction": fraction,
                    "scenario": scenario, "source": source_level,
                    "target": target_level, "method": method,
                    **{key: result[key] for key in (
                        "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
                    )},
                })
            prediction_frame = observed[
                ["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]
            ].copy()
            for method, values in predictions.items():
                prediction_frame[f"prediction_{method}"] = values
            prediction_frame = prediction_frame.join(route_report)
            prediction_frame.to_csv(scenario_dir / "oracle_predictions.csv", index=False)
            (scenario_dir / "metrics.json").write_text(
                json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
            )
    return summary, protocol_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=root / "outputs/oracle_routing_sweep")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.20])
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    args = parser.parse_args()
    if any(not 0 <= fraction < 1 for fraction in args.fractions):
        parser.error("all fractions must be in [0,1)")
    if args.window < 1:
        parser.error("--window must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_dataset(args.data_dir, tuple(TIMING))
    audit.to_csv(args.output / "input_audit.csv", index=False)
    all_rows, fraction_audits = [], {}
    for fraction in args.fractions:
        rows, protocol = run_fraction(
            data, fraction, args.scenarios, args.output, args.window,
            args.seed, args.expert_trees, args.rf_trees,
        )
        all_rows.extend(rows)
        fraction_audits[str(fraction)] = protocol
    summary = pd.DataFrame(all_rows).sort_values(
        ["adaptation_fraction", "scenario", "method"]
    )
    summary.to_csv(args.output / "oracle_summary.csv", index=False)
    manifest = {
        "diagnostic_only": True,
        "oracle_definition": (
            "True application label selects a correct expert when available; otherwise the expert "
            "with maximum true-class probability. This is intentional inference leakage."
        ),
        "fractions": args.fractions,
        "scenarios": {scenario: SCENARIOS[scenario] for scenario in args.scenarios},
        "window": args.window,
        "seed": args.seed,
        "expert_trees": args.expert_trees,
        "rf_trees": args.rf_trees,
        "protocol_audit": fraction_audits,
        "interpretation": [
            "Oracle minus actual CDR measures the maximum gain available from routing.",
            "Oracle accuracy below 1 means no trained expert predicts some records correctly.",
            "The 20% protocol is few-shot calibration, not unseen cross-congestion.",
            "Oracle results must never be reported as deployable model performance.",
        ],
    }
    (args.output / "oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
