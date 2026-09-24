"""Compare fixed CDR-MLC, RF, and Adaptive CDR-MLC on Clean-Valid.

Scenarios:
  1: Low -> Medium
  2: Low -> High
  3: Medium -> High

For every scenario, all methods are scored on the identical intersection of
causal, complete target rows. Source models are reused for scenarios 1 and 2.
The target level is never used for fitting or adaptive configuration selection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from adaptive_cdr_mlc import (
    APPLICATIONS,
    Config,
    fit_final as fit_adaptive,
    load_dataset,
    make_preprocessor,
    predict as predict_adaptive,
    select_classifier_columns,
    trend_frame,
)
from adaptive_cdr_mlc_scenarios import split_source, select_source_configurations
from sensitive_cdr_mlc import (
    SensitiveConfig, predict as predict_sensitive, select_and_fit as fit_sensitive,
)


TIMING = ["TcpRtt", "SynAck", "AckDat"]
STATS = ["mean", "max", "median", "min", "std"]
SCENARIOS = {
    "1": ("Low", "Medium"),
    "2": ("Low", "High"),
    "3": ("Medium", "High"),
}
CLEAN_ROUTER_CANDIDATES = (
    "TcpRtt", "SynAck", "AckDat", "Dur", "SrcLoad", "DstLoad", "Load",
    "SrcRate", "DstRate", "Rate", "pLoss", "SrcLoss", "DstLoss", "Loss",
    "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes", "DstBytes",
)


def chronological_level_calibration(data: pd.DataFrame, fraction: float):
    """Return first-fraction calibration and disjoint remaining tails per capture."""
    calibration, held_out = [], []
    for _, group in data.groupby("capture_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * fraction)
        if fraction > 0 and not 0 < cut < len(group):
            raise ValueError(f"capture too short for adaptation fraction {fraction}")
        calibration.append(group.iloc[:cut].copy())
        held_out.append(group.iloc[cut:].copy())
    empty = data.iloc[:0].copy()
    return (
        pd.concat(calibration, ignore_index=True) if calibration and fraction > 0 else empty,
        pd.concat(held_out, ignore_index=True) if held_out else empty,
    )


def build_calibrated_protocol(data: pd.DataFrame, source_level: str,
                              adaptation_fraction: float):
    """Build source development data and disjoint target tails.

    A source level is retained in full. For Medium/High levels not already the
    source, the chronological prefix is added to development and the remaining
    tail is reserved. Thus Low-source scenarios share one frozen calibrated
    model, while Medium-source scenario 3 adds only the High prefix.
    """
    development_parts = [data[data.congestion_level.eq(source_level)].copy()]
    target_tails = {}
    calibration_audit = []
    for level in ("Medium", "High"):
        level_data = data[data.congestion_level.eq(level)].copy()
        if level == source_level:
            target_tails[level] = level_data.reset_index(drop=True)
            calibration_audit.append({
                "level": level, "role": "full_source_level",
                "calibration_rows": len(level_data), "held_out_rows": 0,
            })
            continue
        calibration, held_out = chronological_level_calibration(
            level_data, adaptation_fraction
        )
        if adaptation_fraction > 0:
            development_parts.append(calibration)
        target_tails[level] = held_out.reset_index(drop=True)
        calibration_audit.append({
            "level": level, "role": "target_level_calibration",
            "calibration_rows": len(calibration), "held_out_rows": len(held_out),
        })
    development = pd.concat(development_parts, ignore_index=True)
    return development, target_tails, calibration_audit


def rf_config(seed: int, trees: int) -> dict:
    return {
        "n_estimators": trees,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "n_jobs": -1,
        "random_state": seed,
    }


def fit_rf(source: pd.DataFrame, excluded_features, seed: int, trees: int) -> dict:
    numeric, categorical = select_classifier_columns(source, excluded_features)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(source[numeric + categorical])
    model = RandomForestClassifier(**rf_config(seed, trees)).fit(
        x, source.traffic_label.to_numpy()
    )
    return {
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "model": model,
        "train_rows": len(source),
    }


def predict_rf(model: dict, target: pd.DataFrame) -> np.ndarray:
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(target[columns])
    return model["model"].predict(x)


def fit_fixed_cdr(source: pd.DataFrame, window: int, seed: int,
                  expert_trees: int) -> dict:
    view = trend_frame(source, TIMING, window)
    trend_columns = [f"{feature}_{stat}" for feature in TIMING for stat in STATS]
    if len(view) < 3:
        raise ValueError("fixed CDR-MLC has fewer than three complete source windows")
    scaler = StandardScaler().fit(view[trend_columns])
    z = scaler.transform(view[trend_columns])
    router = MiniBatchKMeans(
        n_clusters=3, batch_size=1024, n_init=10, max_iter=100,
        random_state=seed,
    ).fit(z)
    routes = router.predict(z)
    if len(np.unique(routes)) != 3:
        raise ValueError("fixed CDR-MLC produced fewer than three source clusters")

    raw = source.loc[view.index]
    numeric, categorical = select_classifier_columns(raw, TIMING)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    labels = raw.traffic_label.to_numpy()
    experts = {}
    cluster_counts = []
    for cluster in range(3):
        mask = routes == cluster
        if not mask.any():
            raise ValueError(f"fixed CDR-MLC source cluster {cluster} is empty")
        experts[cluster] = RandomForestClassifier(
            **rf_config(seed, expert_trees)
        ).fit(x[mask], labels[mask])
        counts = pd.Series(labels[mask]).value_counts().to_dict()
        cluster_counts.append({
            "cluster": cluster,
            "rows": int(mask.sum()),
            **{f"class_{label}": int(counts.get(label, 0)) for label in APPLICATIONS},
        })
    return {
        "window": window,
        "trend_columns": trend_columns,
        "scaler": scaler,
        "router": router,
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "experts": experts,
        "source_eligible_index": view.index.tolist(),
        "cluster_counts": cluster_counts,
    }


def predict_fixed_cdr(model: dict, target: pd.DataFrame) -> pd.Series:
    view = trend_frame(target, TIMING, model["window"])
    routes = model["router"].predict(
        model["scaler"].transform(view[model["trend_columns"]])
    )
    raw = target.loc[view.index]
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(raw[columns])
    prediction = np.empty(len(raw), dtype=object)
    for cluster, expert in model["experts"].items():
        mask = routes == cluster
        if mask.any():
            prediction[mask] = expert.predict(x[mask])
    return pd.Series(prediction, index=raw.index, name="CDR_MLC")


def metrics(truth, prediction, labels) -> dict:
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            truth, prediction, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
    }


def fit_source_models(source: pd.DataFrame, source_level: str, output: Path,
                      config: Config, fixed_window: int, rf_trees: int,
                      expert_trees: int, min_cluster_fraction: float,
                      sensitive_config: SensitiveConfig) -> dict:
    source_output = output / f"source_{source_level.lower()}"
    source_output.mkdir(parents=True, exist_ok=True)
    fixed = fit_fixed_cdr(source, fixed_window, config.random_state, expert_trees)
    eligible_source = source.loc[fixed["source_eligible_index"]]
    rf_all = fit_rf(eligible_source, (), config.random_state, rf_trees)
    rf_expert_inputs = fit_rf(eligible_source, TIMING, config.random_state, rf_trees)

    parts = split_source(source)
    adaptive_selection_output = source_output / "adaptive_selection"
    adaptive_selection_output.mkdir(parents=True, exist_ok=True)
    selections = select_source_configurations(
        parts, config, adaptive_selection_output, min_cluster_fraction
    )
    adaptive = fit_adaptive(parts, selections, config)
    sensitive, sensitive_selection = fit_sensitive(
        source, source_output / "sensitive_selection", sensitive_config
    )
    pd.DataFrame(fixed["cluster_counts"]).to_csv(
        source_output / "fixed_cdr_cluster_counts.csv", index=False
    )
    return {
        "fixed": fixed,
        "rf_all": rf_all,
        "rf_expert_inputs": rf_expert_inputs,
        "adaptive": adaptive,
        "adaptive_selections": selections,
        "sensitive": sensitive,
        "sensitive_selection": sensitive_selection,
    }


def evaluate_scenario(models: dict, target: pd.DataFrame, scenario: str,
                      source_level: str, target_level: str, output: Path) -> list[dict]:
    scenario_output = output / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
    scenario_output.mkdir(parents=True, exist_ok=True)
    fixed_prediction = predict_fixed_cdr(models["fixed"], target)
    adaptive_prediction, _, adaptive_routes, adaptive_observed = predict_adaptive(
        models["adaptive"], target
    )
    adaptive_series = pd.Series(
        adaptive_prediction, index=adaptive_observed.index, name="Adaptive_CDR_MLC"
    )
    sensitive_prediction, sensitive_routes = predict_sensitive(models["sensitive"], target)
    common = sorted(
        set(fixed_prediction.index)
        & set(adaptive_series.index)
        & set(sensitive_prediction.index)
    )
    if not common:
        raise ValueError(f"scenario {scenario}: no common target rows")
    observed = target.loc[common].copy()
    truth = observed.traffic_label.to_numpy()
    predictions = {
        "CDR_MLC": fixed_prediction.loc[common].to_numpy(),
        "RF_all_clean_valid": predict_rf(models["rf_all"], observed),
        "RF_expert_inputs": predict_rf(models["rf_expert_inputs"], observed),
        "Adaptive_CDR_MLC": adaptive_series.loc[common].to_numpy(),
        "Sensitive_CDR_MLC": sensitive_prediction.loc[common].to_numpy(),
    }
    rows, detailed = [], {}
    for method, prediction in predictions.items():
        result = metrics(truth, prediction, APPLICATIONS)
        detailed[method] = result
        rows.append({
            "scenario": scenario,
            "source": source_level,
            "target": target_level,
            "method": method,
            **{key: result[key] for key in (
                "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
            )},
        })
    prediction_frame = observed[
        ["source_file", "source_row", "timestamp", "traffic_label", "congestion_level"]
    ].copy()
    for method, prediction in predictions.items():
        prediction_frame[f"prediction_{method}"] = prediction
        prediction_frame[f"correct_{method}"] = prediction == truth
    prediction_frame.to_csv(scenario_output / "predictions_common_rows.csv", index=False)
    (scenario_output / "metrics.json").write_text(
        json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
    )
    adaptive_routes.loc[common].to_csv(
        scenario_output / "adaptive_routes_common_rows.csv", index=False
    )
    sensitive_route_frame = pd.DataFrame(
        sensitive_routes,
        index=sensitive_prediction.index,
        columns=[f"sensitive_route_{cluster}" for cluster in range(sensitive_routes.shape[1])],
    )
    sensitive_route_frame.loc[common].to_csv(
        scenario_output / "sensitive_routes_common_rows.csv", index=False
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/clean_valid_comparison"
    )
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--fixed-window", type=int, default=3)
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 10, 20])
    parser.add_argument("--ranking-top-k", type=int, default=5)
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gating", choices=["soft", "hard"], default="soft")
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--min-cluster-fraction", type=float, default=.02)
    parser.add_argument("--sensitive-min-cluster-fraction", type=float, default=.10)
    parser.add_argument("--modulation-strength", type=float, default=1.0)
    parser.add_argument("--global-blend", type=float, default=.25)
    parser.add_argument("--soft-temperature", type=float, default=1.0)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--adaptation-fraction", type=float, default=0.0,
        help="Chronological prefix of non-source Medium/High captures added to training",
    )
    args = parser.parse_args()
    manifest_path = args.data_dir / "clean_valid_manifest.json"
    if not manifest_path.exists():
        parser.error("Clean-Valid manifest not found; run build_clean_valid.py first")
    if args.fixed_window < 1:
        parser.error("--fixed-window must be positive")
    if not 0 <= args.adaptation_fraction < 1:
        parser.error("--adaptation-fraction must be in [0,1)")
    args.output.mkdir(parents=True, exist_ok=True)

    config = Config(
        windows=tuple(args.windows),
        candidates=CLEAN_ROUTER_CANDIDATES,
        ranking_top_k=args.ranking_top_k,
        selection_seeds=tuple(args.selection_seeds),
        gating=args.gating,
        expert_estimators=args.expert_trees,
    ).validate()
    sensitive_config = SensitiveConfig(
        windows=tuple(args.windows), ranking_top_k=args.ranking_top_k,
        random_state=42, expert_trees=args.expert_trees,
        global_trees=args.rf_trees,
        modulation_strength=args.modulation_strength,
        global_blend=args.global_blend,
        soft_temperature=args.soft_temperature,
        min_cluster_fraction=args.sensitive_min_cluster_fraction,
    ).validate()
    data, audit = load_dataset(args.data_dir, config.candidates)
    audit.to_csv(args.output / "clean_valid_input_audit.csv", index=False)
    requested = {scenario: SCENARIOS[scenario] for scenario in args.scenarios}
    grouped = {}
    for scenario, (source, target) in requested.items():
        grouped.setdefault(source, []).append((scenario, target))

    summary = []
    model_manifests = {}
    for source_level, targets in grouped.items():
        source, target_tails, calibration_audit = build_calibrated_protocol(
            data, source_level, args.adaptation_fraction
        )
        models = fit_source_models(
            source, source_level, args.output, config, args.fixed_window,
            args.rf_trees, args.expert_trees, args.min_cluster_fraction,
            sensitive_config,
        )
        if args.save_models:
            joblib.dump(models, args.output / f"source_{source_level.lower()}" / "models.joblib")
        model_manifests[source_level] = {
            "fixed_cdr_source_rows": len(models["fixed"]["source_eligible_index"]),
            "fixed_cdr_numeric": models["fixed"]["numeric"],
            "fixed_cdr_categorical": models["fixed"]["categorical"],
            "rf_all_numeric": models["rf_all"]["numeric"],
            "rf_all_categorical": models["rf_all"]["categorical"],
            "rf_expert_inputs_numeric": models["rf_expert_inputs"]["numeric"],
            "rf_expert_inputs_categorical": models["rf_expert_inputs"]["categorical"],
            "adaptive_development_rows": models["adaptive"]["development_rows"],
            "adaptive_selections": models["adaptive_selections"],
            "sensitive_selection": models["sensitive_selection"],
            "development_rows_raw": len(source),
            "calibration_audit": calibration_audit,
        }
        for scenario, target_level in targets:
            target = target_tails[target_level]
            summary.extend(evaluate_scenario(
                models, target, scenario, source_level, target_level, args.output
            ))

    summary_frame = pd.DataFrame(summary).sort_values(["scenario", "method"])
    summary_frame.to_csv(args.output / "comparison_summary.csv", index=False)
    run_manifest = {
        "protocol": {
            "scenarios": requested,
            "identical_target_rows_within_each_scenario": True,
            "entire_target_level_unseen": args.adaptation_fraction == 0,
            "fixed_cdr_router_features": TIMING,
            "fixed_cdr_window": args.fixed_window,
            "rf_all_clean_valid_is_primary_rf": True,
            "rf_expert_inputs_excludes_fixed_router_features": TIMING,
            "adaptation_fraction": args.adaptation_fraction,
            "protocol_name": (
                "few_shot_multilevel_calibration"
                if args.adaptation_fraction > 0 else "unseen_cross_congestion"
            ),
            "target_level_prefix_used_for_calibration": args.adaptation_fraction > 0,
            "held_out_target_tail_used_in_training_or_selection": False,
        },
        "adaptive_config": asdict(config),
        "sensitive_config": asdict(sensitive_config),
        "rf_trees": args.rf_trees,
        "expert_trees": args.expert_trees,
        "clean_valid_manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "source_models": model_manifests,
        "limitations": [
            "The Clean-Valid policy was derived from a whole-dataset audit.",
            "Only one capture exists per application/congestion combination.",
            "Source-only adaptive selection cannot validate correspondence to three congestion levels.",
            "The fixed and adaptive routers may use different source-window counts; target scoring rows are identical.",
            "Sensitivity modulation is source-only pseudo-severity, not a true High-level weight.",
            "With adaptation_fraction > 0, target capture prefixes and held-out tails may share TCP connections; this is not independent-capture generalization.",
        ],
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
