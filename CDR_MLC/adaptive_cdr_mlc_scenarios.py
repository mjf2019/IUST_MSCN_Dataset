"""Leakage-free Adaptive CDR-MLC for paper scenarios 1-3.

Scenario 1: Low -> Medium
Scenario 2: Low -> High
Scenario 3: Medium -> High

Feature/window selection uses only a chronological split of the source level.
Because a source level contains only one known congestion condition, selection
uses unsupervised validation separation/stability, not three-level ARI.
The target level is untouched until final evaluation.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from adaptive_cdr_mlc import (
    APPLICATIONS,
    DEFAULT_CANDIDATES,
    Config,
    aligned_probabilities,
    common_rows,
    fit_final,
    load_dataset,
    predict,
    trend_frame,
)
from minibatch_clustering import make_minibatch_kmeans


SCENARIOS = {
    "1": ("Low", "Medium"),
    "2": ("Low", "High"),
    "3": ("Medium", "High"),
}


def split_source(source: pd.DataFrame, train_fraction: float = 0.75):
    """Chronological selection split inside each source-level capture."""
    train, validation = [], []
    for _, group in source.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(train_fraction * len(group))
        if not 0 < cut < len(group):
            raise ValueError(f"source capture too short: {group.sequence_id.iloc[0]}")
        train.append(group.iloc[:cut].copy())
        validation.append(group.iloc[cut:].copy())
    return {
        "train": pd.concat(train, ignore_index=True),
        "validation": pd.concat(validation, ignore_index=True),
    }


def source_candidate_ranking(
    source_train: pd.DataFrame,
    application: str,
    candidates,
    ranking_window: int,
    config: Config,
):
    """Rank individual fields without target-level observations or labels.

    Each field produces five trailing statistics. A three-cluster model is fit
    on source-training rows of one application. The score is training
    silhouette, used only to reduce the joint-search candidate set.
    """
    rows = []
    data = source_train[source_train.traffic_label.eq(application)]
    for feature in candidates:
        if feature not in data:
            continue
        windows = trend_frame(data, [feature], ranking_window)
        columns = [f"{feature}_{stat}" for stat in ("mean", "max", "median", "min", "std")]
        if len(windows) < max(20, config.n_clusters + 1):
            continue
        x = windows[columns].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x)
        z = scaler.transform(x)
        model = make_minibatch_kmeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        ).fit(z)
        labels = model.predict(z)
        counts = np.bincount(labels, minlength=config.n_clusters)
        if (counts == 0).any():
            continue
        sample = min(2000, len(z))
        score = float(
            silhouette_score(z, labels, sample_size=sample, random_state=config.random_state)
        )
        rows.append(
            {
                "application": application,
                "feature": feature,
                "ranking_window": ranking_window,
                "source_train_silhouette": score,
                "n": len(windows),
                "smallest_cluster_fraction": float(counts.min() / counts.sum()),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["source_train_silhouette", "smallest_cluster_fraction", "feature"],
        ascending=[False, False, True],
    )
    if len(ranking) < 3:
        raise ValueError(f"{application}: fewer than three usable source-level candidates")
    return ranking.reset_index(drop=True)


def evaluate_configuration(
    source_parts,
    application,
    features,
    window,
    config,
    min_cluster_fraction,
):
    """Source-only validation score with separation and seed stability."""
    train = trend_frame(
        source_parts["train"][source_parts["train"].traffic_label.eq(application)],
        features,
        window,
    )
    validation = trend_frame(
        source_parts["validation"][source_parts["validation"].traffic_label.eq(application)],
        features,
        window,
    )
    columns = [f"{feature}_{stat}" for feature in features for stat in ("mean", "max", "median", "min", "std")]
    if len(train) < max(20, config.n_clusters + 1) or len(validation) <= config.n_clusters:
        return None
    scaler = StandardScaler().fit(train[columns])
    train_z = scaler.transform(train[columns])
    validation_z = scaler.transform(validation[columns])
    labels_by_seed, silhouettes, balance = [], [], []
    for seed in config.selection_seeds:
        router = make_minibatch_kmeans(
            n_clusters=config.n_clusters,
            batch_size=config.batch_size,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=seed,
        ).fit(train_z)
        labels = router.predict(validation_z)
        counts = np.bincount(labels, minlength=config.n_clusters)
        if (counts == 0).any():
            return None
        smallest = float(counts.min() / counts.sum())
        if smallest < min_cluster_fraction:
            return None
        labels_by_seed.append(labels)
        balance.append(smallest)
        silhouettes.append(float(silhouette_score(validation_z, labels)))
    stability = []
    for left, right in itertools.combinations(labels_by_seed, 2):
        stability.append(float(adjusted_mutual_info_score(left, right)))
    # With one seed, stability is undefined rather than falsely reported as 1.
    stability_mean = float(np.mean(stability)) if stability else None
    silhouette_mean = float(np.mean(silhouettes))
    silhouette_std = float(np.std(silhouettes))
    return {
        "source_validation_silhouette_mean": silhouette_mean,
        "source_validation_silhouette_std": silhouette_std,
        "source_validation_seed_stability_ami": stability_mean,
        "smallest_cluster_fraction_mean": float(np.mean(balance)),
        "source_train_n": len(train),
        "source_validation_n": len(validation),
        # Selection prioritizes separation, then stability/balance, then smaller W.
        "selection_key": (
            silhouette_mean,
            stability_mean if stability_mean is not None else 0.0,
            float(np.mean(balance)),
            -silhouette_std,
            -window,
        ),
    }


def select_source_configurations(
    source_parts,
    config: Config,
    output: Path,
    min_cluster_fraction: float,
):
    """Choose label-specific fields/W without looking at the target level."""
    selections, ranking_rows, trial_rows = {}, [], []
    ranking_window = min(config.windows)
    for application in APPLICATIONS:
        ranking = source_candidate_ranking(
            source_parts["train"], application, config.candidates, ranking_window, config
        )
        ranking_rows.append(ranking)
        top = ranking.feature.head(min(config.ranking_top_k, len(ranking))).tolist()
        best = None
        for features in itertools.combinations(top, 3):
            for window in config.windows:
                result = evaluate_configuration(
                    source_parts,
                    application,
                    features,
                    window,
                    config,
                    min_cluster_fraction,
                )
                base = {
                    "application": application,
                    "features": "|".join(features),
                    "window": int(window),
                }
                if result is None:
                    trial_rows.append({**base, "status": "rejected_or_insufficient"})
                    continue
                trial_rows.append(
                    {**base, "status": "ok", **{k: v for k, v in result.items() if k != "selection_key"}}
                )
                candidate = (result["selection_key"], tuple(features))
                if best is None or candidate > best[0]:
                    best = (
                        candidate,
                        {
                            "features": list(features),
                            "window": int(window),
                            **{k: v for k, v in result.items() if k != "selection_key"},
                        },
                    )
        if best is None:
            raise ValueError(
                f"{application}: no valid source-only configuration; lower "
                "--min-cluster-fraction or reduce windows"
            )
        selections[application] = best[1]
    pd.concat(ranking_rows, ignore_index=True).to_csv(
        output / "source_feature_ranking.csv", index=False
    )
    pd.DataFrame(trial_rows).to_csv(output / "source_selection_trials.csv", index=False)
    (output / "selected_source_configurations.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    return selections


def source_as_fit_parts(source_parts):
    """fit_final expects train/validation and refits on their union."""
    return {
        "train": source_parts["train"],
        "validation": source_parts["validation"],
    }


def evaluate_target(model, target: pd.DataFrame, output: Path, scenario, source_level, target_level):
    """Evaluate once on the completely held-out target congestion level."""
    prediction, probabilities, routes, observed = predict(model, target)
    truth = observed.traffic_label.to_numpy()
    classes = model["classes"]
    gate_columns = [f"gate_{label}" for label in classes]
    gate_prediction = np.asarray(classes)[routes[gate_columns].to_numpy().argmax(axis=1)]
    metrics = {
        "scenario": scenario,
        "source_level": source_level,
        "target_level": target_level,
        "target_rows_raw": len(target),
        "target_rows_scored": len(observed),
        "adaptive_accuracy": float(accuracy_score(truth, prediction)),
        "adaptive_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "gate_only_accuracy": float(accuracy_score(truth, gate_prediction)),
        "classification_report": classification_report(
            truth, prediction, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
    }
    routes["true_label"] = truth
    routes["gate_prediction"] = gate_prediction
    routes["gate_correct"] = gate_prediction == truth
    routes["final_correct"] = prediction == truth
    for index, label in enumerate(classes):
        routes[f"probability_{label}"] = probabilities[:, index]
    routes.to_csv(output / "target_predictions.csv", index=False)
    (output / "target_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def run_source_group(data, source_level, targets, config, root_output, min_cluster_fraction, save_model):
    """Select/refit once per source level, then evaluate requested target levels."""
    source = data[data.congestion_level.eq(source_level)].copy()
    source_parts = split_source(source)
    selection_dir = root_output / f"source_{source_level.lower()}"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selections = select_source_configurations(
        source_parts, config, selection_dir, min_cluster_fraction
    )
    model = fit_final(source_as_fit_parts(source_parts), selections, config)
    if save_model:
        joblib.dump(model, selection_dir / "adaptive_source_model.joblib")
    results = {}
    for scenario, target_level in targets:
        output = root_output / f"scenario_{scenario}_{source_level.lower()}_to_{target_level.lower()}"
        output.mkdir(parents=True, exist_ok=True)
        target = data[data.congestion_level.eq(target_level)].copy().reset_index(drop=True)
        metrics = evaluate_target(
            model, target, output, scenario, source_level, target_level
        )
        manifest = {
            "protocol": "Target congestion level is not used in selection or fitting.",
            "scenario": scenario,
            "source_level": source_level,
            "target_level": target_level,
            "config": asdict(config),
            "selection_criterion": "source-validation silhouette; seed AMI and cluster balance are tie-breakers",
            "min_cluster_fraction": min_cluster_fraction,
            "selected": selections,
            "source_development_rows": model["development_rows"],
            "target_rows_scored": metrics["target_rows_scored"],
            "limitations": [
                "A single source congestion level cannot optimize three-level ARI.",
                "Silhouette separation does not prove physical congestion-level meaning.",
                "Source and target are separate capture files, but only one capture exists per application/level.",
                "Application-conditioned configuration is invoked through a predicted gate; true target labels are evaluation-only.",
            ],
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        results[scenario] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=root / "DATASETS/CDR-MLC/New_Version")
    parser.add_argument("--output", type=Path, default=root / "outputs/adaptive_scenarios")
    parser.add_argument("--scenarios", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"])
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 10, 20])
    parser.add_argument("--ranking-top-k", type=int, default=5)
    parser.add_argument("--selection-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gating", choices=["soft", "hard"], default="soft")
    parser.add_argument("--min-cluster-fraction", type=float, default=0.02)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_cluster_fraction < 1 / 3:
        parser.error("--min-cluster-fraction must be in [0, 1/3)")
    config = Config(
        windows=tuple(args.windows),
        candidates=tuple(args.candidates),
        ranking_top_k=args.ranking_top_k,
        selection_seeds=tuple(args.selection_seeds),
        gating=args.gating,
    ).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    data, audit = load_dataset(args.data_dir, config.candidates)
    audit.to_csv(args.output / "input_audit.csv", index=False)
    grouped = {}
    for scenario in args.scenarios:
        source, target = SCENARIOS[scenario]
        grouped.setdefault(source, []).append((scenario, target))
    all_results = {}
    for source, targets in grouped.items():
        all_results.update(
            run_source_group(
                data, source, targets, config, args.output,
                args.min_cluster_fraction, args.save_model,
            )
        )
    summary = [
        {
            "scenario": scenario,
            "source": result["source_level"],
            "target": result["target_level"],
            "adaptive_accuracy": result["adaptive_accuracy"],
            "adaptive_balanced_accuracy": result["adaptive_balanced_accuracy"],
            "gate_only_accuracy": result["gate_only_accuracy"],
            "target_rows_scored": result["target_rows_scored"],
        }
        for scenario, result in sorted(all_results.items())
    ]
    pd.DataFrame(summary).to_csv(args.output / "scenario_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
