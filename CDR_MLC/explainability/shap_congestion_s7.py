"""Explain congestion-level discrimination on the chronological S7 split.

Diagnostic RF: 24 original numerical Clean-Valid features -> Low/Medium/High.
No categorical encoding, window-derived columns, or deployed MF model changes.
Training is performed only when the USER runs the script and no matching cached
diagnostic model exists. SHAP uses held-out records balanced by level/application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parent
sys.path.insert(0, str(CDR_MLC))
SENSITIVE = ("AckDat", "SynAck", "TcpRtt")


def fingerprint(development, features, config):
    import pandas as pd

    columns = ["source_file", "source_row", "traffic_label", "congestion_level", *features]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(development[columns], index=False).values.tobytes())
    return digest.hexdigest()


def validate_split(development, test, features):
    required = set(features) | {"source_file", "source_row", "sequence_id",
                                "traffic_label", "congestion_level"}
    for name, frame in (("development", development), ("test", test)):
        if required - set(frame.columns):
            raise ValueError(f"{name}: missing {sorted(required - set(frame.columns))}")
        if frame[["source_file", "source_row"]].duplicated().any():
            raise ValueError(f"{name}: duplicate record identities")
    left = set(zip(development.source_file, development.source_row))
    right = set(zip(test.source_file, test.source_row))
    if left & right:
        raise ValueError("Development/test record overlap")
    for sequence, group in development.groupby("sequence_id", sort=False):
        tail = test[test.sequence_id.eq(sequence)]
        if tail.empty or group.timestamp.max() > tail.timestamp.min():
            raise ValueError(f"{sequence}: invalid chronological partition boundary")


def train_or_load(development, features, args):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    config = {"schema_version": 1, "features": features, "target": "congestion_level",
              "train_fraction": args.train_fraction, "seed": args.seed,
              "trees": args.trees, "min_samples_leaf": args.min_samples_leaf,
              "weighting": "equal total weight for each level/application stratum"}
    identity = fingerprint(development, features, config)
    if args.cache.is_file() and not args.force_refit:
        artifact = joblib.load(args.cache)
        if artifact.get("signature") != identity:
            raise ValueError("Cached diagnostic model does not match data/config; use --force-refit or another --cache")
        print(f"LOAD matching S7 diagnostic model: {args.cache}", flush=True)
        return artifact, True
    print(f"FIT diagnostic RF: {len(development)} development records, {len(features)} numeric features", flush=True)
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    matrix = imputer.fit_transform(development[features])
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite transformed training features")
    # Application labels are used only for development weighting, not as inputs.
    strata = development.groupby(["congestion_level", "traffic_label"], observed=True)
    counts = strata.source_row.transform("size").to_numpy(dtype=float)
    weights = len(development) / (strata.ngroups * counts)
    model = RandomForestClassifier(
        n_estimators=args.trees, min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt", class_weight=None, n_jobs=1, random_state=args.seed,
    ).fit(matrix, development.congestion_level.to_numpy(), sample_weight=weights)
    seconds = time.perf_counter() - started
    artifact = {"signature": identity, "config": config, "features": features,
                "imputer": imputer, "model": model, "training_seconds": seconds,
                "weighted_strata": int(strata.ngroups), "training_rows": len(development)}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.cache, compress=3)
    print(f"DONE fit: {seconds:.2f}s; saved locally: {args.cache}", flush=True)
    return artifact, False


def scores(truth, prediction):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {"Acc": float(accuracy_score(truth, prediction)),
            "BAcc": float(balanced_accuracy_score(truth, prediction)),
            "MF1": float(f1_score(truth, prediction, average="macro", zero_division=0))}


def rankings(values, features, sampled, classes, output):
    import numpy as np
    import pandas as pd

    def table(block):
        view = pd.DataFrame({"Feature": features,
                             "SHAP": np.abs(block).mean(axis=(0, 2))})
        view = view.sort_values("SHAP", ascending=False).reset_index(drop=True)
        view.insert(0, "Rank", np.arange(1, len(view) + 1))
        return view

    global_table = table(values)
    global_table.to_csv(output / "congestion_shap_importance.csv", index=False)
    by_level, by_application, by_output = [], [], []
    for level in ("Low", "Medium", "High"):
        mask = sampled.congestion_level.eq(level).to_numpy()
        view = table(values[mask])
        view.insert(0, "Level", level)
        by_level.append(view)
    for label in sorted(sampled.traffic_label.unique()):
        mask = sampled.traffic_label.eq(label).to_numpy()
        view = table(values[mask])
        view.insert(0, "Application", label)
        by_application.append(view)
    for i, name in enumerate(classes):
        view = table(values[:, :, i:i + 1])
        view.insert(0, "OutputLevel", str(name))
        by_output.append(view)
    level_table = pd.concat(by_level, ignore_index=True)
    level_table.to_csv(output / "congestion_shap_by_level.csv", index=False)
    pd.concat(by_application, ignore_index=True).to_csv(output / "congestion_shap_by_application.csv", index=False)
    pd.concat(by_output, ignore_index=True).to_csv(output / "congestion_shap_by_output_level.csv", index=False)
    focus = global_table[global_table.Feature.isin(SENSITIVE)]
    focus.to_csv(output / "timing_features_shap.csv", index=False)
    return global_table, level_table, focus


def plot(global_table, by_level, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    shown = global_table.head(15).iloc[::-1]
    colors = ["#d97706" if name in SENSITIVE else "#2677a8" for name in shown.Feature]
    axes[0].barh(shown.Feature, shown.SHAP, color=colors)
    axes[0].set_title("Congestion-level classifier: feature importance")
    axes[0].set_xlabel("Mean absolute SHAP value")
    positions = np.arange(len(SENSITIVE))
    width = .24
    for i, (level, color) in enumerate(zip(("Low", "Medium", "High"), ("#2677a8", "#d97706", "#38856d"))):
        lookup = by_level[by_level.Level.eq(level)].set_index("Feature").SHAP
        axes[1].bar(positions + (i - 1) * width, lookup.loc[list(SENSITIVE)],
                    width=width, label=level, color=color)
    axes[1].set_xticks(positions, SENSITIVE)
    axes[1].set_ylabel("Mean absolute SHAP value")
    axes[1].set_title("TCP timing features by observed congestion level")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "congestion_shap_s7.pdf", bbox_inches="tight")
    fig.savefig(output / "congestion_shap_s7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--rows-per-level", type=int, default=10)
    parser.add_argument("--trees", type=int, default=110)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=HERE / "outputs/s7_models/congestion_rf.joblib")
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/s7_pilot")
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between zero and one")
    if min(args.rows_per_level, args.trees, args.min_samples_leaf) < 1:
        parser.error("Sample, tree and leaf counts must be positive")

    import numpy as np
    import pandas as pd
    import shap
    import sklearn
    from sklearn.metrics import confusion_matrix
    from adaptive_cdr_mlc import DEFAULT_CANDIDATES, load_dataset
    from build_clean_valid import NUMERIC
    from mixed_level_protocols_leakage_safe import build_protocol, frame_identity
    from shap_analysis import balanced_positions, normalize_shap

    features = list(NUMERIC)
    if len(features) != 24 or len(set(features)) != 24 or not set(SENSITIVE).issubset(features):
        raise ValueError("Expected the current 24-field numerical Clean-Valid schema")
    print("S7 | Target=congestion level | Inputs=24 original numeric features | No One-Hot", flush=True)
    data, _ = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    for feature in features:
        if feature not in data:
            raise ValueError(f"Missing numerical feature: {feature}")
        data[feature] = pd.to_numeric(data[feature], errors="raise").replace([np.inf, -np.inf], np.nan)
    development, test, protocol = build_protocol(data, "ALL-80-20", args.train_fraction)
    validate_split(development, test, features)
    # Fix the SHAP sample before fitting; do not select samples by predictions.
    positions = balanced_positions(test, args.rows_per_level, args.seed)
    sampled = test.iloc[positions].copy().reset_index(drop=True)
    print(f"SPLIT development={len(development)} test={len(test)} SHAP={len(sampled)}", flush=True)
    artifact, reused = train_or_load(development, features, args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "features.json").write_text(json.dumps(features, indent=2) + "\n", encoding="utf-8")
    test_x = artifact["imputer"].transform(test[features])
    if not np.isfinite(test_x).all() or test_x.shape[1] != 24:
        raise ValueError("Invalid transformed test matrix")
    estimator = artifact["model"]
    prediction = estimator.predict(test_x)
    metric_rows = [dict(Scope="FullTest", N=len(test), **scores(test.congestion_level, prediction)),
                   dict(Scope="SHAPSample", N=len(sampled), **scores(sampled.congestion_level, prediction[positions]))]
    metric_table = pd.DataFrame(metric_rows)
    metric_table.to_csv(args.output / "diagnostic_metrics.csv", index=False)
    levels = ["Low", "Medium", "High"]
    pd.DataFrame(confusion_matrix(test.congestion_level, prediction, labels=levels),
                 index=levels, columns=levels).to_csv(args.output / "confusion_matrix.csv", index_label="TrueLevel")
    sampled["record_id"] = sampled.source_file.astype(str) + ":" + sampled.source_row.astype(str)
    sampled[["record_id", "sequence_id", "source_row", "traffic_label", "congestion_level"]].to_csv(
        args.output / "sample_audit.csv", index=False)
    matrix = test_x[positions]
    print(f"START TreeSHAP: {len(matrix)} held-out records, {len(features)} features", flush=True)
    started = time.perf_counter()
    explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent", model_output="raw")
    values = normalize_shap(explainer.shap_values(matrix, check_additivity=True),
                            len(matrix), len(features), len(estimator.classes_))
    shap_seconds = time.perf_counter() - started
    reconstructed = values.sum(axis=1) + np.asarray(explainer.expected_value)
    error = float(np.max(np.abs(estimator.predict_proba(matrix) - reconstructed)))
    if error > 1e-4:
        raise ValueError(f"SHAP probability reconstruction error: {error}")
    global_table, level_table, focus = rankings(values, features, sampled, estimator.classes_, args.output)
    np.savez_compressed(args.output / "congestion_shap_values.npz", values=values,
                        feature_names=np.asarray(features), output_levels=np.asarray(estimator.classes_, dtype=str),
                        record_ids=np.asarray(sampled.record_id, dtype=str))
    plot(global_table, level_table, args.output)
    print(f"DONE TreeSHAP: {shap_seconds:.2f}s; max reconstruction error={error:.2g}", flush=True)
    print("Columns: Scope=evaluation rows; N=records; Acc=accuracy; BAcc=balanced accuracy; MF1=macro-F1", flush=True)
    print(metric_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print("Importance: Rank=global feature rank; Feature=input name; SHAP=mean absolute attribution", flush=True)
    print(global_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print("TCP timing features (their ranks are measured, not assumed):", flush=True)
    print(focus.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    manifest = {
        "purpose": "Diagnostic congestion-level discrimination, not application classification or the deployed MF pipeline",
        "protocol": "S7 / ALL-80-20", "partition_definition": protocol,
        "feature_names": features, "feature_count": 24, "categorical_encoding": False,
        "development_identity": frame_identity(development), "test_identity": frame_identity(test),
        "training_rows": len(development), "test_rows": len(test), "shap_rows": len(sampled),
        "sample_counts": sampled.groupby(["congestion_level", "traffic_label"]).size().rename("rows").reset_index().to_dict(orient="records"),
        "seed": args.seed, "rows_per_level": args.rows_per_level,
        "training": artifact["config"], "model_reused": reused,
        "training_seconds": artifact["training_seconds"], "shap_seconds": shap_seconds,
        "training_data_signature": artifact["signature"], "model_cache": str(args.cache),
        "max_probability_reconstruction_error": error,
        "shap_settings": {"feature_perturbation": "tree_path_dependent", "output": "class probability"},
        "interpretation": "Predictive association with congestion level; not evidence of causal effects or guaranteed feature ranks",
        "versions": {"python": platform.python_version(), "shap": shap.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "diagnostic_metrics": metric_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETED: {args.output}", flush=True)


if __name__ == "__main__":
    main()
