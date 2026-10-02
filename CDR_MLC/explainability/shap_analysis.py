"""Post-hoc TreeSHAP for saved IUST_MSCN RF and MF meta-classifier models.

Loads the inference_resources artifacts without training or changing any model.
Constructs causal windows on complete captures BEFORE sampling explanation rows.
Meta explanations concern the selected meta-classifier, not the full pipeline
or its confidence fallback. Low examples are in-sample for the saved Low models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parent
sys.path.insert(0, str(CDR_MLC))


def feature_names(mf, context_columns, width):
    from adaptive_cdr_mlc import APPLICATIONS

    names = [f"distance_expert_{k + 1}" for k in range(3)]
    names += [f"prob_expert_{k + 1}_{label}"
              for k in range(3) for label in APPLICATIONS]
    for quantity in ("confidence", "margin", "entropy", "route"):
        names += [f"{quantity}_expert_{k + 1}" for k in range(3)]
    if list(context_columns) != list(mf["congestion_columns"]):
        raise ValueError("Saved and reconstructed congestion columns differ")
    names += list(context_columns)
    names += [f"utility_expert_{k + 1}" for k in range(3)]
    if len(names) != width:
        raise ValueError(f"Meta feature names: {len(names)} != input width {width}")
    return names


def balanced_positions(raw, rows_per_level, seed):
    """Fixed quota per level, spread equally over available application classes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    positions = []
    for level in ("Low", "Medium", "High"):
        level_positions = np.flatnonzero(raw.congestion_level.eq(level).to_numpy())
        if len(level_positions) == 0:
            raise ValueError(f"No eligible {level} records")
        classes = sorted(raw.iloc[level_positions].traffic_label.unique())
        if rows_per_level < len(classes):
            raise ValueError("--rows-per-level must cover every application class")
        quotas = [rows_per_level // len(classes) + (i < rows_per_level % len(classes))
                  for i in range(len(classes))]
        for label, quota in zip(classes, quotas):
            available = level_positions[
                raw.iloc[level_positions].traffic_label.eq(label).to_numpy()
            ]
            if len(available) < quota:
                raise ValueError(f"{level}/{label}: need {quota}, have {len(available)}")
            positions.extend(rng.choice(available, size=quota, replace=False).tolist())
    return np.asarray(sorted(positions), dtype=int)


def normalize_shap(values, rows, features, outputs):
    import numpy as np

    # SHAP <0.45 returns one array per output; >=0.45 uses a final output axis.
    if isinstance(values, list):
        result = np.stack(values, axis=-1)
    else:
        result = np.asarray(values)
        if result.ndim == 2 and outputs == 1:
            result = result[..., None]
    expected = (rows, features, outputs)
    if result.shape != expected:
        raise ValueError(f"Unexpected SHAP shape {result.shape}; expected {expected}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite SHAP values")
    return result


def explain(estimator, matrix, names, selected, output, label):
    import numpy as np
    import pandas as pd
    import shap

    print(f"START {label}: {len(matrix)} records, {len(names)} features", flush=True)
    started = time.perf_counter()
    explainer = shap.TreeExplainer(
        estimator, feature_perturbation="tree_path_dependent", model_output="raw"
    )
    values = explainer.shap_values(matrix, check_additivity=True)
    values = normalize_shap(values, len(matrix), len(names), len(estimator.classes_))
    seconds = time.perf_counter() - started
    importance = np.abs(values).mean(axis=(0, 2))
    table = pd.DataFrame({"feature": names, "mean_abs_shap": importance})
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(output / f"{label}_importance.csv", index=False)
    level_tables = []
    class_tables = []
    for level in ("Low", "Medium", "High"):
        mask = selected.congestion_level.eq(level).to_numpy()
        level_tables.append(pd.DataFrame({
            "level": level, "feature": names,
            "mean_abs_shap": np.abs(values[mask]).mean(axis=(0, 2)),
        }))
        for class_index, class_name in enumerate(estimator.classes_):
            class_tables.append(pd.DataFrame({
                "level": level, "output_class": class_name, "feature": names,
                "mean_abs_shap": np.abs(values[mask, :, class_index]).mean(axis=0),
            }))
    pd.concat(level_tables).to_csv(output / f"{label}_by_level.csv", index=False)
    pd.concat(class_tables).to_csv(output / f"{label}_by_output_class.csv", index=False)
    np.savez_compressed(
        output / f"{label}_shap_values.npz", values=values,
        feature_names=np.asarray(names), classes=np.asarray(estimator.classes_, dtype=str),
        record_ids=np.asarray(selected.record_id, dtype=str),
    )
    predictions = estimator.predict_proba(matrix)
    reconstructed = values.sum(axis=1) + np.asarray(explainer.expected_value)
    error = float(np.max(np.abs(predictions - reconstructed)))
    if error > 1e-4:
        raise ValueError(f"SHAP reconstruction error for {label}: {error}")
    print(f"DONE {label}: {seconds:.2f}s; max reconstruction error={error:.2g}", flush=True)
    print(table.head(10).to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    return table, values, {
        "model": label, "rows": len(matrix), "features": len(names),
        "classes": list(map(str, estimator.classes_)), "shap_seconds": seconds,
        "max_probability_reconstruction_error": error,
    }


def input_group(name):
    if name.startswith("router_"):
        return "Context: " + name.removeprefix("router_").rsplit("_", 2)[0]
    prefixes = {
        "distance_": "Cluster distances", "prob_": "Expert probabilities",
        "confidence_": "Expert confidence", "margin_": "Expert margins",
        "entropy_": "Expert entropy", "route_": "Cluster assignment",
        "utility_": "Expert utilities",
    }
    return next((group for prefix, group in prefixes.items() if name.startswith(prefix)), name)


def figures(rf_table, meta_table, grouped, output, top):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, tables, titles in (
        ("shap_feature_importance", [rf_table, meta_table], ["RF", "MF meta-classifier"]),
        ("shap_grouped_importance", [rf_table, grouped], ["RF", "MF meta-classifier input groups"]),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        for axis, table, title in zip(axes, tables, titles):
            view = table.head(top).iloc[::-1]
            colors = ["#d97706" if any(f in feature for f in ("AckDat", "TcpRtt", "SynAck"))
                      else "#2677a8" for feature in view.feature]
            axis.barh(view.feature, view.mean_abs_shap, color=colors)
            axis.set_title(title)
            axis.set_xlabel("Mean absolute SHAP value (class probability)")
            axis.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--models", type=Path, default=CDR_MLC / "benchmarks/inference_resources/artifacts/models")
    parser.add_argument("--rows-per-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--output", type=Path, default=HERE / "outputs/pilot")
    args = parser.parse_args()
    if args.rows_per_level < 1 or args.top_features < 1:
        parser.error("Sample and plot sizes must be positive")
    paths = {name: args.models / f"{name}.joblib" for name in ("rf", "mf-cdr")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("Missing saved models; no training is performed: " + ", ".join(missing))

    import joblib
    import numpy as np
    import pandas as pd
    import shap
    import sklearn
    from adaptive_cdr_mlc import DEFAULT_CANDIDATES, load_dataset
    from benchmarks.inference_resources.common import frame_sha256, record_id, set_estimator_jobs
    from congestion_selective_router_cdr_mlc import _enhanced_outputs
    from meta_stacked_cdr_mlc_leakage_safe import _meta_features
    from utility_router_cdr_mlc import _utility_matrix

    artifacts = {name: joblib.load(path) for name, path in paths.items()}
    for name, artifact in artifacts.items():
        if artifact.get("kind") != name or artifact.get("train_level") != "Low":
            raise ValueError(f"{name}: expected saved Low-only inference_resources artifact")
        set_estimator_jobs(artifact, 1)
    data, _ = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    low = data[data.congestion_level.eq("Low")]
    identity = frame_sha256(low)
    for name, artifact in artifacts.items():
        if artifact.get("train_identity_sha256") != identity:
            raise ValueError(f"{name}: dataset does not match saved training identity")
    mf = artifacts["mf-cdr"]["model"]
    # Never sample raw records before extracting their causal history.
    print("BUILD causal model inputs on complete captures; no fitting", flush=True)
    started = time.perf_counter()
    raw, _, _, _, _, base, columns = _enhanced_outputs(mf, data, mf["congestion_config"])
    positions = balanced_positions(raw, args.rows_per_level, args.seed)
    selected = raw.iloc[positions].copy().reset_index(drop=True)
    selected["record_id"] = record_id(selected)
    selected["sample_role"] = np.where(
        selected.congestion_level.eq("Low"), "training_level_posthoc", "unseen_level"
    )
    chosen_base = base[positions]
    utility = _utility_matrix(mf["utility_models"], mf["utility_constants"], chosen_base)
    meta_x = _meta_features(chosen_base, utility)
    meta_names = feature_names(mf, columns, meta_x.shape[1])
    rf = artifacts["rf"]["model"]
    rf_x = rf["preprocessor"].transform(selected[rf["numeric"] + rf["categorical"]])
    if hasattr(rf_x, "toarray"):
        rf_x = rf_x.toarray()
    rf_names = [name.split("__", 1)[-1] for name in rf["preprocessor"].get_feature_names_out()]
    if len(rf_names) != rf_x.shape[1]:
        raise ValueError("RF feature names do not match transformed columns")
    if not {"AckDat", "TcpRtt", "SynAck"}.issubset(rf_names):
        raise ValueError("Saved RF is missing one of the three congestion-sensitive features")
    preprocessing_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    selected[["record_id", "sequence_id", "source_row", "congestion_level",
              "traffic_label", "sample_role"]].to_csv(args.output / "sample_audit.csv", index=False)
    rf_table, _, rf_stats = explain(rf["model"], rf_x, rf_names, selected, args.output, "rf")
    meta_model = mf["meta_models"][mf["selected_meta_variant"]]
    meta_table, meta_values, meta_stats = explain(
        meta_model, meta_x, meta_names, selected, args.output, "meta"
    )
    groups = {}
    for i, name in enumerate(meta_names):
        groups.setdefault(input_group(name), []).append(i)
    # Add signed attributions within a group before taking absolute values.
    grouped = pd.DataFrame([
        {"feature": name, "mean_abs_shap": float(np.abs(meta_values[:, indices, :].sum(axis=1)).mean())}
        for name, indices in groups.items()
    ]).sort_values("mean_abs_shap", ascending=False)
    grouped.to_csv(args.output / "meta_grouped_importance.csv", index=False)
    figures(rf_table, meta_table, grouped, args.output, args.top_features)
    pd.DataFrame([rf_stats, meta_stats]).to_csv(args.output / "timing.csv", index=False)
    manifest = {
        "scope": "RF and selected MF meta-classifier; not end-to-end MF or fallback",
        "training_performed": False, "rows_per_level": args.rows_per_level,
        "total_rows": len(selected), "seed": args.seed,
        "feature_perturbation": "tree_path_dependent", "model_output": "raw (RF class probabilities)",
        "low_examples": "Training-level post-hoc explanations; not held-out test observations",
        "medium_high_examples": "Unseen congestion levels for the saved Low-only models",
        "causal_history": "Constructed before sampling; retained separately per sequence",
        "grouping": "Mean absolute sum of signed group attributions, averaged over output classes",
        "preprocessing_seconds": preprocessing_seconds,
        "selected_meta_variant": mf["selected_meta_variant"],
        "selected_meta_confidence": mf["selected_meta_confidence"],
        "versions": {"shap": shap.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "python": sys.version.split()[0]},
        "artifacts": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in paths.items()},
        "results": [rf_stats, meta_stats],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETED: {args.output}", flush=True)


if __name__ == "__main__":
    main()
