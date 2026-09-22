"""Clean legacy Long/Short shuffled datasets and evaluate Long -> Short.

This is a cross-domain experiment, not a Low/Medium/High congestion scenario.
Both raw files are curated with one frozen feature policy. Exact cleaned-row
overlap is audited before fitting. Raw files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS
from compare_clean_valid import (
    TIMING, fit_fixed_cdr, fit_rf, metrics, predict_fixed_cdr, predict_rf,
)
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES
from meta_stacked_cdr_mlc import (
    MetaStackConfig, fit_meta_stacker as fit_legacy_meta,
    predict_all as predict_legacy_meta,
)
from meta_stacked_cdr_mlc_leakage_safe import (
    fit_meta_stacker as fit_safe_meta,
    predict_all as predict_safe_meta,
)


NUMERIC = [
    "Dur", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes",
    "DstBytes", "dMeanPktSz", "SrcLoad", "DstLoad", "Load", "SrcRate",
    "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", "pLoss", "SrcWin",
    "DstWin", "TcpRtt", "SynAck", "AckDat",
]
REMOVAL_POLICY = {
    "capture_fingerprint_or_unresolved_semantics": ["IdleTime"],
    "constant_or_protocol_fingerprint": [
        "pRetran", "SrcRetra", "PCRatio", "dTtl", "sTtl", "DstRetra",
        "StdDev",
    ],
    "invalid_or_effectively_empty_gap": ["SrcGap", "DstGap"],
    "redundant_duration_alias_keep_Dur": ["Sum", "Min", "Mean", "Max", "Dur.1"],
    "labels_or_export_metadata": ["label", "Label", "Cause", "Dir", "Proto.1"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: object) -> str:
    mapping = {
        "http": "HTTP", "sftp": "SFTP", "smtp": "SMTP",
        "ssh": "SSH", "video": "Video",
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"unknown application label {value!r}")
    return mapping[key]


def clean_domain(path: Path, domain: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False, on_bad_lines="error")
    raw.columns = raw.columns.str.strip()
    label_column = "label" if "label" in raw.columns else "Label"
    if label_column not in raw:
        raise ValueError(f"{path}: missing label column")
    missing = sorted(set(NUMERIC) - set(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing reviewed fields {missing}")

    clean = pd.DataFrame(index=np.arange(len(raw)))
    invalid = {}
    for column in NUMERIC:
        values = pd.to_numeric(raw[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        negative = values.lt(0)
        if negative.any():
            raise ValueError(
                f"{path.name}: retained field {column} has "
                f"{int(negative.sum())} negative values"
            )
        clean[column] = values.to_numpy()
        invalid[column] = int(values.isna().sum())

    clean["traffic_label"] = raw[label_column].map(normalize_label).to_numpy()
    # The raw files contain no reliable event timestamp. This monotonically
    # increasing value preserves file order and is never a classifier input.
    clean["timestamp"] = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(np.arange(len(clean)), unit="ms")
    )
    clean["source_row"] = np.arange(len(clean)) + 2
    clean["source_file"] = path.name
    clean["sequence_id"] = domain
    clean["congestion_level"] = domain
    counts = clean.traffic_label.value_counts().to_dict()
    return clean, {
        "domain": domain,
        "source": str(path.resolve()),
        "source_sha256": sha256(path),
        "raw_rows": int(len(raw)),
        "retained_rows": int(len(clean)),
        "source_columns": list(raw.columns),
        "retained_numeric": NUMERIC,
        "missing_by_retained_numeric": invalid,
        "class_counts": {
            label: int(counts.get(label, 0)) for label in APPLICATIONS
        },
    }


def row_hash(frame: pd.DataFrame) -> pd.Series:
    comparable = frame[NUMERIC + ["traffic_label"]].copy()
    # Stable string representation makes the audit independent of DataFrame
    # indices and file ordering.
    for column in NUMERIC:
        comparable[column] = comparable[column].map(
            lambda value: "<NA>" if pd.isna(value) else format(float(value), ".17g")
        )
    return pd.util.hash_pandas_object(comparable, index=False)


def indexed_predictions(result: dict, prefix: str) -> dict[str, pd.Series]:
    index = result["observed"].index
    return {
        f"{prefix}{name}": pd.Series(value, index=index)
        for name, value in result.items()
        if name.startswith("CDR_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    repository = root.parent
    parser.add_argument(
        "--train-file", type=Path,
        default=repository / "DATASETS/CDR-MLC/scale_1/Long/CDR-MLC-Shuffle.csv",
    )
    parser.add_argument(
        "--test-file", type=Path,
        default=repository / "DATASETS/CDR-MLC/scale_1/Short/CDR-MLC-Shuffle.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/legacy_long_to_short",
    )
    parser.add_argument(
        "--overlap-policy", choices=["error", "drop", "allow"], default="error",
        help="Action for exact cleaned test rows also present in training",
    )
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    clean_root = args.output / "clean"
    clean_root.mkdir(parents=True, exist_ok=True)

    train, train_audit = clean_domain(args.train_file, "Long")
    test, test_audit = clean_domain(args.test_file, "Short")
    train_hashes = set(row_hash(train).astype("uint64").tolist())
    test_hash = row_hash(test).astype("uint64")
    overlap = test_hash.isin(train_hashes)
    overlap_rows = int(overlap.sum())
    overlap_fraction = float(overlap.mean()) if len(overlap) else 0.0
    overlap_audit = {
        "exact_test_rows_seen_in_train": overlap_rows,
        "test_rows_before_policy": int(len(test)),
        "exact_overlap_fraction": overlap_fraction,
        "policy": args.overlap_policy,
    }
    (args.output / "overlap_audit.json").write_text(
        json.dumps(overlap_audit, indent=2) + "\n", encoding="utf-8"
    )
    if overlap_rows and args.overlap_policy == "error":
        raise ValueError(
            f"{overlap_rows}/{len(test)} exact cleaned test rows occur in train; "
            "inspect overlap_audit.json, then use --overlap-policy drop only if "
            "deduplicated evaluation matches the intended protocol"
        )
    if overlap_rows and args.overlap_policy == "drop":
        test = test.loc[~overlap].copy().reset_index(drop=True)
        test["source_row"] = np.arange(len(test)) + 2
    if test.empty:
        raise ValueError("test set is empty after overlap policy")

    # Save curated feature data separately. Internal identifiers are excluded.
    train[NUMERIC + ["traffic_label"]].to_csv(
        clean_root / "long_train_clean.csv", index=False
    )
    test[NUMERIC + ["traffic_label"]].to_csv(
        clean_root / "short_test_clean.csv", index=False
    )

    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=tuple(DEFAULT_CONGESTION_FEATURES),
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()

    fixed = fit_fixed_cdr(
        train, args.window, args.seed, args.expert_trees
    )
    eligible = train.loc[fixed["source_eligible_index"]]
    rf_all = fit_rf(eligible, (), args.seed, args.rf_trees)
    rf_expert = fit_rf(eligible, TIMING, args.seed, args.rf_trees)
    legacy_meta = fit_legacy_meta(train, config)
    safe_meta = fit_safe_meta(train, config)

    fixed_prediction = predict_fixed_cdr(fixed, test)
    legacy_result = predict_legacy_meta(legacy_meta, test)
    safe_result = predict_safe_meta(safe_meta, test)
    series = {
        "CDR_MLC": fixed_prediction,
        **indexed_predictions(legacy_result, "Legacy_"),
        **indexed_predictions(safe_result, "Safe_"),
    }
    common = sorted(set.intersection(*(set(value.index) for value in series.values())))
    if not common:
        raise ValueError("no common test rows across methods")
    observed = test.loc[common].copy()
    truth = observed.traffic_label.to_numpy()
    predictions = {
        name: value.loc[common].to_numpy() for name, value in series.items()
    }
    predictions["RF_all_clean_valid"] = predict_rf(rf_all, observed)
    predictions["RF_expert_inputs"] = predict_rf(rf_expert, observed)

    rows, detailed = [], {}
    for method, prediction in predictions.items():
        score = metrics(truth, prediction, APPLICATIONS)
        detailed[method] = score
        rows.append({
            "train_domain": "Long",
            "test_domain": "Short",
            "method": method,
            **{key: score[key] for key in (
                "n", "accuracy", "balanced_accuracy",
                "macro_f1", "weighted_f1",
            )},
        })
    summary = pd.DataFrame(rows).sort_values("method")
    summary.to_csv(args.output / "long_to_short_summary.csv", index=False)
    (args.output / "long_to_short_metrics.json").write_text(
        json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
    )
    prediction_frame = observed[[
        "timestamp", "traffic_label", "congestion_level"
    ]].copy()
    for method, prediction in predictions.items():
        prediction_frame[f"prediction_{method}"] = prediction
    prediction_frame.to_csv(args.output / "long_to_short_predictions.csv",
                            index=False)

    manifest = {
        "protocol": "independent_file_cross_domain_Long_train_Short_test",
        "train": train_audit,
        "test": test_audit,
        "test_rows_after_overlap_policy": len(test),
        "overlap": overlap_audit,
        "removal_policy": REMOVAL_POLICY,
        "config": asdict(config),
        "rf_trees": args.rf_trees,
        "important_limitations": [
            "The supplied shuffled files contain no congestion-level column; "
            "this is not a Low/Medium/High scenario experiment.",
            "The supplied files contain no reliable event timestamp. Sliding "
            "windows follow stored shuffled-row order, so temporal trend claims "
            "must not be made from this experiment.",
            "Oracle-router outputs use test labels and are diagnostic only.",
        ],
    }
    (args.output / "long_to_short_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
