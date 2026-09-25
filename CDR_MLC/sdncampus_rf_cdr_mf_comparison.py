"""Leakage-safe 80/20 SDNCampus comparison of RF, CDR-MLC, and MF-CDR-MLC.

SDNCampus is a CICFlowMeter dataset and does not expose the Argus fields
TcpRtt/SynAck/AckDat used by IUST-MSCN.  This adapter therefore uses three
*predeclared* CICFlow timing measurements as router-only proxies.  It neither
requires nor constructs Low/Medium/High congestion labels.

Every application was captured separately in the source study.  Because the
released table has no five-tuple/capture identifier, its application label is
used only to reconstruct those six capture sequences and to create an 80/20
within-capture chronological split.  Labels, sequence IDs, timestamps, IPs,
ports, and identifiers are never classifier features.

The script intentionally does not tune features or hyperparameters on test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

import adaptive_cdr_mlc as adaptive_module
import compare_clean_valid as comparison_module
import congestion_feature_cdr_mlc as congestion_module
import learned_router_cdr_mlc as learned_module
import meta_stacked_cdr_mlc_leakage_safe as meta_module
from benchmarks.console_output import print_compact_results
from compare_clean_valid import fit_rf, metrics, predict_rf
from meta_stacked_cdr_mlc_leakage_safe import (
    MetaStackConfig,
    fit_meta_stacker,
    predict_all,
)


METHODS = ("RF_Clean_Valid", "Original_CDR_MLC", "MF_CDR_MLC")
PAPER_TIMING = ("TcpRtt", "SynAck", "AckDat")
CONTEXT_ALIASES = {
    "TcpRtt": ("flowiatmean",),
    "SynAck": ("fwdiatmean", "forwardiatmean"),
    "AckDat": ("bwdiatmean", "backwardiatmean"),
}
LABEL_ALIASES = ("label", "class", "application", "app")
TIME_ALIASES = ("timestamp", "flowstarttime", "starttime")
IDENTIFIER_KEYS = {
    "flowid", "srcip", "sourceip", "dstip", "destinationip",
    "srcport", "sourceport", "dstport", "destinationport",
    "timestamp", "flowstarttime", "starttime", "simillarhttp",
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _column_map(frame: pd.DataFrame) -> dict[str, str]:
    result = {}
    for column in frame.columns:
        key = _key(column)
        if key and key not in result:
            result[key] = column
    return result


def _resolve(mapping: dict[str, str], aliases, purpose: str) -> str:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    raise ValueError(
        f"SDNCampus is missing {purpose}; accepted normalized names: {aliases}"
    )


def default_data_path(root: Path) -> Path:
    candidates = (
        root / "DATASETS/SDNCampus/SDNCampus_original.csv",
        root.parent / "AMCAL/SDNCampus_TEST/Dataset/SDNCampus_original.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[1]


def _parse_time(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return parsed


def load_sdncampus(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False, on_bad_lines="error")
    raw.columns = raw.columns.astype(str).str.strip()
    mapping = _column_map(raw)
    label_column = _resolve(mapping, LABEL_ALIASES, "application label")
    time_column = _resolve(mapping, TIME_ALIASES, "timestamp")
    proxy_sources = {
        target: _resolve(mapping, aliases, f"context proxy for {target}")
        for target, aliases in CONTEXT_ALIASES.items()
    }

    frame = raw.copy()
    frame["traffic_label"] = (
        frame[label_column].astype("string").str.strip().str.replace(
            r"\s+", "_", regex=True
        )
    )
    if frame.traffic_label.isna().any() or frame.traffic_label.eq("").any():
        raise ValueError("empty SDNCampus application labels")
    frame["source_row"] = np.arange(len(frame), dtype=np.int64) + 2
    parsed = _parse_time(frame[time_column])
    if parsed.notna().sum() != len(frame):
        bad = int(parsed.isna().sum())
        raise ValueError(f"{bad} SDNCampus timestamps could not be parsed")
    frame["timestamp"] = parsed
    frame["source_file"] = path.name
    frame["sequence_id"] = "SDNCampus::" + frame.traffic_label.astype(str)

    for target, source in proxy_sources.items():
        frame[target] = pd.to_numeric(frame[source], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    # Remove duplicate proxy sources and any identifier/label fields before
    # select_classifier_columns sees the table.  Protocol is retained because
    # it is an observed transport attribute rather than an endpoint identity.
    protected = {
        "traffic_label", "source_row", "timestamp", "source_file",
        "sequence_id", *PAPER_TIMING,
    }
    drop = []
    proxy_source_set = set(proxy_sources.values())
    for column in frame.columns:
        if column in protected:
            continue
        if (
            column == label_column
            or column == time_column
            or column in proxy_source_set
            or _key(column) in IDENTIFIER_KEYS
            or _key(column).startswith("unnamed")
        ):
            drop.append(column)
    frame = frame.drop(columns=drop)

    # The current shared MF implementation expects a congestion_level column
    # only for optional level-aware balancing/selection diagnostics.  A single
    # constant sentinel disables that distinction without injecting any
    # congestion information.  The column is forbidden as a model feature.
    frame["congestion_level"] = "Unlabeled"

    metadata = {
        "input": str(path),
        "raw_rows": int(len(raw)),
        "retained_rows": int(len(frame)),
        "classes": sorted(frame.traffic_label.unique().tolist()),
        "label_column": label_column,
        "timestamp_column": time_column,
        "context_proxy_mapping": proxy_sources,
        "dropped_identifier_or_duplicate_columns": sorted(drop),
        "congestion_labels_available": False,
        "congestion_levels_constructed": False,
        "constant_api_sentinel": "Unlabeled",
    }
    return frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    ).reset_index(drop=True), metadata


def split_80_20(frame: pd.DataFrame, train_fraction: float):
    train, test, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * train_fraction)
        if not 4 <= cut < len(group):
            raise ValueError(f"{sequence_id}: too short for an 80/20 split")
        train.append(group.iloc[:cut].copy())
        test.append(group.iloc[cut:].copy())
        audit.append({
            "sequence_id": sequence_id,
            "label": str(group.traffic_label.iloc[0]),
            "total": int(len(group)),
            "train": int(cut),
            "test": int(len(group) - cut),
            "train_last_time": group.iloc[cut - 1].timestamp.isoformat(),
            "test_first_time": group.iloc[cut].timestamp.isoformat(),
        })
    return (
        pd.concat(train).sort_index(kind="stable"),
        pd.concat(test).sort_index(kind="stable"),
        audit,
    )


def _set_application_ontology(labels: list[str]) -> None:
    """Make the existing generic pipeline use this dataset's fixed ontology."""
    ontology = list(labels)
    adaptive_module.APPLICATIONS = ontology
    comparison_module.APPLICATIONS = ontology
    learned_module.APPLICATIONS = ontology
    congestion_module.APPLICATIONS = ontology
    meta_module.APPLICATIONS = ontology


def _identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    )[["sequence_id", "source_file", "source_row"]]
    return hashlib.sha256(
        ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _timed(function, *args, **kwargs):
    start = perf_counter()
    value = function(*args, **kwargs)
    return value, perf_counter() - start


def evaluate(train, test, config, rf_trees, labels):
    mf_model, mf_fit = _timed(fit_meta_stacker, train, config)
    mf_result, mf_predict = _timed(predict_all, mf_model, test)
    observed = mf_result["observed"]
    truth = observed.traffic_label.to_numpy()

    eligible_train = train.loc[mf_model["source_eligible_index"]]
    rf_model, rf_fit = _timed(
        fit_rf, eligible_train, (), config.random_state, rf_trees
    )
    rf_prediction, rf_predict_time = _timed(predict_rf, rf_model, observed)

    predictions = {
        "RF_Clean_Valid": rf_prediction,
        "Original_CDR_MLC": mf_result["CDR_MLC_actual_router"],
        "MF_CDR_MLC": mf_result["CDR_MLC_meta_stacker"],
    }
    timing = {
        "RF_Clean_Valid": (rf_fit, rf_predict_time),
        # The original branch is embedded in MF.  Its standalone cost is not
        # silently estimated from the joint fit.
        "Original_CDR_MLC": (np.nan, np.nan),
        "MF_CDR_MLC": (mf_fit, mf_predict),
    }
    rows, detail = [], {}
    for method in METHODS:
        result = metrics(truth, predictions[method], labels)
        fit_seconds, predict_seconds = timing[method]
        rows.append({
            "protocol": "SDNCampus-80-20",
            "seed": config.random_state,
            "method": method,
            **{key: result[key] for key in (
                "n", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"
            )},
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        })
        detail[method] = result
    audit = {
        "evaluated_rows": int(len(observed)),
        "evaluated_test_identity": _identity(observed),
        "same_test_rows_for_all_methods": True,
        "partition_rows": mf_model["partition_rows"],
        "leakage_control": mf_model["leakage_control"],
        "selected_meta_variant": mf_model["selected_meta_variant"],
        "selected_meta_confidence": mf_model["selected_meta_confidence"],
    }
    return pd.DataFrame(rows), detail, audit


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path(root))
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs/sdncampus_rf_cdr_mf_80_20",
    )
    parser.add_argument("--train-fraction", type=float, default=.80)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=10)
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isclose(args.train_fraction, .80):
        parser.error("this confirmatory runner requires --train-fraction 0.80")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_sdncampus(args.data)
    labels = sorted(data.traffic_label.unique().tolist())
    if len(labels) < 2:
        raise ValueError("SDNCampus must contain at least two application classes")
    _set_application_ontology(labels)

    train, test, split_audit = split_80_20(data, args.train_fraction)

    config = MetaStackConfig(
        window=args.window,
        congestion_window=args.congestion_window,
        congestion_features=PAPER_TIMING,
        expert_trees=args.expert_trees,
        utility_trees=args.utility_trees,
        meta_trees=args.meta_trees,
        random_state=args.seed,
    ).validate()
    results, detailed, model_audit = evaluate(
        train, test, config, args.rf_trees, labels
    )
    results.to_csv(args.output / "sdncampus_results.csv", index=False)
    (args.output / "sdncampus_detailed_metrics.json").write_text(
        json.dumps(detailed, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": "SDNCampus",
        "protocol": "per-capture chronological 80% train / 20% test",
        "input_audit": input_audit,
        "split_audit": split_audit,
        "train_identity": _identity(train),
        "test_identity": _identity(test),
        "classes": labels,
        "context_proxy_features": list(PAPER_TIMING),
        "congestion_labels_available": False,
        "congestion_levels_constructed": False,
        "level_aware_balancing_disabled_by_single_constant_sentinel": True,
        "test_used_for_training_selection_or_preprocessing": False,
        "model_config": asdict(config),
        "rf_trees": args.rf_trees,
        "model_audit": model_audit,
    }
    (args.output / "sdncampus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(results)


if __name__ == "__main__":
    main()
