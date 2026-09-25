"""Shared CESNET-QUICEXT-25 class ontology and seven temporal protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MONTHS = ("2024-06", "2024-07", "2024-08")
TRANSFER_SCENARIOS = {
    "1": ("2024-06", "2024-07", "forward"),
    "2": ("2024-06", "2024-08", "forward"),
    "3": ("2024-07", "2024-06", "backward"),
    "4": ("2024-07", "2024-08", "forward"),
    "5": ("2024-08", "2024-06", "backward"),
    "6": ("2024-08", "2024-07", "backward"),
}
PROTOCOL_IDS = tuple("1234567")
CONTEXT_ALIASES = {
    "TcpRtt": "ppi_duration",
    "SynAck": "ppi_ipt_mean",
    "AckDat": "ppi_roundtrips",
}
METADATA_COLUMNS = {
    "record_id", "source_file", "source_row", "sequence_id", "period",
    "timestamp", "traffic_label", "congestion_level",
}


def load_class_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    classes = spec.get("classes", [])
    if len(classes) != 20 or len(set(classes)) != 20:
        raise ValueError("class specification must contain exactly 20 unique classes")
    if classes != sorted(classes):
        raise ValueError("class specification must use a stable lexical order")
    if spec.get("transport_context_mapping") != CONTEXT_ALIASES:
        raise ValueError("unexpected transport context mapping")
    return spec


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["timestamp", "source_file", "source_row"], kind="stable"
    ).reset_index(drop=True)


def load_months(data_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load model-ready monthly files without selecting classes or scenarios."""
    months, audit = {}, []
    required = {
        "record_id", "source_file", "source_row", "sequence_id", "period",
        "timestamp", "traffic_label", *CONTEXT_ALIASES.values(),
    }
    for month in MONTHS:
        path = data_dir / f"{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared month: {path}")
        frame = pd.read_parquet(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name}: missing required columns {missing}")
        if not frame["period"].astype(str).eq(month).all():
            raise ValueError(f"{path.name}: period values do not match {month}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame["traffic_label"] = frame["traffic_label"].astype(str)
        for alias, source in CONTEXT_ALIASES.items():
            frame[alias] = pd.to_numeric(frame[source], errors="coerce")
        # Existing MF-CDR-MLC utilities use this label only for development
        # diagnostics/balancing; it is never an inference feature.
        frame["congestion_level"] = month
        if frame["record_id"].duplicated().any():
            raise ValueError(f"{path.name}: duplicate record_id values")
        months[month] = _ordered(frame)
        audit.append({
            "period": month,
            "rows": len(frame),
            "classes_raw": int(frame.traffic_label.nunique()),
        })
    all_ids = pd.concat(
        [frame[["record_id"]] for frame in months.values()], ignore_index=True
    )
    if all_ids.record_id.duplicated().any():
        raise ValueError("record_id overlap exists across monthly files")
    return months, pd.DataFrame(audit)


def record_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["timestamp", "source_file", "source_row"], kind="stable"
    )["record_id"].astype(str)
    payload = "\n".join(ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _retain_classes(frame: pd.DataFrame, classes: tuple[str, ...]) -> pd.DataFrame:
    retained = frame.loc[frame.traffic_label.isin(classes)].copy()
    unknown = sorted(set(retained.traffic_label.unique()) - set(classes))
    if unknown:
        raise RuntimeError(f"unexpected retained labels: {unknown}")
    return _ordered(retained)


def build_protocol(
    months: dict[str, pd.DataFrame],
    scenario: str,
    classes: tuple[str, ...],
    all_development_fraction: float = .80,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build one fixed-class temporal protocol with disjoint record IDs."""
    if scenario not in PROTOCOL_IDS:
        raise ValueError(f"unknown scenario {scenario}")
    if len(classes) != 20 or len(set(classes)) != 20:
        raise ValueError("all protocols require the same 20-class ontology")
    if scenario in TRANSFER_SCENARIOS:
        source, target, direction = TRANSFER_SCENARIOS[scenario]
        development_raw = months[source]
        test_raw = months[target]
        kind = "complete-month-transfer"
    else:
        if not 0 < all_development_fraction < 1:
            raise ValueError("all_development_fraction must be in (0,1)")
        combined = _ordered(pd.concat(
            [months[month] for month in MONTHS], ignore_index=True
        ))
        cut = int(len(combined) * all_development_fraction)
        if not 0 < cut < len(combined):
            raise ValueError("empty S7 development or test partition")
        development_raw = combined.iloc[:cut].copy()
        test_raw = combined.iloc[cut:].copy()
        source, target, direction = "all-prefix", "all-tail", "forward"
        kind = "global-chronological-80-20"

    development = _retain_classes(development_raw, classes)
    test = _retain_classes(test_raw, classes)
    if development.empty or test.empty:
        raise ValueError(f"S{scenario}: empty fixed-class partition")
    train_labels = set(development.traffic_label.unique())
    test_labels = set(test.traffic_label.unique())
    expected = set(classes)
    if train_labels != expected or test_labels != expected:
        raise ValueError(
            f"S{scenario}: all 20 classes must occur in both partitions; "
            f"missing development={sorted(expected-train_labels)}, "
            f"missing test={sorted(expected-test_labels)}"
        )
    overlap = set(development.record_id) & set(test.record_id)
    if overlap:
        raise RuntimeError(f"S{scenario}: {len(overlap)} development/test overlaps")
    definition = {
        "protocol": f"S{scenario}",
        "kind": kind,
        "source": source,
        "target": target,
        "direction": direction,
        "class_count": len(classes),
        "classes": list(classes),
        "development_rows_before_class_filter": len(development_raw),
        "development_rows": len(development),
        "development_coverage": len(development) / len(development_raw),
        "test_rows_before_class_filter": len(test_raw),
        "test_rows": len(test),
        "test_coverage": len(test) / len(test_raw),
        "development_identity": record_identity(development),
        "test_identity": record_identity(test),
        "overlap_rows": 0,
    }
    return development, test, definition


def numeric_model_features(frame: pd.DataFrame) -> list[str]:
    """Return finite-capable numeric inputs, excluding compatibility aliases."""
    excluded = METADATA_COLUMNS | set(CONTEXT_ALIASES)
    features = []
    for name in frame.columns:
        if name in excluded:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            features.append(name)
    if not features:
        raise ValueError("no usable numeric model features")
    return features


def encode_labels(frame: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    encoded = frame.traffic_label.map(mapping)
    if encoded.isna().any():
        raise ValueError("frame contains a label outside the fixed ontology")
    return encoded.to_numpy(dtype=np.int64)
