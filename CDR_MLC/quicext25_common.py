"""Shared CESNET-QUICEXT-25 class ontology and seven temporal protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.feature_selection import mutual_info_classif


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
CONTEXT_NAMES = tuple(CONTEXT_ALIASES)
SHORTCUT_TOKENS = {
    "id", "record_id", "source_row", "timestamp", "time_first", "time_last",
    "period", "month", "day", "label", "class", "sni", "hostname",
    "domain", "user_agent", "src_ip", "dst_ip", "src_port", "dst_port",
    "sport", "dport", "quic_version", "tls_version", "protocol",
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


def apply_context_aliases(
    frame: pd.DataFrame, mapping: dict[str, str]
) -> pd.DataFrame:
    """Expose three selected QUIC measurements through the TCP-era API names."""
    if tuple(mapping) != CONTEXT_NAMES or len(set(mapping.values())) != 3:
        raise ValueError("context mapping must assign three distinct source features")
    missing = sorted(set(mapping.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"context mapping refers to missing columns: {missing}")
    result = frame.copy()
    for alias, source in mapping.items():
        result[alias] = pd.to_numeric(result[source], errors="coerce")
    return result


def _is_shortcut_feature(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in SHORTCUT_TOKENS or lowered.endswith("_id")


def _rank01(values: pd.Series) -> pd.Series:
    if len(values) == 1:
        return pd.Series(1.0, index=values.index)
    return values.rank(method="average", pct=True)


def select_transport_context_features(
    development: pd.DataFrame,
    *,
    n_features: int = 3,
    temporal_blocks: int = 4,
    max_rows: int = 50_000,
    min_valid_fraction: float = .95,
    random_state: int = 42,
) -> tuple[dict[str, str], pd.DataFrame, dict]:
    """Select source-only temporal proxies without observing the target period.

    Temporal sensitivity is the class-conditional Wasserstein distance between
    adjacent chronological blocks, normalized by the feature IQR.  It is
    combined with development-label mutual information.  Greedy selection then
    penalizes Spearman redundancy with features already selected.
    """
    if n_features != len(CONTEXT_NAMES):
        raise ValueError(f"exactly {len(CONTEXT_NAMES)} context features are required")
    if temporal_blocks < 3:
        raise ValueError("temporal_blocks must be at least 3")
    if max_rows < 100:
        raise ValueError("max_rows must be at least 100")
    required = {"timestamp", "traffic_label"}
    missing = sorted(required - set(development.columns))
    if missing:
        raise ValueError(f"development data missing selection columns: {missing}")

    ordered = _ordered(development)
    candidates = [
        name for name in numeric_model_features(ordered)
        if not _is_shortcut_feature(name)
    ]
    audit_rows, usable = [], []
    numeric_cache = {}
    for name in candidates:
        values = pd.to_numeric(ordered[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        numeric_cache[name] = values
        valid_fraction = float(values.notna().mean())
        negative_fraction = float((values.dropna() < 0).mean()) if values.notna().any() else 1.0
        unique = int(values.nunique(dropna=True))
        eligible = (
            valid_fraction >= min_valid_fraction
            and negative_fraction == 0.0
            and unique > 1
        )
        audit_rows.append({
            "feature": name,
            "eligible": bool(eligible),
            "exclusion_reason": "" if eligible else (
                "low_valid_fraction" if valid_fraction < min_valid_fraction else
                "negative_values" if negative_fraction > 0 else "constant"
            ),
            "valid_fraction": valid_fraction,
            "negative_fraction": negative_fraction,
            "unique_values": unique,
        })
        if eligible:
            usable.append(name)
    if len(usable) < n_features:
        raise ValueError(f"only {len(usable)} eligible temporal proxy features")

    # Deterministic cap for MI and correlation cost.  Chronology is retained in
    # the full frame used by the temporal sensitivity calculation below.
    if len(ordered) > max_rows:
        sample = ordered.sample(n=max_rows, random_state=random_state).sort_index()
    else:
        sample = ordered
    x = pd.DataFrame({
        name: pd.to_numeric(sample[name], errors="coerce") for name in usable
    }).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    y = sample.traffic_label.astype("category").cat.codes.to_numpy()
    mi = mutual_info_classif(
        x.to_numpy(dtype=float), y, discrete_features=False,
        random_state=random_state,
    )

    # Global chronological blocks, followed by within-class comparisons, avoid
    # confusing changes in application prevalence with feature drift.
    block = np.minimum(
        np.floor(np.arange(len(ordered)) * temporal_blocks / max(len(ordered), 1)),
        temporal_blocks - 1,
    ).astype(int)
    labels = ordered.traffic_label.astype(str).to_numpy()
    sensitivity = {}
    for name in usable:
        values = numeric_cache[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        q25, q75 = np.nanquantile(values, [.25, .75])
        scale = max(float(q75 - q25), 1e-12)
        distances, weights = [], []
        for label in np.unique(labels):
            label_mask = labels == label
            for left in range(temporal_blocks - 1):
                a = values[finite & label_mask & (block == left)]
                b = values[finite & label_mask & (block == left + 1)]
                if len(a) < 5 or len(b) < 5:
                    continue
                weight = min(len(a), len(b))
                distances.append(wasserstein_distance(a, b) / scale)
                weights.append(weight)
        sensitivity[name] = float(np.average(distances, weights=weights)) if weights else 0.0

    ranking = pd.DataFrame(audit_rows).set_index("feature")
    ranking["temporal_sensitivity"] = pd.Series(sensitivity)
    ranking["mutual_information"] = pd.Series(dict(zip(usable, mi)))
    eligible_index = ranking.index[ranking.eligible]
    ranking.loc[eligible_index, "temporal_rank"] = _rank01(
        ranking.loc[eligible_index, "temporal_sensitivity"]
    )
    ranking.loc[eligible_index, "mi_rank"] = _rank01(
        ranking.loc[eligible_index, "mutual_information"]
    )
    ranking["base_score"] = ranking.temporal_rank * ranking.mi_rank

    correlations = x[usable].corr(method="spearman").abs().fillna(0.0)
    selected = []
    remaining = set(usable)
    selection_details = {}
    while len(selected) < n_features:
        best_name, best_key, best_detail = None, None, None
        for name in sorted(remaining):
            redundancy = (
                float(correlations.loc[name, selected].max()) if selected else 0.0
            )
            adjusted = float(ranking.loc[name, "base_score"]) * (1.0 - redundancy)
            key = (adjusted, float(ranking.loc[name, "base_score"]), name)
            if best_key is None or key > best_key:
                best_name, best_key = name, key
                best_detail = (redundancy, adjusted)
        selected.append(best_name)
        remaining.remove(best_name)
        selection_details[best_name] = {
            "selection_order": len(selected),
            "max_abs_spearman_to_previous": best_detail[0],
            "adjusted_score": best_detail[1],
        }

    ranking["selected"] = ranking.index.isin(selected)
    ranking["selection_order"] = pd.Series({
        name: detail["selection_order"] for name, detail in selection_details.items()
    })
    ranking["max_abs_spearman_to_previous"] = pd.Series({
        name: detail["max_abs_spearman_to_previous"]
        for name, detail in selection_details.items()
    })
    ranking["adjusted_score"] = pd.Series({
        name: detail["adjusted_score"] for name, detail in selection_details.items()
    })
    ranking = ranking.reset_index().sort_values(
        ["selected", "selection_order", "base_score"],
        ascending=[False, True, False], kind="stable",
    )
    mapping = dict(zip(CONTEXT_NAMES, selected))
    audit = {
        "selection_scope": "development_only",
        "target_rows_observed": 0,
        "objective": "temporal_rank * mutual_information_rank * redundancy_penalty",
        "temporal_measure": "class-conditional adjacent-block normalized Wasserstein distance",
        "temporal_blocks": temporal_blocks,
        "mi_rows": len(sample),
        "min_valid_fraction": min_valid_fraction,
        "random_state": random_state,
        "selected_features": selected,
        "context_mapping": mapping,
    }
    return mapping, ranking, audit


def encode_labels(frame: pd.DataFrame, classes: tuple[str, ...]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(classes)}
    encoded = frame.traffic_label.map(mapping)
    if encoded.isna().any():
        raise ValueError("frame contains a label outside the fixed ontology")
    return encoded.to_numpy(dtype=np.int64)
