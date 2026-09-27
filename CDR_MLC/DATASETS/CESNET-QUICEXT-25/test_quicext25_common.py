from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quicext25_common import (  # noqa: E402
    MONTHS, PROTOCOL_IDS, apply_context_aliases, build_protocol,
    load_class_spec, select_transport_context_features,
)


def synthetic_months(classes):
    months = {}
    for month_number, month in enumerate(MONTHS, start=6):
        rows = []
        for cycle in range(10):
            for class_number, label in enumerate(classes):
                source_row = cycle * len(classes) + class_number
                rows.append({
                    "record_id": f"{month}:{source_row}",
                    "source_file": f"{month}.parquet",
                    "source_row": source_row,
                    "sequence_id": f"{month}:day",
                    "period": month,
                    "timestamp": (
                        pd.Timestamp(f"2024-{month_number:02d}-01", tz="UTC")
                        + pd.Timedelta(seconds=source_row)
                    ),
                    "traffic_label": label,
                })
        months[month] = pd.DataFrame(rows)
    return months


def test_all_protocols_use_identical_classes_and_disjoint_records():
    spec_path = Path(__file__).with_name("quicext25_classes.json")
    classes = tuple(load_class_spec(spec_path)["classes"])
    months = synthetic_months(classes)
    for scenario in PROTOCOL_IDS:
        development, test, audit = build_protocol(months, scenario, classes)
        assert set(development.traffic_label) == set(classes)
        assert set(test.traffic_label) == set(classes)
        assert not (set(development.record_id) & set(test.record_id))
        assert audit["class_count"] == 20
        assert audit["overlap_rows"] == 0


def test_context_selection_is_development_only_and_deterministic():
    spec_path = Path(__file__).with_name("quicext25_classes.json")
    classes = tuple(load_class_spec(spec_path)["classes"])
    months = synthetic_months(classes)
    development, target, _ = build_protocol(months, "2", classes)
    for feature_number in range(6):
        name = f"feature_{feature_number}"
        development[name] = (
            development.source_row * (feature_number + 1)
            + development.traffic_label.map({x: i for i, x in enumerate(classes)})
        ).astype(float)
        target[name] = -999_999.0 - feature_number
    first, ranking, audit = select_transport_context_features(
        development, max_rows=len(development), random_state=42
    )
    second, _, _ = select_transport_context_features(
        development, max_rows=len(development), random_state=42
    )
    assert first == second
    assert len(set(first.values())) == 3
    assert audit["target_rows_observed"] == 0
    assert ranking.selected.sum() == 3
    aliased = apply_context_aliases(development, first)
    for alias, source in first.items():
        assert aliased[alias].equals(pd.to_numeric(development[source]))
