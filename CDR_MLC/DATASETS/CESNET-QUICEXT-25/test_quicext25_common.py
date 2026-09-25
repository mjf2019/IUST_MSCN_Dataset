from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quicext25_common import (  # noqa: E402
    MONTHS, PROTOCOL_IDS, build_protocol, load_class_spec,
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
