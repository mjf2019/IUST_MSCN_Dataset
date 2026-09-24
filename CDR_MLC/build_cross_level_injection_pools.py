"""Build leakage-safe cross-level training pools from Clean-Valid.

For each destination congestion level, the training pool contains its own
chronological development portion plus a chronological prefix from each of the
other two levels. With the defaults this produces Low+(20% Medium,20% High),
Medium+(20% Low,20% High), and High+(20% Low,20% Medium).

The final target tail of every original capture is exported separately and is
never eligible for injection. Source files are not modified, and the physical
congestion level of every injected row is retained as metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adaptive_cdr_mlc import (  # noqa: E402
    APPLICATIONS,
    DEFAULT_CANDIDATES,
    LEVELS,
    load_dataset,
)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep capture boundaries explicit and chronology stable."""
    return frame.sort_values(
        ["sequence_id", "timestamp", "source_row"], kind="stable"
    ).reset_index(drop=True)


def row_ids(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame.source_file.astype(str), frame.source_row.astype(int)))


def identity_hash(frame: pd.DataFrame) -> str:
    identity = ordered(frame)[["source_file", "source_row"]]
    payload = identity.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def partition_captures(data: pd.DataFrame, injection_fraction: float,
                       test_fraction: float):
    """Create per-capture development, injection-prefix and fixed-test views."""
    partitions, audit = {}, []
    for sequence_id, group in data.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        development_end = int(len(group) * (1.0 - test_fraction))
        injection_end = int(len(group) * injection_fraction)
        if development_end <= 0:
            raise ValueError(f"{sequence_id}: empty development partition")
        if injection_fraction > 0 and injection_end <= 0:
            raise ValueError(f"{sequence_id}: empty nonzero injection prefix")
        if injection_end > development_end:
            raise ValueError(f"{sequence_id}: injection overlaps fixed test tail")
        development = group.iloc[:development_end].copy()
        injection = group.iloc[:injection_end].copy()
        fixed_test = group.iloc[development_end:].copy()
        partitions[sequence_id] = {
            "development": development,
            "injection": injection,
            "fixed_test": fixed_test,
        }
        audit.append({
            "sequence_id": sequence_id,
            "application": str(group.traffic_label.iloc[0]),
            "original_level": str(group.congestion_level.iloc[0]),
            "total_rows": int(len(group)),
            "development_rows": int(len(development)),
            "injection_prefix_rows": int(len(injection)),
            "fixed_test_rows": int(len(fixed_test)),
            "realized_injection_fraction": float(len(injection) / len(group)),
            "development_test_overlap": int(
                len(row_ids(development) & row_ids(fixed_test))
            ),
            "injection_test_overlap": int(
                len(row_ids(injection) & row_ids(fixed_test))
            ),
        })
    return partitions, pd.DataFrame(audit)


def annotate(frame: pd.DataFrame, destination: str, role: str) -> pd.DataFrame:
    result = frame.copy()
    # Preserve physical truth. Read one pool file directly; do not overwrite
    # congestion_level with the destination pool name.
    result["original_congestion_level"] = result.congestion_level.astype(str)
    result["pool_level"] = destination
    result["pool_role"] = role
    result["is_injected"] = role == "injected"
    return result


def build_pool(partitions: dict, destination: str):
    parts, audit = [], []
    for application in APPLICATIONS:
        for origin in LEVELS:
            sequence_id = f"{application}_{origin}"
            if sequence_id not in partitions:
                raise ValueError(f"missing capture {sequence_id}")
            role = "primary" if origin == destination else "injected"
            partition_name = "development" if role == "primary" else "injection"
            selected = annotate(
                partitions[sequence_id][partition_name], destination, role
            )
            parts.append(selected)
            audit.append({
                "pool_level": destination,
                "application": application,
                "original_level": origin,
                "role": role,
                "rows": int(len(selected)),
                "source_sequence_id": sequence_id,
            })
    pool = ordered(pd.concat(parts, ignore_index=True))
    ids = list(zip(pool.source_file.astype(str), pool.source_row.astype(int)))
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{destination}: duplicated original rows inside pool")
    return pool, audit


def build_fixed_tests(partitions: dict):
    tests = {}
    for level in LEVELS:
        parts = [
            partitions[f"{application}_{level}"]["fixed_test"]
            for application in APPLICATIONS
        ]
        test = ordered(pd.concat(parts, ignore_index=True))
        test["original_congestion_level"] = test.congestion_level.astype(str)
        test["pool_level"] = level
        test["pool_role"] = "fixed_test"
        test["is_injected"] = False
        tests[level] = test
    return tests


def run(args):
    if not 0 <= args.test_fraction < 1:
        raise ValueError("--test-fraction must be in [0,1)")
    if not 0 < args.injection_fraction <= 1 - args.test_fraction:
        raise ValueError(
            "--injection-fraction must be positive and must not overlap test"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    pool_dir = args.output / "training_pools"
    test_dir = args.output / "fixed_tests"
    pool_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    data, input_audit = load_dataset(
        args.data_dir, tuple(DEFAULT_CANDIDATES)
    )
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    partitions, capture_audit = partition_captures(
        data, args.injection_fraction, args.test_fraction
    )
    fixed_tests = build_fixed_tests(partitions)

    pool_audit, manifest_pools = [], {}
    all_fixed_test_ids = set().union(*(row_ids(test) for test in fixed_tests.values()))
    for destination in LEVELS:
        pool, rows = build_pool(partitions, destination)
        overlap = row_ids(pool) & all_fixed_test_ids
        if overlap:
            raise RuntimeError(
                f"{destination}: {len(overlap)} training rows overlap fixed tests"
            )
        path = pool_dir / f"{destination.lower()}_training_pool.csv"
        pool.to_csv(path, index=False)
        pool_audit.extend(rows)
        manifest_pools[destination] = {
            "file": str(path),
            "rows": int(len(pool)),
            "primary_rows": int((pool.pool_role == "primary").sum()),
            "injected_rows": int((pool.pool_role == "injected").sum()),
            "training_fixed_test_overlap": 0,
            "identity_sha256": identity_hash(pool),
        }

    manifest_tests = {}
    for level, test in fixed_tests.items():
        path = test_dir / f"{level.lower()}_fixed_test.csv"
        test.to_csv(path, index=False)
        manifest_tests[level] = {
            "file": str(path),
            "rows": int(len(test)),
            "identity_sha256": identity_hash(test),
        }

    capture_audit.to_csv(args.output / "capture_partition_audit.csv", index=False)
    pd.DataFrame(pool_audit).to_csv(
        args.output / "pool_composition_audit.csv", index=False
    )
    manifest = {
        "data_dir": str(args.data_dir),
        "injection_fraction": args.injection_fraction,
        "fixed_test_fraction": args.test_fraction,
        "sampling": "causal chronological prefix per application-level capture",
        "pool_definition": {
            "Low": "Low development + Medium prefix + High prefix",
            "Medium": "Medium development + Low prefix + High prefix",
            "High": "High development + Low prefix + Medium prefix",
        },
        "label_policy": (
            "congestion_level and original_congestion_level retain physical origin; "
            "pool_level identifies the destination training pool"
        ),
        "training_pools": manifest_pools,
        "fixed_tests": manifest_tests,
        "source_files_modified": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    display = pd.DataFrame([
        {
            "Pool": level,
            "PrimaryN": info["primary_rows"],
            "InjectN": info["injected_rows"],
            "TrainN": info["rows"],
            "TestN": manifest_tests[level]["rows"],
            "Overlap": info["training_fixed_test_overlap"],
        }
        for level, info in manifest_pools.items()
    ])
    print(display.to_string(index=False, justify="left"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=HERE / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "DATASETS/CDR-MLC/Cross_Level_Injection_20",
    )
    parser.add_argument("--injection-fraction", type=float, default=.20)
    parser.add_argument("--test-fraction", type=float, default=.20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
