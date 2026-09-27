"""Build a balanced ten-class SDNCampus + encrypted-protocol dataset.

The six application classes in ``SDNCampus_original.csv`` are augmented with
four explicitly encrypted protocol captures: IPsec, L2TP-VPN, OpenVPN, and
SSH.  The output keeps only the CICFlowMeter columns shared by every input.
Each class is deterministically reduced to the smallest class size using
evenly spaced records in original capture order; no synthetic rows, duplicate
oversampling, or test-aware selection is used.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_ALIASES = ("label", "class", "application", "app")
ENCRYPTED_FILES = {
    "ipsec.csv": "IPsec",
    "l2tpvpn.csv": "L2TP-VPN",
    "openvpn.csv": "OpenVPN",
    "ssh.csv": "SSH",
}
REQUIRED_CONTEXT_KEYS = ("flowiatmean", "fwdiatmean", "bwdiatmean")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _header(path: Path):
    columns = pd.read_csv(path, nrows=0).columns.astype(str).str.strip().tolist()
    mapping = {}
    for column in columns:
        key = _key(column)
        if key in mapping:
            raise ValueError(
                f"{path.name}: columns {mapping[key]!r} and {column!r} "
                f"normalize to the same key {key!r}"
            )
        mapping[key] = column
    return columns, mapping


def _label_column(mapping, path):
    for alias in LABEL_ALIASES:
        if alias in mapping:
            return mapping[alias]
    raise ValueError(f"{path.name}: no application-label column found")


def _count_rows(path: Path, probe_column: str) -> int:
    return int(sum(
        len(chunk) for chunk in pd.read_csv(
            path, usecols=[probe_column], chunksize=100_000,
            low_memory=False, on_bad_lines="error",
        )
    ))


def _positions(length: int, target: int) -> np.ndarray:
    if target > length or target < 1:
        raise ValueError(f"invalid balanced sample size {target} for {length} rows")
    if target == length:
        return np.arange(length, dtype=np.int64)
    positions = np.floor(
        (np.arange(target, dtype=float) + .5) * length / target
    ).astype(np.int64)
    if len(np.unique(positions)) != target:
        raise RuntimeError("deterministic balanced sampler produced duplicates")
    return positions


def _read_features(path, mapping, shared_keys, canonical_columns):
    source_columns = [mapping[key] for key in shared_keys]
    frame = pd.read_csv(
        path, usecols=source_columns, low_memory=False, on_bad_lines="error"
    )
    rename = {
        mapping[key]: canonical_columns[key] for key in shared_keys
    }
    return frame.rename(columns=rename)[
        [canonical_columns[key] for key in shared_keys]
    ]


def main() -> None:
    root = Path(__file__).resolve().parent
    default_dir = root / "DATASETS/SDNCAMPUS"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument(
        "--output", type=Path,
        default=default_dir / "SDNCampus_encrypted_balanced.csv",
    )
    parser.add_argument(
        "--audit", type=Path,
        default=default_dir / "SDNCampus_encrypted_balanced_audit.json",
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=None,
        help="Optional cap; by default the smallest observed class is used.",
    )
    args = parser.parse_args()

    original = args.input_dir / "SDNCampus_original.csv"
    encrypted = {
        args.input_dir / filename: label
        for filename, label in ENCRYPTED_FILES.items()
    }
    inputs = [original, *encrypted]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SDNCampus inputs: {missing}")

    headers, mappings = {}, {}
    for path in inputs:
        headers[path], mappings[path] = _header(path)
    label_column = _label_column(mappings[original], original)

    # Label names are not features. A label-like column in an encrypted input
    # is also excluded even though its class is intentionally set by filename.
    feature_sets = []
    for path in inputs:
        label_keys = {alias for alias in LABEL_ALIASES if alias in mappings[path]}
        feature_sets.append(set(mappings[path]) - label_keys)
    shared_keys = set.intersection(*feature_sets)
    absent = set(REQUIRED_CONTEXT_KEYS) - shared_keys
    if absent:
        raise ValueError(
            "shared CICFlowMeter schema lacks required context fields: "
            f"{sorted(absent)}"
        )
    # Preserve the original SDNCampus column order in the integrated output.
    shared_keys = [
        _key(column) for column in headers[original]
        if _key(column) in shared_keys
    ]
    canonical_columns = {
        key: mappings[original][key] for key in shared_keys
    }

    original_labels = pd.read_csv(
        original, usecols=[label_column], low_memory=False,
        on_bad_lines="error",
    )[label_column].astype("string").str.strip()
    if original_labels.isna().any() or original_labels.eq("").any():
        raise ValueError("original SDNCampus contains empty labels")
    original_counts = {
        str(label): int(count)
        for label, count in original_labels.value_counts(sort=False).items()
    }
    encrypted_counts = {}
    for path, label in encrypted.items():
        probe = mappings[path][shared_keys[0]]
        encrypted_counts[label] = _count_rows(path, probe)
    counts = {**original_counts, **encrypted_counts}
    if len(counts) != len(original_counts) + len(encrypted_counts):
        raise ValueError("an encrypted protocol label collides with an original class")
    target = min(counts.values())
    if args.samples_per_class is not None:
        if args.samples_per_class < 1:
            parser.error("--samples-per-class must be positive")
        target = min(target, args.samples_per_class)

    feature_names = [canonical_columns[key] for key in shared_keys]
    output_parts = []
    original_features = _read_features(
        original, mappings[original], shared_keys, canonical_columns
    )
    for label in original_counts:
        indices = np.flatnonzero(original_labels.to_numpy() == label)
        selected = indices[_positions(len(indices), target)]
        part = original_features.iloc[selected].copy()
        part["Label"] = label
        output_parts.append(part)
    del original_features

    for path, label in encrypted.items():
        features = _read_features(
            path, mappings[path], shared_keys, canonical_columns
        )
        part = features.iloc[_positions(len(features), target)].copy()
        part["Label"] = label
        output_parts.append(part)
        del features

    combined = pd.concat(output_parts, ignore_index=True)
    expected_rows = target * len(counts)
    if len(combined) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, observed {len(combined)}")
    observed = combined.Label.value_counts().to_dict()
    if set(observed.values()) != {target}:
        raise RuntimeError(f"output is not balanced: {observed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    audit = {
        "dataset": "SDNCampus + explicitly encrypted protocols",
        "original_classes": original_counts,
        "encrypted_protocol_classes": encrypted_counts,
        "rows_before_balancing": counts,
        "samples_per_class": target,
        "classes": list(counts),
        "class_count": len(counts),
        "output_rows": len(combined),
        "balanced": True,
        "sampling": "deterministic evenly spaced records in original order",
        "oversampling": False,
        "synthetic_rows": False,
        "shared_feature_count": len(feature_names),
        "shared_features": feature_names,
        "output": str(args.output),
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
