"""Build capped, leakage-audited ISCX-Tor or ISCX-VPN CSV data.

The class is derived from each released per-application .flow filename. Classes
larger than --max-per-class are deterministically downsampled to that cap;
smaller classes are kept in full. No oversampling or synthetic rows are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL = ("Flgs", "State", "TcpOpt")
NUMERIC = (
    "Dur", "TotPkts", "SrcPkts", "DstPkts",
    "TotBytes", "SrcBytes", "DstBytes",
    "sMeanPktSz", "dMeanPktSz",
    "SrcLoad", "DstLoad", "Load",
    "SrcRate", "DstRate", "Rate",
    "SrcLoss", "DstLoss", "Loss", "pLoss",
    "SrcWin", "TcpRtt", "SynAck", "AckDat",
)
LABELS = {
    "audio": "Audio",
    "browsing": "Browsing",
    "chat": "Chat",
    "file": "FileTransfer",
    "ftp": "FTP",
    "mail": "Mail",
    "p2p": "P2P",
    "stream": "Streaming",
    "video": "Video",
    "voip": "VoIP",
}
REMOVAL_POLICY = {
    "IdleTime": "capture/time fingerprint; excluded",
    "DstWin": "known shortcut/unstable export field; excluded",
    "Label": "stale or empty exporter field; class is derived from filename",
    "metadata": "IPs, ports, timestamps, identifiers and exporter metadata excluded",
    "unused": "fields outside the predeclared common classifier schema excluded",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_indices(size: int, keep: int) -> np.ndarray:
    """Return stable, unique, order-preserving indices across the full capture."""
    if keep >= size:
        return np.arange(size, dtype=np.int64)
    return np.floor(np.arange(keep, dtype=np.float64) * size / keep).astype(np.int64)


def clean_file(path: Path) -> tuple[pd.DataFrame, dict]:
    label_key = path.stem.strip().lower()
    if label_key not in LABELS:
        raise ValueError(
            f"{path.name}: unknown class filename; supported stems: {sorted(LABELS)}"
        )
    raw = pd.read_csv(path, low_memory=False, on_bad_lines="error")
    raw.columns = raw.columns.astype(str).str.strip()
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()

    required = set(CATEGORICAL) | set(NUMERIC)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    frame = raw.loc[:, [*CATEGORICAL, *NUMERIC]].copy()
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64) + 2)
    frame.insert(0, "source_file", path.name)
    frame.insert(0, "traffic_label", LABELS[label_key])
    frame.insert(0, "sequence_id", f"{path.parent.name}::{path.stem}")

    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("", pd.NA)

    negative_counts = {}
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        # Argus can print negative zero; it is not a negative measurement.
        values = values.mask(values.abs() < 1e-15, 0.0)
        invalid = values.lt(0)
        negative_counts[column] = int(invalid.sum())
        frame[column] = values.mask(invalid, np.nan)

    audit = {
        "file": path.name,
        "class": LABELS[label_key],
        "raw_rows": int(len(raw)),
        "missing_after_cleaning": {
            column: int(frame[column].isna().sum())
            for column in (*CATEGORICAL, *NUMERIC)
        },
        "negative_values_replaced_with_missing": {
            key: value for key, value in negative_counts.items() if value
        },
        "sha256": file_sha256(path),
    }
    return frame, audit


def build(source: Path, output: Path, audit_path: Path, dataset_name: str,
          max_per_class: int) -> None:
    if max_per_class < 1:
        raise ValueError("--max-per-class must be positive")
    files = sorted(source.glob("*.flow"))
    if not files:
        raise FileNotFoundError(f"no .flow files found under {source}")

    cleaned, file_audits = [], []
    for path in files:
        frame, file_audit = clean_file(path)
        cleaned.append(frame)
        file_audits.append(file_audit)

    combined = pd.concat(cleaned, ignore_index=True)
    sampled, sampling_audit = [], {}
    for label, group in combined.groupby("traffic_label", sort=True):
        group = group.sort_values(["source_file", "source_row"], kind="stable")
        raw_count = int(len(group))
        keep = min(raw_count, max_per_class)
        selected = group.iloc[evenly_spaced_indices(raw_count, keep)].copy()
        sampled.append(selected)
        sampling_audit[str(label)] = {
            "raw": raw_count,
            "retained": keep,
            "capped": bool(raw_count > max_per_class),
        }

    result = pd.concat(sampled, ignore_index=True)
    result = result.sort_values(
        ["traffic_label", "source_file", "source_row"], kind="stable"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    audit = {
        "dataset": dataset_name,
        "source": str(source),
        "output": str(output),
        "sampling_policy": (
            f"retain every row for classes with <= {max_per_class} rows; "
            f"deterministically cap larger classes at {max_per_class}; "
            "no oversampling and no synthetic samples"
        ),
        "max_per_class": max_per_class,
        "raw_rows": int(len(combined)),
        "retained_rows": int(len(result)),
        "raw_class_counts": {
            str(k): int(v)
            for k, v in combined.traffic_label.value_counts().sort_index().items()
        },
        "retained_class_counts": {
            str(k): int(v)
            for k, v in result.traffic_label.value_counts().sort_index().items()
        },
        "class_sampling": sampling_audit,
        "retained_columns": result.columns.tolist(),
        "removal_policy": REMOVAL_POLICY,
        "input_files": file_audits,
        "output_sha256": file_sha256(output),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "dataset": dataset_name,
        "raw_rows": len(combined),
        "retained_rows": len(result),
        "class_counts": audit["retained_class_counts"],
        "output": str(output),
        "audit": str(audit_path),
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--max-per-class", type=int, default=1000)
    args = parser.parse_args()
    if args.audit is None:
        args.audit = args.output.with_name(args.output.stem + "_audit.json")
    return args


def main() -> None:
    args = parse_args()
    build(
        args.source, args.output, args.audit,
        args.dataset_name, args.max_per_class,
    )


if __name__ == "__main__":
    main()
