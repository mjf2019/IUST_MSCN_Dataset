"""Convert the legacy scale_1/Short dataset to the reviewed Clean-Valid schema.

The three legacy level CSV files are split into five application captures per
level.  Raw files are never modified.  Only fields accepted by the revised
Clean-Valid policy are retained.  Synthetic metadata exists solely to satisfy
the common loader and is forbidden as a classifier input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


APPLICATIONS = ["HTTP", "SFTP", "SMTP", "SSH", "Video"]
LEVEL_FILES = {
    "Low": "level_1.csv",
    "Medium": "level_2.csv",
    "High": "level_3.csv",
}
SERVICES = {
    "HTTP": ("192.168.2.122", 8080),
    "SFTP": ("192.168.2.120", 22),
    "SMTP": ("192.168.2.120", 8025),
    "SSH": ("192.168.2.120", 22),
    "Video": ("192.168.2.121", 5000),
}
CLIENT = "192.168.1.111"

# Intersection of reviewed Clean-Valid numeric fields and the legacy schema.
NUMERIC = [
    "Dur", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes",
    "DstBytes", "dMeanPktSz", "SrcLoad", "DstLoad", "Load", "SrcRate",
    "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", "pLoss", "SrcWin",
    "DstWin", "TcpRtt", "SynAck", "AckDat",
]
METADATA = ["StartTime", "SrcAddr", "DstAddr", "Proto", "Sport", "Dport"]
OUTPUT_COLUMNS = METADATA + NUMERIC

# Same decisions used for the revised Clean-Valid data. Some revised-only
# columns are listed for audit completeness even though they do not occur here.
REMOVAL_POLICY = {
    "capture_fingerprint_or_unresolved_semantics": ["IdleTime"],
    "constant_or_protocol_fingerprint": [
        "pRetran", "SrcRetra", "PCRatio", "dTtl", "sTtl", "DstRetra",
        "StdDev",
    ],
    "invalid_or_effectively_empty_gap": ["SrcGap", "DstGap"],
    "redundant_duration_alias_keep_Dur": ["Sum", "Min", "Mean", "Max", "Dur.1"],
    "labels_or_export_metadata": ["label", "Label", "Cause", "Dir", "Proto.1"],
    "unavailable_revised_clean_fields": ["sMeanPktSz", "Flgs", "State", "TcpOpt"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_label(value: object) -> str:
    key = str(value).strip().lower()
    mapping = {
        "http": "HTTP", "sftp": "SFTP", "smtp": "SMTP",
        "ssh": "SSH", "video": "Video",
    }
    if key not in mapping:
        raise ValueError(f"unknown legacy application label: {value!r}")
    return mapping[key]


def clean_level(path: Path, level: str) -> tuple[dict[str, pd.DataFrame], dict]:
    raw = pd.read_csv(path, low_memory=False, on_bad_lines="error")
    raw.columns = raw.columns.str.strip()
    label_column = "label" if "label" in raw.columns else "Label"
    if label_column not in raw.columns:
        raise ValueError(f"{path}: missing application label column")
    missing = sorted(set(NUMERIC) - set(raw.columns))
    if missing:
        raise ValueError(f"{path}: missing reviewed legacy fields {missing}")

    labels = raw[label_column].map(normalize_label)
    captures: dict[str, pd.DataFrame] = {}
    capture_audit = []
    level_offset = {"Low": 0, "Medium": 1, "High": 2}[level]
    base_time = pd.Timestamp("2024-01-01") + pd.Timedelta(days=level_offset)

    for application in APPLICATIONS:
        positions = np.flatnonzero(labels.eq(application).to_numpy())
        if not len(positions):
            raise ValueError(f"{path}: no rows for {application}")
        selected = raw.iloc[positions].copy()
        clean = pd.DataFrame(index=np.arange(len(selected)))
        # StartTime is a monotonic ordering surrogate only. It is excluded from
        # every model by adaptive_cdr_mlc.FORBIDDEN.
        timestamps = base_time + pd.to_timedelta(np.arange(len(selected)), unit="ms")
        clean["StartTime"] = timestamps.strftime("%Y/%m/%d %H:%M:%S.%f")
        server, port = SERVICES[application]
        clean["SrcAddr"] = CLIENT
        clean["DstAddr"] = server
        clean["Proto"] = "tcp"
        clean["Sport"] = 40000
        clean["Dport"] = port

        invalid_counts = {}
        for column in NUMERIC:
            values = pd.to_numeric(selected[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            negative = values.lt(0)
            if negative.any():
                raise ValueError(
                    f"{path.name}/{application}: {column} has "
                    f"{int(negative.sum())} negative values"
                )
            clean[column] = values.to_numpy()
            invalid_counts[column] = int(values.isna().sum())

        captures[application] = clean[OUTPUT_COLUMNS]
        capture_audit.append({
            "application": application,
            "level": level,
            "source_rows": int(len(selected)),
            "first_source_row": int(positions[0] + 2),
            "last_source_row": int(positions[-1] + 2),
            "missing_by_retained_numeric": invalid_counts,
        })

    return captures, {
        "file": path.name,
        "level": level,
        "raw_rows": int(len(raw)),
        "source_sha256": sha256(path),
        "source_columns": list(raw.columns),
        "captures": capture_audit,
    }


def build(source: Path, output: Path, overwrite: bool = False) -> dict:
    expected_sources = {name for name in LEVEL_FILES.values()}
    observed_sources = {path.name for path in source.glob("level_*.csv")}
    if observed_sources != expected_sources:
        raise ValueError(
            f"expected {sorted(expected_sources)}; "
            f"missing={sorted(expected_sources-observed_sources)}, "
            f"unexpected={sorted(observed_sources-expected_sources)}"
        )

    expected_outputs = {
        f"{application}_{level}.flow"
        for application in APPLICATIONS for level in LEVEL_FILES
    }
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in expected_outputs if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{len(existing)} legacy Clean-Valid captures already exist; "
            "use --overwrite explicitly"
        )

    audits = []
    output_audits = []
    for level, filename in LEVEL_FILES.items():
        captures, audit = clean_level(source / filename, level)
        audits.append(audit)
        for application, clean in captures.items():
            name = f"{application}_{level}.flow"
            destination = output / name
            temporary = output / f"{name}.tmp"
            clean.to_csv(temporary, index=False)
            temporary.replace(destination)
            output_audits.append({
                "file": name,
                "application": application,
                "level": level,
                "rows": int(len(clean)),
                "columns": int(len(clean.columns)),
                "output_sha256": sha256(destination),
            })

    manifest = {
        "dataset": "IUST_MSCN legacy scale_1 Short Clean-Valid",
        "source_directory": str(source.resolve()),
        "output_directory": str(output.resolve()),
        "source_level_files": audits,
        "captures": output_audits,
        "total_retained_rows": int(sum(item["rows"] for item in output_audits)),
        "service_metadata_not_model_inputs": METADATA,
        "categorical_model_fields": [],
        "numeric_model_fields": NUMERIC,
        "removal_policy": REMOVAL_POLICY,
        "ordering_note": (
            "The legacy files have no StartTime. Original per-application row order "
            "is preserved and represented by a monotonic synthetic timestamp. "
            "Timestamp and endpoint metadata are forbidden model inputs."
        ),
        "comparability_note": (
            "The legacy schema lacks sMeanPktSz, Flgs, State and TcpOpt; they are not "
            "fabricated as model features. Comparisons use the reviewed fields "
            "actually present in the legacy data."
        ),
    }
    (output / "clean_valid_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    repository = root.parent
    parser.add_argument(
        "--source", type=Path,
        default=repository / "DATASETS/CDR-MLC/scale_1/Short",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "DATASETS/CDR-MLC/Legacy_Clean_Valid",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args.source, args.output, args.overwrite)
    print(json.dumps({
        "captures": len(result["captures"]),
        "retained_rows": result["total_retained_rows"],
        "output": result["output_directory"],
    }, indent=2))


if __name__ == "__main__":
    main()
