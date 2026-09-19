"""Build the reproducible Clean-Valid CDR-MLC dataset from revised Argus files.

The output keeps only endpoint metadata needed for service filtering plus the
reviewed model fields. It removes capture fingerprints, empty/constant fields,
invalid gap fields, and redundant duration aliases. Source files are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


APPLICATIONS = ["HTTP", "SFTP", "SMTP", "SSH", "Video"]
LEVELS = ["Low", "Medium", "High"]
CLIENT = "192.168.1.111"
SERVICES = {
    "HTTP": ("192.168.2.122", 8080),
    "SFTP": ("192.168.2.120", 22),
    "SMTP": ("192.168.2.120", 8025),
    "SSH": ("192.168.2.120", 22),
    "Video": ("192.168.2.121", 5000),
}

# These fields are retained only so every downstream loader can verify service
# direction. They must never be classifier inputs.
SERVICE_METADATA = ["StartTime", "SrcAddr", "DstAddr", "Proto", "Sport", "Dport"]
CATEGORICAL = ["Flgs", "State", "TcpOpt"]
NUMERIC = [
    "Dur", "TotPkts", "SrcPkts", "DstPkts", "TotBytes", "SrcBytes", "DstBytes",
    "sMeanPktSz", "dMeanPktSz", "SrcLoad", "DstLoad", "Load", "SrcRate",
    "DstRate", "Rate", "SrcLoss", "DstLoss", "Loss", "pLoss", "SrcWin",
    "DstWin", "TcpRtt", "SynAck", "AckDat",
]
OUTPUT_COLUMNS = SERVICE_METADATA + CATEGORICAL + NUMERIC

# Reasons are fixed before the model comparison and written into the manifest.
REMOVAL_POLICY = {
    "capture_fingerprint_or_unresolved_semantics": ["IdleTime"],
    "fully_missing_in_revised_dataset": [
        "SIntPkt", "DIntPkt", "SIntDist", "DIntDist", "SIntPktAct", "DIntPktAct",
        "SIntPktIdl", "DIntPktIdl", "SIntActDist", "DIntActDist", "SIntIdlDist",
        "DIntIdlDist", "sMinPktSz", "sMaxPktSz", "dMinPktSz", "dMaxPktSz",
    ],
    "constant_in_revised_dataset": [
        "StdDev", "sTtl", "dTtl", "SrcRetra", "DstRetra", "PCRatio", "pRetran",
        "Proto.1",
    ],
    "invalid_or_effectively_empty_gap": ["SrcGap", "DstGap"],
    "redundant_duration_alias_keep_Dur": ["Dur.1", "Mean", "Sum", "Min", "Max"],
    "labels_or_export_metadata": ["Label", "Cause", "Dir"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_capture(path: Path, application: str) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, low_memory=False, on_bad_lines="error")
    raw.columns = raw.columns.str.strip()
    missing = sorted(set(OUTPUT_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"{path.name}: missing required Clean-Valid fields {missing}")
    for column in raw.select_dtypes("object"):
        raw[column] = raw[column].str.strip().replace("", np.nan)

    server, port = SERVICES[application]
    sport = pd.to_numeric(raw.Sport, errors="coerce")
    dport = pd.to_numeric(raw.Dport, errors="coerce")
    tcp = raw.Proto.astype(str).str.lower().eq("tcp")
    forward = tcp & raw.SrcAddr.eq(CLIENT) & raw.DstAddr.eq(server) & dport.eq(port)
    reverse = tcp & raw.SrcAddr.eq(server) & raw.DstAddr.eq(CLIENT) & sport.eq(port)
    if reverse.any():
        raise ValueError(
            f"{path.name}: contains {int(reverse.sum())} reverse rows; canonicalization "
            "must be specified before building Clean-Valid"
        )
    clean = raw.loc[forward, OUTPUT_COLUMNS].copy()
    if clean.empty:
        raise ValueError(f"{path.name}: no canonical service records")
    clean["Sport"] = pd.to_numeric(clean.Sport, errors="coerce").astype("Int64")
    clean["Dport"] = pd.to_numeric(clean.Dport, errors="coerce").astype("Int64")
    for column in NUMERIC:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        negative = clean[column].lt(0)
        if negative.any():
            raise ValueError(f"{path.name}: {column} has {int(negative.sum())} negative values")
    timestamp = pd.to_datetime(
        clean.StartTime, format="%Y/%m/%d %H:%M:%S.%f", errors="raise"
    )
    clean = clean.assign(_timestamp=timestamp, _source_order=np.flatnonzero(forward))
    clean = clean.sort_values(["_timestamp", "_source_order"], kind="stable").drop(
        columns=["_timestamp", "_source_order"]
    )
    return clean, {
        "file": path.name,
        "raw_rows": int(len(raw)),
        "retained_rows": int(len(clean)),
        "dropped_nonservice_rows": int(len(raw) - len(clean)),
        "missing_by_retained_field": {
            column: int(clean[column].isna().sum()) for column in OUTPUT_COLUMNS
        },
        "source_sha256": sha256(path),
    }


def build(source: Path, output: Path, overwrite: bool = False) -> dict:
    expected = {
        f"{application}_{level}.flow" for application in APPLICATIONS for level in LEVELS
    }
    observed = {path.name for path in source.glob("*.flow")}
    if observed != expected:
        raise ValueError(
            f"expected exactly 15 revised captures; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in expected if (output / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Clean-Valid already contains {len(existing)} capture(s); use --overwrite explicitly"
        )

    audits = []
    for name in sorted(expected):
        match = re.fullmatch(r"(HTTP|SFTP|SMTP|SSH|Video)_(Low|Medium|High)\.flow", name)
        if not match:
            raise AssertionError(name)
        application, _ = match.groups()
        clean, audit = clean_capture(source / name, application)
        destination = output / name
        temporary = output / f"{name}.tmp"
        clean.to_csv(temporary, index=False)
        temporary.replace(destination)
        audit["output_sha256"] = sha256(destination)
        audit["output_columns"] = len(clean.columns)
        audits.append(audit)

    manifest = {
        "dataset": "IUST_MSCN_Dataset Clean-Valid",
        "source_directory": str(source.resolve()),
        "output_directory": str(output.resolve()),
        "captures": audits,
        "total_raw_rows": int(sum(item["raw_rows"] for item in audits)),
        "total_retained_rows": int(sum(item["retained_rows"] for item in audits)),
        "service_metadata_not_model_inputs": SERVICE_METADATA,
        "categorical_model_fields": CATEGORICAL,
        "numeric_model_fields": NUMERIC,
        "removal_policy": REMOVAL_POLICY,
        "methodological_note": (
            "The policy was frozen after a whole-dataset quality audit. This dataset is for "
            "transparent curation/ablation; it does not restore an untouched evaluation set."
        ),
    }
    (output / "clean_valid_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--source", type=Path, default=root / "DATASETS/CDR-MLC/New_Version"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "DATASETS/CDR-MLC/Clean_Valid"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = build(args.source, args.output, args.overwrite)
    print(json.dumps({
        "captures": len(result["captures"]),
        "raw_rows": result["total_raw_rows"],
        "retained_rows": result["total_retained_rows"],
        "output": result["output_directory"],
    }, indent=2))


if __name__ == "__main__":
    main()
