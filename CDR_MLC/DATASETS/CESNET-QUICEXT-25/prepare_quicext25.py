#!/usr/bin/env python3
"""Prepare three CESNET-QUICEXT-25 months for seven transfer scenarios.

The input ZIP files are processed one daily Parquet member and one record batch
at a time. Raw identifiers and label-source fields are never copied to the
model-ready Parquet files. QUIC_SNI is used only to derive an eTLD+1 target.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tldextract


MONTHS = ("2024-06", "2024-07", "2024-08")
EXPECTED_MD5 = {
    "2024-06": "4b1fd8bcf5ddd143f7b350aa7c0d4814",
    "2024-07": "be97737ce804412dbe42c90a69a8aa05",
    "2024-08": "f9f6eb67dc539a9aadc16b5dc08f0951",
}
PPI_PACKETS = 30
HIST_BINS = 8

REQUIRED_COLUMNS = (
    "TIME_FIRST", "TIME_LAST", "DURATION", "BYTES", "BYTES_REV",
    "PACKETS", "PACKETS_REV", "QUIC_SNI", "PPI", "PPI_LEN",
    "PPI_DURATION", "PPI_ROUNDTRIPS", "PHIST_SRC_SIZES",
    "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT",
)
OPTIONAL_SCALARS = (
    "QUIC_TOKEN_LENGTH", "QUIC_MULTIPLEXED", "QUIC_ZERO_RTT",
)
SCALAR_RENAMES = {
    "DURATION": "duration",
    "BYTES": "bytes_src",
    "BYTES_REV": "bytes_dst",
    "PACKETS": "packets_src",
    "PACKETS_REV": "packets_dst",
    "PPI_LEN": "ppi_len",
    "PPI_DURATION": "ppi_duration",
    "PPI_ROUNDTRIPS": "ppi_roundtrips",
    "QUIC_TOKEN_LENGTH": "quic_token_length",
    "QUIC_MULTIPLEXED": "quic_multiplexed",
    "QUIC_ZERO_RTT": "quic_zero_rtt",
}
HISTOGRAMS = {
    "PHIST_SRC_SIZES": "phist_src_size",
    "PHIST_DST_SIZES": "phist_dst_size",
    "PHIST_SRC_IPT": "phist_src_ipt",
    "PHIST_DST_IPT": "phist_dst_ipt",
}
FORBIDDEN_MODEL_FIELDS = (
    "QUIC_SNI", "QUIC_USER_AGENT", "DST_IP", "DST_IP_SUBNET", "DST_ASN",
    "DST_COUNTRY", "QUIC_OCCID", "QUIC_OSCID", "QUIC_SCID",
    "QUIC_RETRY_SCID", "TIME_FIRST", "TIME_LAST", "PROTOCOL",
)
CANDIDATE_CONTEXT_FEATURES = (
    "ppi_duration", "ppi_ipt_mean", "ppi_roundtrips",
)


def scenario_definitions() -> dict[str, dict]:
    pairs = (
        ("S1", "2024-06", "2024-07", "forward"),
        ("S2", "2024-06", "2024-08", "forward"),
        ("S3", "2024-07", "2024-06", "backward"),
        ("S4", "2024-07", "2024-08", "forward"),
        ("S5", "2024-08", "2024-06", "backward"),
        ("S6", "2024-08", "2024-07", "backward"),
    )
    result = {
        name: {
            "kind": "single_month_transfer",
            "development_months": [source],
            "test_months": [target],
            "direction": direction,
            "source_validation": {
                "method": "chronological_tail",
                "fraction": 0.20,
            },
        }
        for name, source, target, direction in pairs
    }
    result["S7"] = {
        "kind": "all_months_chronological",
        "development_months": list(MONTHS),
        "test_months": list(MONTHS),
        "direction": "forward",
        "outer_split": {
            "method": "global_chronological_prefix_tail",
            "development_fraction": 0.80,
            "test_fraction": 0.20,
        },
        "source_validation": {
            "method": "chronological_tail_of_development",
            "fraction": 0.20,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=here / "raw")
    parser.add_argument("--output-dir", type=Path, default=here / "processed")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-checksum", action="store_true",
        help="Skip verification of the official MD5 checksums.",
    )
    return parser.parse_args()


def md5sum(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def locate_archives(raw_dir: Path) -> dict[str, Path]:
    archives = {}
    for month in MONTHS:
        path = raw_dir / f"{month}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"missing input archive: {path}")
        archives[month] = path
    return archives


def verify_archives(archives: dict[str, Path], skip: bool) -> dict[str, dict]:
    audit = {}
    for month, path in archives.items():
        observed = None if skip else md5sum(path)
        expected = EXPECTED_MD5[month]
        if observed is not None and observed.lower() != expected:
            raise ValueError(
                f"checksum mismatch for {path.name}: {observed} != {expected}"
            )
        audit[month] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "expected_md5": expected,
            "observed_md5": observed,
            "checksum_verified": not skip,
        }
    return audit


def _as_sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _as_sequence(decoded)
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return []


def _ppi_components(value) -> tuple[list, list, list]:
    if isinstance(value, dict):
        lowered = {str(key).lower(): item for key, item in value.items()}
        ipt = lowered.get("ipt", lowered.get("inter_packet_times", []))
        direction = lowered.get("dir", lowered.get("directions", []))
        size = lowered.get("size", lowered.get("sizes", []))
        return _as_sequence(ipt), _as_sequence(direction), _as_sequence(size)
    sequence = _as_sequence(value)
    if len(sequence) < 3:
        return [], [], []
    return (
        _as_sequence(sequence[0]),
        _as_sequence(sequence[1]),
        _as_sequence(sequence[2]),
    )


def _matrix(values: pd.Series, component: int, width: int) -> np.ndarray:
    matrix = np.full((len(values), width), np.nan, dtype=np.float32)
    for row, value in enumerate(values):
        sequence = _ppi_components(value)[component]
        if not sequence:
            continue
        numeric = pd.to_numeric(pd.Series(sequence[:width]), errors="coerce")
        matrix[row, :len(numeric)] = numeric.to_numpy(dtype=np.float32)
    return matrix


def _histogram_matrix(values: pd.Series, width: int) -> np.ndarray:
    matrix = np.full((len(values), width), np.nan, dtype=np.float32)
    for row, value in enumerate(values):
        sequence = _as_sequence(value)[:width]
        if not sequence:
            continue
        numeric = pd.to_numeric(pd.Series(sequence), errors="coerce")
        matrix[row, :len(numeric)] = numeric.to_numpy(dtype=np.float32)
    return matrix


def _row_stat(matrix: np.ndarray, operation) -> np.ndarray:
    output = np.full(len(matrix), np.nan, dtype=np.float32)
    valid = ~np.isnan(matrix).all(axis=1)
    if valid.any():
        with np.errstate(all="ignore"):
            output[valid] = operation(matrix[valid], axis=1).astype(np.float32)
    return output


def _direction_changes(direction: np.ndarray) -> np.ndarray:
    output = np.zeros(len(direction), dtype=np.float32)
    for row in range(len(direction)):
        values = direction[row]
        values = values[~np.isnan(values)]
        if len(values) > 1:
            output[row] = np.count_nonzero(np.diff(values))
    return output


def canonical_label(value, extractor, cache: dict[str, str | None]):
    if value is None:
        return None
    raw = str(value).strip().lower().rstrip(".")
    if not raw or raw in {"nan", "none", "null"}:
        return None
    raw = raw.removeprefix("*.")
    if raw in cache:
        return cache[raw]
    try:
        ipaddress.ip_address(raw)
        label = None
    except ValueError:
        extracted = extractor(raw)
        label = extracted.top_domain_under_public_suffix or None
    cache[raw] = label
    return label


def transform_batch(
    raw: pd.DataFrame,
    month: str,
    source_file: str,
    source_offset: int,
    extractor,
    label_cache: dict[str, str | None],
) -> tuple[pd.DataFrame, int]:
    source_rows = np.arange(source_offset, source_offset + len(raw), dtype=np.int64)
    labels = raw["QUIC_SNI"].map(
        lambda value: canonical_label(value, extractor, label_cache)
    )
    keep = labels.notna().to_numpy()
    dropped = int((~keep).sum())
    raw = raw.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    source_rows = source_rows[keep]

    feature_columns: dict[str, object] = {
        "record_id": [f"{month}:{source_file}:{row}" for row in source_rows],
        "source_file": source_file,
        "source_row": source_rows,
        "sequence_id": f"{month}:{Path(source_file).stem}",
        "period": month,
        "timestamp": pd.to_datetime(raw["TIME_FIRST"], errors="coerce", utc=True),
        "traffic_label": labels.astype("string"),
    }
    for source, target in SCALAR_RENAMES.items():
        values = raw[source] if source in raw else np.nan
        feature_columns[target] = pd.to_numeric(values, errors="coerce").astype(np.float32)

    ipt = _matrix(raw["PPI"], 0, PPI_PACKETS)
    direction = _matrix(raw["PPI"], 1, PPI_PACKETS)
    size = _matrix(raw["PPI"], 2, PPI_PACKETS)
    for prefix, matrix in (("ppi_ipt", ipt), ("ppi_dir", direction), ("ppi_size", size)):
        for index in range(PPI_PACKETS):
            feature_columns[f"{prefix}_{index + 1:02d}"] = matrix[:, index]

    for name, matrix in (("ppi_ipt", ipt), ("ppi_size", size)):
        feature_columns[f"{name}_mean"] = _row_stat(matrix, np.nanmean)
        feature_columns[f"{name}_std"] = _row_stat(matrix, np.nanstd)
        feature_columns[f"{name}_min"] = _row_stat(matrix, np.nanmin)
        feature_columns[f"{name}_max"] = _row_stat(matrix, np.nanmax)
        feature_columns[f"{name}_median"] = _row_stat(matrix, np.nanmedian)
    feature_columns["ppi_direction_changes"] = _direction_changes(direction)

    for source, prefix in HISTOGRAMS.items():
        matrix = _histogram_matrix(raw[source], HIST_BINS)
        for index in range(HIST_BINS):
            feature_columns[f"{prefix}_{index + 1:02d}"] = matrix[:, index]

    # Construct once to avoid severe pandas fragmentation on full monthly data.
    result = pd.DataFrame(feature_columns)
    invalid_time = result["timestamp"].isna()
    dropped += int(invalid_time.sum())
    result = result.loc[~invalid_time].reset_index(drop=True)
    return result, dropped


def safe_copy_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path):
    with archive.open(member, "r") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def process_month(
    month: str,
    archive_path: Path,
    output_path: Path,
    batch_size: int,
    compression: str,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; use --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".parquet.incomplete")
    temporary_output.unlink(missing_ok=True)

    extractor = tldextract.TLDExtract(suffix_list_urls=())
    label_cache: dict[str, str | None] = {}
    label_counts: Counter = Counter()
    rows_written = dropped_rows = input_rows = 0
    members_audit = []
    writer = None
    arrow_schema = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                (item for item in archive.infolist()
                 if not item.is_dir() and item.filename.lower().endswith(".parquet")),
                key=lambda item: item.filename,
            )
            if not members:
                raise ValueError(f"no Parquet members found in {archive_path}")
            with tempfile.TemporaryDirectory(prefix=f"quicext-{month}-") as temp:
                temp_dir = Path(temp)
                for member_index, member in enumerate(members, start=1):
                    local = temp_dir / f"member-{member_index:03d}.parquet"
                    safe_copy_member(archive, member, local)
                    parquet = pq.ParquetFile(local)
                    available = set(parquet.schema_arrow.names)
                    missing = sorted(set(REQUIRED_COLUMNS) - available)
                    if missing:
                        raise ValueError(f"{member.filename}: missing columns {missing}")
                    columns = list(REQUIRED_COLUMNS) + [
                        name for name in OPTIONAL_SCALARS if name in available
                    ]
                    member_input = member_output = member_dropped = source_offset = 0
                    for batch in parquet.iter_batches(
                        batch_size=batch_size, columns=columns, use_threads=True
                    ):
                        raw = batch.to_pandas()
                        transformed, dropped = transform_batch(
                            raw, month, member.filename, source_offset,
                            extractor, label_cache,
                        )
                        source_offset += len(raw)
                        member_input += len(raw)
                        member_dropped += dropped
                        if transformed.empty:
                            continue
                        table = pa.Table.from_pandas(
                            transformed, preserve_index=False,
                            schema=arrow_schema,
                        )
                        if writer is None:
                            arrow_schema = table.schema
                            writer = pq.ParquetWriter(
                                temporary_output, table.schema,
                                compression=compression,
                                use_dictionary=["source_file", "sequence_id", "period", "traffic_label"],
                            )
                        writer.write_table(table)
                        label_counts.update(transformed["traffic_label"].astype(str))
                        member_output += len(transformed)
                    input_rows += member_input
                    rows_written += member_output
                    dropped_rows += member_dropped
                    members_audit.append({
                        "member": member.filename,
                        "input_rows": member_input,
                        "output_rows": member_output,
                        "dropped_rows": member_dropped,
                    })
                    local.unlink(missing_ok=True)
                    print(
                        f"[{month}] {member_index}/{len(members)} {member.filename}: "
                        f"{member_output:,} rows",
                        flush=True,
                    )
        if writer is None:
            raise ValueError(f"no valid labeled rows were produced for {month}")
    except Exception:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temporary_output.replace(output_path)

    return {
        "month": month,
        "archive": str(archive_path.resolve()),
        "output": str(output_path.resolve()),
        "input_rows": input_rows,
        "output_rows": rows_written,
        "dropped_rows": dropped_rows,
        "class_count": len(label_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "members": members_audit,
    }


def feature_schema(columns: list[str]) -> dict:
    metadata = [
        "record_id", "source_file", "source_row", "sequence_id", "period",
        "timestamp", "traffic_label",
    ]
    ppi_sequence = [
        name for name in columns
        if name.startswith(("ppi_ipt_", "ppi_dir_", "ppi_size_"))
        and name[-2:].isdigit()
    ]
    histogram = [name for name in columns if name.startswith("phist_")]
    aggregate = [
        name for name in columns
        if name.startswith("ppi_") and name not in ppi_sequence
    ]
    scalar = [
        name for name in columns
        if name not in metadata + ppi_sequence + histogram + aggregate
    ]
    return {
        "metadata": metadata,
        "label": "traffic_label",
        "scalar_features": scalar,
        "ppi_sequence_features": ppi_sequence,
        "ppi_aggregate_features": aggregate,
        "histogram_features": histogram,
        "candidate_context_features": list(CANDIDATE_CONTEXT_FEATURES),
        "forbidden_raw_fields": list(FORBIDDEN_MODEL_FIELDS),
        "recommended_tabular_features": scalar + aggregate + histogram + ppi_sequence,
        "recommended_1d_cnn_order": scalar + aggregate + histogram + ppi_sequence,
        "ppi_packet_limit": PPI_PACKETS,
        "histogram_bins": HIST_BINS,
    }


def write_manifests(output_dir: Path, archive_audit: dict, months: list[dict]):
    first = pq.ParquetFile(months[0]["output"])
    columns = first.schema_arrow.names
    schema = feature_schema(columns)
    (output_dir / "feature_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n", encoding="utf-8"
    )
    scenarios = {
        "schema_version": 1,
        "months": list(MONTHS),
        "scenarios": scenario_definitions(),
        "class_policy": {
            "label_derivation": "eTLD+1 derived from QUIC_SNI",
            "selection_partition": "development training rows only",
            "default_max_classes": 20,
            "default_min_training_rows_per_class": 1000,
            "ranking": "descending training frequency with lexical tie break",
            "target_labels_absent_from_training": "exclude and report",
            "QUIC_SNI_available_to_models": False,
        },
    }
    (output_dir / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2) + "\n", encoding="utf-8"
    )
    counts = []
    for audit in months:
        counts.extend(
            {"period": audit["month"], "traffic_label": label, "rows": rows}
            for label, rows in audit["label_counts"].items()
        )
    pd.DataFrame(counts).sort_values(
        ["period", "rows", "traffic_label"], ascending=[True, False, True]
    ).to_csv(output_dir / "label_counts_audit.csv", index=False)
    dataset = {
        "dataset": "CESNET-QUICEXT-25",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_record": "https://doi.org/10.5281/zenodo.17249078",
        "archive_audit": archive_audit,
        "month_audit": months,
        "feature_schema": "feature_schema.json",
        "scenario_manifest": "scenarios.json",
        "label_counts_are_audit_only": True,
        "model_selection_must_not_use_target_label_counts": True,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = locate_archives(raw_dir)
    archive_audit = verify_archives(archives, args.skip_checksum)
    month_audits = []
    for month in MONTHS:
        month_audits.append(process_month(
            month=month,
            archive_path=archives[month],
            output_path=output_dir / f"{month}.parquet",
            batch_size=args.batch_size,
            compression=args.compression,
            overwrite=args.overwrite,
        ))
    write_manifests(output_dir, archive_audit, month_audits)
    print(f"Prepared data and seven-scenario manifest: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
