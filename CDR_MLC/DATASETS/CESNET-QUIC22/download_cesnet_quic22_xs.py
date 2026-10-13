#!/usr/bin/env python3
"""Download the official CESNET-QUIC22 XS dataset via CESNET DataZoo.

This script intentionally performs download only. Dataset configuration,
feature filtering, temporal splitting, and preprocessing are separate steps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


DATASET_NAME = "CESNET-QUIC22"
DATASET_SIZE = "XS"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CESNET-QUIC22 in the official XS size."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Storage root passed to CESNET DataZoo. "
            "Default: <script-directory>/raw"
        ),
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress DataZoo progress output.",
    )
    return parser.parse_args()


def package_version() -> str:
    try:
        return version("cesnet-datazoo")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        from cesnet_datazoo.datasets import CESNET_QUIC22
    except ImportError:
        print(
            "ERROR: cesnet-datazoo is not installed. Run "
            "'python -m pip install -r requirements-download.txt'.",
            file=sys.stderr,
        )
        return 2

    print(f"Dataset : {DATASET_NAME}")
    print(f"Size    : {DATASET_SIZE}")
    print(f"Root    : {data_root}")
    print("Initializing DataZoo; missing XS files will be downloaded...")

    try:
        dataset = CESNET_QUIC22(
            str(data_root),
            size=DATASET_SIZE,
            silent=args.silent,
        )
    except KeyboardInterrupt:
        print("\nDownload interrupted by the user.", file=sys.stderr)
        return 130
    except Exception as exc:  # DataZoo supplies the actionable download error.
        print(f"ERROR: download failed: {exc}", file=sys.stderr)
        return 1

    database_path = Path(dataset.database_path).resolve()
    if not database_path.is_file():
        print(
            f"ERROR: DataZoo returned, but the database is missing: {database_path}",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "dataset": DATASET_NAME,
        "size": DATASET_SIZE,
        "datazoo_version": package_version(),
        "download_completed_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "database_path": str(database_path),
        "database_bytes": database_path.stat().st_size,
        "available_periods": sorted(dataset.time_periods),
    }
    manifest_path = Path(__file__).resolve().parent / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Download verified.")
    print(f"Database: {database_path}")
    print(f"Bytes   : {manifest['database_bytes']:,}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
