"""Inject 20% of both other congestion levels into every Clean-Valid file.

The operation is intentionally in-place so all existing runners keep their
current command and data directory. For every application A:

* A_Low    <- all A_Low    + prefix(A_Medium, 20%) + prefix(A_High, 20%)
* A_Medium <- all A_Medium + prefix(A_Low, 20%)    + prefix(A_High, 20%)
* A_High   <- all A_High   + prefix(A_Low, 20%)    + prefix(A_Medium, 20%)

All pristine inputs are loaded before any destination is written, preventing
recursive injection. A manifest records pre/post hashes and exact row counts.
The script refuses to run twice on an already injected directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


APPLICATIONS = ("HTTP", "SFTP", "SMTP", "SSH", "Video")
LEVELS = ("Low", "Medium", "High")
ORIGIN_COLUMN = "InjectionOriginLevel"
ROLE_COLUMN = "InjectionRole"
MANIFEST_NAME = "cross_level_injection_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_capture(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: missing CSV header")
        if ORIGIN_COLUMN in reader.fieldnames or ROLE_COLUMN in reader.fieldnames:
            raise ValueError(
                f"{path.name}: injection metadata already exists; "
                "restore the previous Git commit before reinjecting"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name}: empty capture")
    return list(reader.fieldnames), rows


def annotated(row: dict, origin: str, role: str) -> dict:
    result = dict(row)
    result[ORIGIN_COLUMN] = origin
    result[ROLE_COLUMN] = role
    return result


def write_atomic(path: Path, fieldnames: list[str], rows: list[dict]):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run(data_dir: Path, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("--fraction must be in (0,1)")
    manifest_path = data_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; restore the pre-injection commit first"
        )

    originals, headers, pre_hashes = {}, {}, {}
    for application in APPLICATIONS:
        for level in LEVELS:
            key = (application, level)
            path = data_dir / f"{application}_{level}.flow"
            if not path.is_file():
                raise FileNotFoundError(path)
            header, rows = read_capture(path)
            headers[key], originals[key] = header, rows
            pre_hashes[path.name] = sha256(path)

    outputs, composition = {}, []
    for application in APPLICATIONS:
        application_headers = [headers[(application, level)] for level in LEVELS]
        if any(header != application_headers[0] for header in application_headers[1:]):
            raise ValueError(f"{application}: level files have different schemas")
        output_header = application_headers[0] + [ORIGIN_COLUMN, ROLE_COLUMN]
        for destination in LEVELS:
            primary = [
                annotated(row, destination, "primary")
                for row in originals[(application, destination)]
            ]
            combined = list(primary)
            composition.append({
                "application": application,
                "destination_level": destination,
                "origin_level": destination,
                "role": "primary",
                "available_rows": len(primary),
                "selected_rows": len(primary),
                "realized_fraction": 1.0,
            })
            for origin in LEVELS:
                if origin == destination:
                    continue
                donor = originals[(application, origin)]
                selected_count = int(len(donor) * fraction)
                if selected_count <= 0:
                    raise ValueError(
                        f"{application}_{origin}: empty nonzero donor prefix"
                    )
                selected = [
                    annotated(row, origin, "injected")
                    for row in donor[:selected_count]
                ]
                combined.extend(selected)
                composition.append({
                    "application": application,
                    "destination_level": destination,
                    "origin_level": origin,
                    "role": "injected",
                    "available_rows": len(donor),
                    "selected_rows": selected_count,
                    "realized_fraction": selected_count / len(donor),
                })
            outputs[(application, destination)] = (output_header, combined)

    # Every output is fully prepared before the first existing file is replaced.
    for (application, destination), (header, rows) in outputs.items():
        write_atomic(data_dir / f"{application}_{destination}.flow", header, rows)

    post_hashes, files = {}, []
    for application in APPLICATIONS:
        for level in LEVELS:
            path = data_dir / f"{application}_{level}.flow"
            rows = outputs[(application, level)][1]
            primary_rows = sum(row[ROLE_COLUMN] == "primary" for row in rows)
            injected_rows = len(rows) - primary_rows
            post_hashes[path.name] = sha256(path)
            files.append({
                "file": path.name,
                "application": application,
                "nominal_level": level,
                "primary_rows": primary_rows,
                "injected_rows": injected_rows,
                "total_rows": len(rows),
                "pre_sha256": pre_hashes[path.name],
                "post_sha256": post_hashes[path.name],
            })

    manifest = {
        "operation": "in-place cross-level prefix injection",
        "fraction_from_each_other_level": fraction,
        "applications": list(APPLICATIONS),
        "levels": list(LEVELS),
        "selection": "first floor(fraction * donor_rows) rows from pristine donor",
        "recursive_injection": False,
        "destination_filename_and_runner_commands_unchanged": True,
        "nominal_level_policy": (
            "existing runners assign the destination level from the filename; "
            "InjectionOriginLevel preserves donor provenance and is forbidden as a feature"
        ),
        "files": files,
        "composition": composition,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("File App Level PrimaryN InjectN TotalN")
    for item in files:
        print(
            item["file"], item["application"], item["nominal_level"],
            item["primary_rows"], item["injected_rows"], item["total_rows"],
        )


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--fraction", type=float, default=.20)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data_dir, arguments.fraction)
