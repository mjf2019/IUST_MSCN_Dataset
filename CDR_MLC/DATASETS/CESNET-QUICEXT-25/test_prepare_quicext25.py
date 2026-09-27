from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "prepare_quicext25", HERE / "prepare_quicext25.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sample_frame(month: str) -> pd.DataFrame:
    return pd.DataFrame({
        "TIME_FIRST": [f"{month}-01T00:00:01Z", f"{month}-01T00:00:02Z"],
        "TIME_LAST": [f"{month}-01T00:00:02Z", f"{month}-01T00:00:03Z"],
        "DURATION": [1.0, 1.0],
        "BYTES": [100, 200], "BYTES_REV": [300, 400],
        "PACKETS": [2, 2], "PACKETS_REV": [2, 2],
        "QUIC_SNI": ["video.example.com", "api.example.org"],
        "PPI": [
            [[0.0, .1], [1, -1], [1200, 800]],
            [[0.0, .2], [1, -1], [1000, 700]],
        ],
        "PPI_LEN": [2, 2], "PPI_DURATION": [.1, .2],
        "PPI_ROUNDTRIPS": [1, 1],
        "PHIST_SRC_SIZES": [[1] * 8, [2] * 8],
        "PHIST_DST_SIZES": [[2] * 8, [3] * 8],
        "PHIST_SRC_IPT": [[3] * 8, [4] * 8],
        "PHIST_DST_IPT": [[4] * 8, [5] * 8],
        "QUIC_TOKEN_LENGTH": [0, 0],
        "QUIC_MULTIPLEXED": [0, 1],
        "QUIC_ZERO_RTT": [0, 0],
        "DST_IP": ["shortcut-a", "shortcut-b"],
    })


def make_archive(root: Path, month: str) -> Path:
    parquet = root / f"{month}-01.parquet"
    sample_frame(month).to_parquet(parquet, index=False)
    archive = root / f"{month}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.write(parquet, arcname=f"nested/{parquet.name}")
    parquet.unlink()
    return archive


def test_three_month_preparation_and_scenarios(tmp_path):
    output = tmp_path / "processed"
    audits = []
    for month in MODULE.MONTHS:
        archive = make_archive(tmp_path, month)
        audits.append(MODULE.process_month(
            month, archive, output / f"{month}.parquet",
            batch_size=1, compression="zstd", overwrite=False,
        ))
    archive_audit = {
        month: {"path": str(tmp_path / f"{month}.zip")}
        for month in MODULE.MONTHS
    }
    MODULE.write_manifests(output, archive_audit, audits)

    frame = pd.read_parquet(output / "2024-06.parquet")
    assert len(frame) == 2
    assert set(frame.traffic_label) == {"example.com", "example.org"}
    assert "QUIC_SNI" not in frame
    assert "DST_IP" not in frame
    assert "ppi_ipt_30" in frame
    assert "phist_dst_ipt_08" in frame
    assert frame.record_id.is_unique

    manifest = json.loads((output / "scenarios.json").read_text())
    assert set(manifest["scenarios"]) == {f"S{i}" for i in range(1, 8)}
    assert manifest["scenarios"]["S1"]["development_months"] == ["2024-06"]
    assert manifest["scenarios"]["S7"]["outer_split"]["test_fraction"] == .20


def test_daily_sampling_is_capped_unique_and_repeatable():
    first = MODULE.select_source_rows(10_000, 500, 42, "2024-06", "20240601.parquet")
    second = MODULE.select_source_rows(10_000, 500, 42, "2024-06", "20240601.parquet")
    other_day = MODULE.select_source_rows(10_000, 500, 42, "2024-06", "20240602.parquet")
    assert len(first) == 500
    assert len(set(first)) == 500
    assert (first == second).all()
    assert not (first == other_day).all()
    assert MODULE.select_source_rows(
        400, 500, 42, "2024-06", "small.parquet"
    ) is None
