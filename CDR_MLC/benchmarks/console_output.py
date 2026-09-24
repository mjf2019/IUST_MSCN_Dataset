"""Compact left-to-right console tables for benchmark runners.

CSV artifacts intentionally retain their complete, descriptive column names.
Only the human-readable console view is abbreviated here.
"""

from __future__ import annotations

import pandas as pd


def _metric(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def _seconds(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.2f}"


def _integer(value) -> str:
    return "-" if pd.isna(value) else str(int(value))


def print_compact_results(
    frame: pd.DataFrame,
    method: str,
    calibration_column: str,
    seconds_column: str,
    extra_columns: dict[str, str] | None = None,
) -> None:
    """Print one physical row per result with short, stable LTR headings."""
    display = pd.DataFrame({
        "AF": frame["adaptation_fraction"].map(lambda value: f"{float(value):.2f}"),
        "TF": frame["fixed_test_fraction"].map(_metric),
        "Scn": frame["scenario"].astype(str),
        "Src": frame["source"].astype(str),
        "Tgt": frame["target"].astype(str),
        "CalN": frame[calibration_column].map(_integer),
        "N": frame["n"].map(_integer),
        "Method": method,
        "Acc": frame["accuracy"].map(_metric),
        "BAcc": frame["balanced_accuracy"].map(_metric),
        "MF1": frame["macro_f1"].map(_metric),
        "WF1": frame["weighted_f1"].map(_metric),
        "Sec": frame[seconds_column].map(_seconds),
    })
    for short_name, full_name in (extra_columns or {}).items():
        display[short_name] = frame[full_name].map(_integer)
    if "status" in frame:
        display["St"] = frame["status"].map(
            lambda value: "OK" if str(value).lower() == "ok" else "N/A"
        )
    print(display.to_string(index=False, justify="left"))
