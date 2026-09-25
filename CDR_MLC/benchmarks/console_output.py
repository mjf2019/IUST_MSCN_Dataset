"""Uniform compact console output and resource measurement for benchmarks."""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

import pandas as pd
import psutil

ABBREVIATIONS = OrderedDict([
    ("AF", "Target adaptation fraction"),
    ("TF", "Fixed test fraction"),
    ("Scn", "Scenario or evaluation protocol"),
    ("Src", "Development congestion level(s)"),
    ("Tgt", "Test congestion level(s)"),
    ("Sd", "Random seed"),
    ("M", "Method"),
    ("CalN", "Labeled target calibration rows"),
    ("N", "Evaluated rows"),
    ("Acc", "Accuracy"),
    ("BAcc", "Balanced accuracy"),
    ("MF1", "Macro F1-score"),
    ("WF1", "Weighted F1-score"),
    ("FitS", "Training time in seconds"),
    ("InfS", "Inference time in seconds"),
    ("us/R", "Inference microseconds per input row"),
    ("R/s", "Inference input rows per second"),
    ("RAM", "Peak process resident memory in MiB"),
    ("GPU", "Peak allocated GPU memory in MiB"),
    ("TTus", "TTFEF microseconds per input row"),
    ("Ep", "Completed training epochs"),
    ("St", "Execution status"),
])


def _metric(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.4f}"


def _seconds(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.2f}"


def _two(value) -> str:
    return "-" if pd.isna(value) else f"{float(value):.2f}"


def _integer(value) -> str:
    return "-" if pd.isna(value) else str(int(value))


def print_abbreviation_table(keys) -> None:
    unique = list(dict.fromkeys(keys))
    legend = pd.DataFrame({
        "Key": unique,
        "Meaning": [ABBREVIATIONS.get(key, key) for key in unique],
    })
    print("\nColumn abbreviations")
    print(legend.to_string(index=False))


class ResourceMonitor:
    """Poll peak process RSS and CUDA allocation during one measured phase."""

    def __init__(self, device=None, interval: float = .01):
        self.device = str(device or "cpu")
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._torch = None

    def _sample(self):
        self.peak_ram_mb = max(
            self.peak_ram_mb,
            self.process.memory_info().rss / (1024.0 ** 2),
        )

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        if self.device.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    self._torch = torch
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                self._torch = None
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        if self._torch is not None:
            self.peak_gpu_mb = (
                self._torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            )


def print_compact_results(
    frame: pd.DataFrame,
    method: str | None = None,
    calibration_column: str | None = None,
    seconds_column: str | None = None,
    extra_columns: dict[str, str] | None = None,
) -> None:
    """Print a two-column legend followed by one compact physical row per run."""
    display = pd.DataFrame(index=frame.index)
    candidates = [
        ("AF", "adaptation_fraction", lambda x: f"{float(x):.2f}"),
        ("TF", "fixed_test_fraction", _metric),
        ("Scn", "protocol", str),
        ("Scn", "scenario", str),
        ("Src", "source", str),
        ("Tgt", "target", str),
        ("Sd", "seed", _integer),
    ]
    used = set()
    for short, full, formatter in candidates:
        if short not in used and full in frame:
            display[short] = frame[full].map(formatter)
            used.add(short)

    if method is not None:
        display["M"] = method
    elif "method" in frame:
        display["M"] = frame["method"].astype(str)

    if calibration_column and calibration_column in frame:
        display["CalN"] = frame[calibration_column].map(_integer)
    if "n" in frame:
        display["N"] = frame["n"].map(_integer)

    metric_map = [
        ("Acc", "accuracy"), ("BAcc", "balanced_accuracy"),
        ("MF1", "macro_f1"), ("WF1", "weighted_f1"),
    ]
    for short, full in metric_map:
        if full in frame:
            display[short] = frame[full].map(_metric)

    if "fit_seconds" in frame:
        display["FitS"] = frame["fit_seconds"].map(_seconds)
    if "predict_seconds" in frame:
        display["InfS"] = frame["predict_seconds"].map(_seconds)
    elif seconds_column and seconds_column in frame:
        display["FitS"] = frame[seconds_column].map(_seconds)

    efficiency = [
        ("us/R", "inference_us_per_row", _two),
        ("us/R", "inference_microseconds_per_input_row", _two),
        ("R/s", "throughput_rows_per_second", _two),
        ("R/s", "throughput_input_rows_per_second", _two),
        ("RAM", "peak_ram_mb", _two),
        ("GPU", "peak_gpu_mb", _two),
        ("TTus", "ttfef_us_per_row", _two),
    ]
    for short, full, formatter in efficiency:
        if short not in display and full in frame:
            display[short] = frame[full].map(formatter)

    for short, full in (extra_columns or {}).items():
        if full in frame:
            display[short] = frame[full].map(_integer)
    if "status" in frame:
        display["St"] = frame["status"].map(
            lambda value: "OK" if str(value).lower() == "ok" else "N/A"
        )

    print_abbreviation_table(display.columns)
    with pd.option_context(
        "display.width", 240,
        "display.max_columns", None,
        "display.expand_frame_repr", False,
    ):
        print("\nResults")
        print(display.to_string(index=False))


def resource_values(*monitors: ResourceMonitor) -> dict[str, float]:
    return {
        "peak_ram_mb": max((m.peak_ram_mb for m in monitors), default=0.0),
        "peak_gpu_mb": max((m.peak_gpu_mb for m in monitors), default=0.0),
    }
