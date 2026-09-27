"""Shared data, serialization, CPU-control and measurement utilities."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from adaptive_cdr_mlc import FORBIDDEN, select_classifier_columns


LEGACY_EXCLUDED = {"IdleTime", "DstWin"}


def record_id(frame: pd.DataFrame) -> list[str]:
    return (
        frame.source_file.astype(str) + ":" + frame.source_row.astype(str)
    ).tolist()


def frame_sha256(frame: pd.DataFrame) -> str:
    payload = "\n".join(record_id(frame)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def dump_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: Path) -> dict:
    return joblib.load(path)


def fit_tabular_preprocessor(train: pd.DataFrame, frames=()):
    numeric, _ = select_classifier_columns(train, route_features=[])
    features = [
        name for name in numeric
        if name not in FORBIDDEN and name not in LEGACY_EXCLUDED
    ]
    if not features:
        raise ValueError("no usable numeric classifier features")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(
        imputer.fit_transform(train[features])
    ).astype(np.float32)
    values = [train_values]
    for frame in frames:
        values.append(
            scaler.transform(imputer.transform(frame[features])).astype(np.float32)
        )
    return features, imputer, scaler, values


def transform_tabular(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    return artifact["scaler"].transform(
        artifact["imputer"].transform(frame[artifact["features"]])
    ).astype(np.float32)


def metric_values(truth, prediction) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(
            truth, prediction, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            truth, prediction, average="weighted", zero_division=0
        )),
    }


def set_estimator_jobs(value, jobs: int) -> None:
    """Recursively cap sklearn estimators contained in a model artifact."""
    visited = set()

    def visit(item):
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if hasattr(item, "n_jobs"):
            try:
                item.n_jobs = jobs
            except (AttributeError, TypeError):
                pass
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list, set)):
            for child in item:
                visit(child)

    visit(value)


def configure_cpu(threads: int) -> None:
    if threads < 1:
        raise ValueError("CPU thread count must be positive")
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


@contextmanager
def cpu_thread_limit(threads: int):
    with threadpool_limits(limits=threads):
        yield


class ProcessSampler:
    """Sample process RSS and CPU use without including model training."""

    def __init__(self, interval=.005):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop = threading.Event()
        self.thread = None
        self.baseline_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.started_wall = 0.0
        self.started_cpu = 0.0
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0

    def _rss(self):
        return self.process.memory_info().rss / (1024.0 ** 2)

    def _poll(self):
        while not self.stop.wait(self.interval):
            self.peak_rss_mb = max(self.peak_rss_mb, self._rss())

    def __enter__(self):
        self.baseline_rss_mb = self._rss()
        self.peak_rss_mb = self.baseline_rss_mb
        self.started_cpu = time.process_time()
        self.started_wall = time.perf_counter()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.wall_seconds = time.perf_counter() - self.started_wall
        self.cpu_seconds = time.process_time() - self.started_cpu
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_rss_mb = max(self.peak_rss_mb, self._rss())

    def values(self):
        return {
            "wall_seconds": self.wall_seconds,
            "process_cpu_seconds": self.cpu_seconds,
            "cpu_core_equivalents": (
                self.cpu_seconds / self.wall_seconds
                if self.wall_seconds > 0 else np.nan
            ),
            "baseline_rss_mb": self.baseline_rss_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_rss_delta_mb": max(
                0.0, self.peak_rss_mb - self.baseline_rss_mb
            ),
        }
