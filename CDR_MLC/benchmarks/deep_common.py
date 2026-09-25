"""Shared, partition-safe utilities for modern deep benchmark runners."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from adaptive_cdr_mlc import (
    APPLICATIONS, DEFAULT_CANDIDATES, FORBIDDEN, load_dataset,
    select_classifier_columns,
)
from compare_clean_valid import SCENARIOS
from mixed_level_protocols_leakage_safe import PROTOCOLS, build_protocol
from benchmarks.console_output import print_compact_results

LEGACY_EXCLUDED = {"IdleTime", "DstWin"}
MIXED = {"4": "LM-H", "5": "LH-M", "6": "MH-L", "7": "ALL-80-20"}
PROTOCOL_IDS = tuple("1234567")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def record_ids(frame: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(frame.source_file.astype(str), frame.source_row.astype(int)))


def evaluations(data: pd.DataFrame, ids, train_fraction: float = .80):
    result = []
    for protocol_id in ids:
        if protocol_id in SCENARIOS:
            source, target = SCENARIOS[protocol_id]
            development = data[data.congestion_level.eq(source)].copy().reset_index(drop=True)
            test = data[data.congestion_level.eq(target)].copy().reset_index(drop=True)
            definition = {
                "protocol": f"S{protocol_id}", "source": source, "target": target,
                "kind": "complete-level transfer",
            }
        else:
            name = MIXED[protocol_id]
            development, test, raw = build_protocol(data, name, train_fraction)
            spec = PROTOCOLS[name]
            source = (
                "Low+Medium+High" if name == "ALL-80-20"
                else "+".join(spec["train_levels"])
            )
            target = (
                "Low+Medium+High" if name == "ALL-80-20"
                else spec["test_level"]
            )
            definition = {
                "protocol": f"S{protocol_id}", "source": source, "target": target,
                "kind": raw.get("evaluation_type", name),
            }
        if development.empty or test.empty:
            raise ValueError(f"{definition['protocol']}: empty partition")
        overlap = record_ids(development) & record_ids(test)
        if overlap:
            raise RuntimeError(
                f"{definition['protocol']}: {len(overlap)} development/test overlaps"
            )
        result.append((development, test, definition))
    return result


def chronological_validation(frame: pd.DataFrame, fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("--validation-fraction must be in (0,1)")
    train, valid = [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * (1.0 - fraction))
        if not 0 < cut < len(group):
            raise ValueError(f"{sequence_id}: insufficient validation rows")
        train.append(group.iloc[:cut].copy())
        valid.append(group.iloc[cut:].copy())
    return (
        pd.concat(train, ignore_index=True),
        pd.concat(valid, ignore_index=True),
    )


def matrices(train: pd.DataFrame, frames: list[pd.DataFrame]):
    numeric, _ = select_classifier_columns(train, route_features=[])
    features = [
        name for name in numeric
        if name not in FORBIDDEN and name not in LEGACY_EXCLUDED
    ]
    if not features:
        raise ValueError("no usable Clean-Valid features")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(
        imputer.fit_transform(train[features])
    ).astype(np.float32)
    outputs = [train_x]
    for frame in frames:
        outputs.append(
            scaler.transform(imputer.transform(frame[features])).astype(np.float32)
        )
    return features, outputs


def labels(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(APPLICATIONS)}
    encoded = frame.traffic_label.astype(str).map(mapping)
    if encoded.isna().any():
        unknown = sorted(frame.loc[encoded.isna(), "traffic_label"].astype(str).unique())
        raise ValueError(f"unknown labels: {unknown}")
    return encoded.to_numpy(dtype=np.int64)


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(APPLICATIONS)).astype(float)
    weights = len(y) / (len(APPLICATIONS) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def scores(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def predict_batches(model, values, batch_size, device, forward=None):
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            logits = forward(model, batch) if forward else model(batch)
            predictions.append(logits.argmax(1).cpu().numpy())
    seconds = time.perf_counter() - started
    return np.concatenate(predictions), seconds


def save_run(output: Path, method: str, rows: list[dict], audits: dict, manifest: dict):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["protocol", "seed"])
    frame.to_csv(output / "results.csv", index=False)
    (output / "split_audit.json").write_text(
        json.dumps(audits, indent=2) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.json").write_text(
        json.dumps({"method": method, **manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print_compact_results(frame, method=method)
    return frame


def load_clean_valid(data_dir: Path):
    return load_dataset(data_dir, tuple(DEFAULT_CANDIDATES))
