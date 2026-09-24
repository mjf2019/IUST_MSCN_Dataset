"""Leakage-safe standard 1D-CNN benchmark on IUST_MSCN Clean-Valid.

This is an independent adaptation of the supplied Standard_1D_CNN notebook.
It retains its convolutional architecture and optimization defaults, while
using chronological source validation and an immutable target-test tail.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
CDR_MLC = BENCHMARKS.parent
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import (  # noqa: E402
    APPLICATIONS,
    DEFAULT_CANDIDATES,
    FORBIDDEN,
    load_dataset,
    select_classifier_columns,
)
from compare_clean_valid import SCENARIOS  # noqa: E402
from benchmarks.console_output import print_compact_results  # noqa: E402

LEGACY_EXCLUDED = {"IdleTime", "DstWin"}


class CNN1D(nn.Module):
    """Architecture from the supplied Standard_1D_CNN notebook."""

    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        filters = (64, 128, 64)
        self.convolutions = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        in_channels = 1
        for out_channels in filters:
            self.convolutions.append(nn.Conv1d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=True
            ))
            self.conv_norms.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        reduced = input_dim
        for _ in filters:
            reduced //= 2
        if reduced < 1:
            raise ValueError(
                f"1D-CNN requires at least 8 numeric features; found {input_dim}"
            )
        self.dense1 = nn.Linear(filters[-1] * reduced, 128)
        self.dense1_norm = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128, 64)
        self.dense2_norm = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, classes)
        self.dropout = nn.Dropout(.30)

    def forward(self, values):
        for convolution, normalizer in zip(self.convolutions, self.conv_norms):
            values = F.max_pool1d(F.relu(normalizer(convolution(values))), 2)
        values = torch.flatten(values, 1)
        values = self.dropout(F.relu(self.dense1_norm(self.dense1(values))))
        values = self.dropout(F.relu(self.dense2_norm(self.dense2(values))))
        return self.output(values)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_target_tail(frame: pd.DataFrame, adaptation_fraction: float,
                      test_fraction: float):
    calibration_parts, test_parts, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        calibration_end = int(len(group) * adaptation_fraction)
        test_start = int(len(group) * (1.0 - test_fraction))
        if calibration_end > test_start:
            raise ValueError(f"{sequence_id}: adaptation prefix overlaps test tail")
        calibration_parts.append(group.iloc[:calibration_end].copy())
        test_parts.append(group.iloc[test_start:].copy())
        audit.append({
            "sequence_id": sequence_id,
            "rows": int(len(group)),
            "calibration_rows": int(calibration_end),
            "unused_rows": int(test_start - calibration_end),
            "test_rows": int(len(group) - test_start),
        })
    empty = frame.iloc[:0].copy()
    calibration = (
        pd.concat(calibration_parts, ignore_index=True)
        if adaptation_fraction > 0 else empty
    )
    return calibration, pd.concat(test_parts, ignore_index=True), audit


def chronological_source_validation(source: pd.DataFrame, validation_fraction: float):
    train_parts, validation_parts = [], []
    for sequence_id, group in source.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * (1.0 - validation_fraction))
        if not 0 < cut < len(group):
            raise ValueError(f"{sequence_id}: insufficient rows for validation")
        train_parts.append(group.iloc[:cut].copy())
        validation_parts.append(group.iloc[cut:].copy())
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
    )


def fitted_matrices(train: pd.DataFrame, other_frames: list[pd.DataFrame]):
    numeric, _ = select_classifier_columns(train, route_features=[])
    numeric = [
        name for name in numeric
        if name not in FORBIDDEN and name not in LEGACY_EXCLUDED
    ]
    if len(numeric) < 8:
        raise ValueError(f"1D-CNN requires at least 8 numeric features; found {len(numeric)}")
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train[numeric])
    scaler = StandardScaler()
    matrices = [scaler.fit_transform(train_imputed).astype(np.float32)]
    for frame in other_frames:
        matrices.append(
            scaler.transform(imputer.transform(frame[numeric])).astype(np.float32)
        )
    return numeric, matrices


def encoded_labels(frame: pd.DataFrame):
    mapping = {name: index for index, name in enumerate(APPLICATIONS)}
    labels = frame.traffic_label.astype(str).map(mapping)
    if labels.isna().any():
        unknown = sorted(frame.loc[labels.isna(), "traffic_label"].astype(str).unique())
        raise ValueError(f"unknown traffic labels: {unknown}")
    return labels.to_numpy(dtype=np.int64)


def loader(values, labels, batch_size: int, shuffle: bool, seed: int):
    dataset = TensorDataset(
        torch.from_numpy(values).unsqueeze(1), torch.from_numpy(labels)
    )
    generator = torch.Generator().manual_seed(seed)
    # A final training batch of one sample is invalid for dense BatchNorm.
    drop_last = bool(shuffle and len(dataset) > 1 and len(dataset) % batch_size == 1)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, generator=generator,
    )


def validation_loss(model, data_loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for values, labels in data_loader:
            values, labels = values.to(device), labels.to(device)
            loss = criterion(model(values), labels)
            total += float(loss.item()) * len(labels)
            count += len(labels)
    return total / max(count, 1)


def fit_model(train_x, train_y, validation_x, validation_y, args, device):
    seed_everything(args.seed)
    model = CNN1D(train_x.shape[1], len(APPLICATIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=5
    )
    train_loader = loader(train_x, train_y, args.batch_size, True, args.seed)
    valid_loader = loader(validation_x, validation_y, args.batch_size, False, args.seed)
    best_loss, best_state, stale, completed = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        for values, labels in train_loader:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
        current = validation_loss(model, valid_loader, criterion, device)
        scheduler.step(current)
        completed = epoch + 1
        if current < best_loss - args.min_delta:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("1D-CNN training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return model, completed, best_loss


def predict(model, values, batch_size: int, device, seed: int):
    dummy = np.zeros(len(values), dtype=np.int64)
    data_loader = loader(values, dummy, batch_size, False, seed)
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch, _ in data_loader:
            predictions.append(model(batch.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def metric_values(truth, prediction):
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
    }


def run(args):
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be in (0,1)")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in (0,1)")
    if any(value < 0 or value > 1 - args.test_fraction for value in args.fractions):
        raise ValueError("an adaptation fraction overlaps the fixed test tail")

    args.output.mkdir(parents=True, exist_ok=True)
    data, input_audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results, split_audits = [], {}

    for scenario in args.scenarios:
        source_level, target_level = SCENARIOS[scenario]
        source = data[data.congestion_level.eq(source_level)].copy()
        target = data[data.congestion_level.eq(target_level)].copy()
        source_train, source_validation = chronological_source_validation(
            source, args.validation_fraction
        )
        for fraction in args.fractions:
            calibration, test, audit = fixed_target_tail(
                target, fraction, args.test_fraction
            )
            train = pd.concat([source_train, calibration], ignore_index=True)
            features, matrices = fitted_matrices(
                train, [source_validation, test]
            )
            train_x, validation_x, test_x = matrices
            train_y = encoded_labels(train)
            validation_y = encoded_labels(source_validation)
            test_y = encoded_labels(test)
            started = time.perf_counter()
            model, epochs_completed, best_validation_loss = fit_model(
                train_x, train_y, validation_x, validation_y, args, device
            )
            prediction = predict(model, test_x, args.batch_size, device, args.seed)
            elapsed = time.perf_counter() - started
            results.append({
                "adaptation_fraction": fraction,
                "fixed_test_fraction": args.test_fraction,
                "scenario": scenario,
                "source": source_level,
                "target": target_level,
                "method": "Standard_1D_CNN",
                "train_n": int(len(train)),
                "validation_n": int(len(source_validation)),
                "target_labeled_n": int(len(calibration)),
                "n": int(len(test)),
                "epochs_completed": epochs_completed,
                "best_validation_loss": best_validation_loss,
                **metric_values(test_y, prediction),
                "fit_and_inference_seconds": elapsed,
            })
            key = f"scenario={scenario}:fraction={fraction:.4f}"
            split_audits[key] = {
                "captures": audit,
                "feature_count": len(features),
                "features": features,
                "calibration_test_overlap_by_capture_row": int(len(
                    set(zip(calibration.source_file, calibration.source_row))
                    & set(zip(test.source_file, test.source_row))
                )),
            }

    summary = pd.DataFrame(results).sort_values(["adaptation_fraction", "scenario"])
    summary.to_csv(args.output / "standard_1d_cnn_summary.csv", index=False)
    (args.output / "split_audit.json").write_text(
        json.dumps(split_audits, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "method": "Standard_1D_CNN",
        "data_dir": str(args.data_dir),
        "fractions": args.fractions,
        "test_fraction": args.test_fraction,
        "validation_fraction": args.validation_fraction,
        "scenarios": args.scenarios,
        "seed": args.seed,
        "device": str(device),
        "architecture": {
            "filters": [64, 128, 64], "kernels": [3, 3, 3],
            "pool_sizes": [2, 2, 2], "dense": [128, 64], "dropout": .30,
        },
        "optimizer": {
            "name": "Adam", "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "protocol": "chronological source validation; source train plus disjoint target prefix; immutable target tail",
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print_compact_results(
        summary, method="1D-CNN", calibration_column="target_labeled_n",
        seconds_column="fit_and_inference_seconds",
        extra_columns={"Ep": "epochs_completed"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "outputs/iust_mscn")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .01, .05, .10, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
