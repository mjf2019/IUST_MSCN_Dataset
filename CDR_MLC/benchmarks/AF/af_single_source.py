"""AF-SingleSource (CODASPY 2021) with a leakage-safe evaluation protocol.

The paper-faithful input is a fixed-length Tor packet-direction vector where
outgoing packets are +1, incoming packets are -1, and padding is 0.  For use
as a benchmark on tabular flow features, select ``--input-mode tabular``; that
variant is deliberately reported as AF-MLP-adapted rather than exact AF.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from benchmarks.console_output import (
    ResourceMonitor, print_compact_results, resource_values,
)


@dataclass(frozen=True)
class AFConfig:
    # Protocol reported in the AF paper.
    source_per_class: int = 25       # M
    target_train_pool: int = 20
    target_test_per_class: int = 70  # T
    n_shots: tuple[int, ...] = (1, 5, 10, 15, 20)  # N
    folds: int = 10
    pretrain_epochs: int = 30
    learning_rate: float = 1e-5
    grl_lambda: float = 1.0
    embedding_dim: int = 512
    batch_size: int = 32
    seed: int = 42


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


class DirectionFeatureExtractor(nn.Module):
    """DF-style 1-D CNN for packet-direction vectors."""

    def __init__(self, input_length: int, embedding_dim: int):
        super().__init__()
        blocks = []
        channels = (1, 32, 64, 128, 256)
        for block in range(4):
            blocks.extend(
                [
                    nn.Conv1d(channels[block], channels[block + 1], 8, padding=4),
                    nn.BatchNorm1d(channels[block + 1]),
                    nn.ELU(),
                    nn.Conv1d(channels[block + 1], channels[block + 1], 8, padding=4),
                    nn.BatchNorm1d(channels[block + 1]),
                    nn.ELU(),
                    nn.MaxPool1d(8, stride=4, padding=2),
                    nn.Dropout(0.1),
                ]
            )
        self.convolution = nn.Sequential(*blocks)
        with torch.no_grad():
            flattened = self.convolution(torch.zeros(1, 1, input_length)).numel()
        self.embedding = nn.Sequential(
            nn.Flatten(), nn.Linear(flattened, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.convolution(x.unsqueeze(1)))


class TabularFeatureExtractor(nn.Module):
    """Documented adaptation for flow-feature datasets; not the DF network."""

    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AFDomainNetwork(nn.Module):
    def __init__(self, extractor: nn.Module, classes: int, config: AFConfig):
        super().__init__()
        self.extractor = extractor
        self.coefficient = config.grl_lambda
        self.source_classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, 256), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(256, classes),
        )
        self.domain_discriminator = nn.Sequential(
            nn.Linear(config.embedding_dim, 256), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor):
        features = self.extractor(x)
        source_logits = self.source_classifier(features)
        reversed_features = GradientReversal.apply(features, self.coefficient)
        domain_logits = self.domain_discriminator(reversed_features)
        return source_logits, domain_logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path, label_column: str):
    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"label column {label_column!r} is absent from {path}")
    labels = frame.pop(label_column).astype(str).to_numpy()
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"non-numeric or missing input columns: {bad}")
    return numeric.to_numpy(dtype=np.float32), labels, list(numeric.columns)


def select_per_class(y: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if len(indices) < count:
            raise ValueError(f"class {label} has {len(indices)} rows; {count} required")
        selected.extend(rng.choice(indices, count, replace=False))
    return np.asarray(selected, dtype=int)


def target_partition(y: np.ndarray, config: AFConfig, rng: np.random.Generator):
    """Create disjoint 20-per-class training pools and 70-per-class tests."""
    pools, tests = [], []
    required = config.target_train_pool + config.target_test_per_class
    for label in np.unique(y):
        indices = np.flatnonzero(y == label).copy()
        if len(indices) < required:
            raise ValueError(f"target class {label} has {len(indices)} rows; {required} required")
        rng.shuffle(indices)
        pools.extend(indices[: config.target_train_pool])
        tests.extend(indices[config.target_train_pool : required])
    pools, tests = np.asarray(pools), np.asarray(tests)
    if np.intersect1d(pools, tests).size:
        raise AssertionError("target train/test overlap")
    return pools, tests


def n_shot_subset(pool: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    chosen = []
    for label in np.unique(y[pool]):
        class_pool = pool[y[pool] == label]
        chosen.extend(class_pool[:n])
    return np.asarray(chosen, dtype=int)


def fit_domain_network(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    input_mode: str,
    config: AFConfig,
    device: torch.device,
) -> nn.Module:
    if input_mode == "direction":
        extractor = DirectionFeatureExtractor(x_source.shape[1], config.embedding_dim)
    else:
        extractor = TabularFeatureExtractor(x_source.shape[1], config.embedding_dim)
    model = AFDomainNetwork(extractor, len(np.unique(y_source)), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    source = TensorDataset(torch.from_numpy(x_source), torch.from_numpy(y_source).long())
    target = TensorDataset(torch.from_numpy(x_target))
    source_loader = DataLoader(source, config.batch_size, shuffle=True, drop_last=False)
    target_loader = DataLoader(target, config.batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(config.pretrain_epochs):
        target_iterator = iter(target_loader)
        for source_x, source_y in source_loader:
            try:
                (target_x,) = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                (target_x,) = next(target_iterator)
            size = min(len(source_x), len(target_x))
            if size < 2:  # BatchNorm requires at least two rows in training.
                continue
            source_x, source_y = source_x[:size].to(device), source_y[:size].to(device)
            target_x = target_x[:size].to(device)
            combined = torch.cat((source_x, target_x))
            class_logits, domain_logits = model(combined)
            domain_y = torch.cat(
                (torch.zeros(size, dtype=torch.long), torch.ones(size, dtype=torch.long))
            ).to(device)
            # GRL supplies the negative lambda only to the extractor.  The
            # discriminator itself must minimize ordinary cross entropy.
            loss = criterion(class_logits[:size], source_y) + criterion(domain_logits, domain_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.extractor


def embeddings(extractor, x: np.ndarray, device: torch.device) -> np.ndarray:
    extractor.eval()
    with torch.no_grad():
        values = extractor(torch.from_numpy(x).to(device)).cpu().numpy()
    return normalize(values, norm="l2")


def metric_row(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run(args) -> None:
    config = AFConfig(seed=args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_source, source_labels, source_columns = read_csv(args.source, args.label_column)
    x_target, target_labels, target_columns = read_csv(args.target, args.label_column)
    if source_columns != target_columns:
        raise ValueError("source and target feature columns/order differ")

    encoder = LabelEncoder().fit(source_labels)
    unseen = sorted(set(target_labels) - set(encoder.classes_))
    if unseen:
        raise ValueError(f"target contains unseen labels: {unseen}")
    y_source, y_target = encoder.transform(source_labels), encoder.transform(target_labels)

    if args.input_mode == "direction":
        if not np.isin(x_source, (-1.0, 0.0, 1.0)).all() or not np.isin(x_target, (-1.0, 0.0, 1.0)).all():
            raise ValueError("direction mode accepts only -1, 0 and +1 packet-direction values")
    else:
        scaler = StandardScaler().fit(x_source)  # Never fit on target/test.
        x_source = scaler.transform(x_source).astype(np.float32)
        x_target = scaler.transform(x_target).astype(np.float32)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, audit = [], []
    method = "AF-SingleSource" if args.input_mode == "direction" else "AF-MLP-adapted"

    for fold in range(config.folds):
        fold_seed = config.seed + fold
        seed_everything(fold_seed)
        rng = np.random.default_rng(fold_seed)
        source_idx = select_per_class(y_source, config.source_per_class, rng)
        train_pool, test_idx = target_partition(y_target, config, rng)

        for n in config.n_shots:
            train_idx = n_shot_subset(train_pool, y_target, n)
            with ResourceMonitor(device) as fit_mem:
                started = time.perf_counter()
                extractor = fit_domain_network(
                    x_source[source_idx], y_source[source_idx], x_target[train_idx],
                    args.input_mode, config, device,
                )
                train_z = embeddings(extractor, x_target[train_idx], device)
                knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean")
                knn.fit(train_z, y_target[train_idx])
                fit_seconds = time.perf_counter() - started
            with ResourceMonitor(device) as infer_mem:
                started = time.perf_counter()
                test_z = embeddings(extractor, x_target[test_idx], device)
                predicted = knn.predict(test_z)
                predict_seconds = time.perf_counter() - started
            row = {
                "method": method, "protocol": f"F{fold + 1}",
                "fold": fold, "seed": fold_seed, "M": 25, "N": n, "T": 70,
                "n": len(test_idx),
            }
            row.update(metric_row(y_target[test_idx], predicted))
            row.update({
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fit_and_inference_seconds": fit_seconds + predict_seconds,
                "inference_us_per_row": 1e6 * predict_seconds / len(test_idx),
                "throughput_rows_per_second": len(test_idx) / predict_seconds,
                **resource_values(fit_mem, infer_mem),
            })
            rows.append(row)
            audit.append(
                {"fold": fold, "N": n, "source_rows": source_idx.tolist(),
                 "target_train_rows": train_idx.tolist(), "target_test_rows": test_idx.tolist(),
                 "overlap": int(np.intersect1d(train_idx, test_idx).size)}
            )

    results = pd.DataFrame(rows)
    results.to_csv(output / "fold_metrics.csv", index=False)
    summary = results.groupby(["method", "N"])[
        ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "fit_and_inference_seconds"]
    ].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {"config": asdict(config), "input_mode": args.input_mode, "device": str(device),
                "source": str(args.source), "target": str(args.target), "label_column": args.label_column}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_compact_results(results, method="AF")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--input-mode", choices=("direction", "tabular"), default="direction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("CDR_MLC/benchmarks/AF/outputs/af_single_source"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
