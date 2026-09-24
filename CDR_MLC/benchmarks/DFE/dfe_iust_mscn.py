"""Leakage-safe DFE adaptation for IUST_MSCN Clean-Valid.

The implementation follows Wang et al., DFE: Deep Flow Embedding for Robust
Network Traffic Classification (TNSE 2025): a 13-convolution residual
backbone, feature-compression loss, constrained triplet loss, and a template
library classified by L2 k-nearest-neighbour majority voting.

IUST_MSCN does not expose the paper's exact 81 flow fields.  Consequently the
available Clean-Valid numeric fields are padded (or deterministically limited)
to 81 and reshaped to 9x9.  Results are named DFE-adapted, not exact DFE.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
BENCHMARKS = HERE.parent
CDR_MLC = BENCHMARKS.parent
MODULE_ROOT = CDR_MLC
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from adaptive_cdr_mlc import DEFAULT_CANDIDATES, FORBIDDEN, load_dataset, select_classifier_columns  # noqa: E402
from compare_clean_valid import APPLICATIONS, SCENARIOS  # noqa: E402
from benchmarks.console_output import print_compact_results  # noqa: E402


@dataclass(frozen=True)
class DFEConfig:
    input_fields: int = 81
    embedding_dim: int = 256
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 200
    patience: int = 10
    alpha1: float = 0.5
    alpha2: float = 2.0
    eta: float = 0.5
    lambda_ntc: float = 1.0
    betas: tuple[float, ...] = (.01, .02, .03, .04, .05, .05)
    templates_per_class: int = 20
    seed: int = 42


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DFEBackbone(nn.Module):
    """Figure-2 backbone: 13 3x3 convolutions and three stride-2 layers."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        specs = [
            (1, 64, 1), (64, 64, 1),
            (64, 64, 2), (64, 64, 1), (64, 64, 1),
            (64, 128, 2), (128, 128, 1), (128, 128, 1),
            (128, 256, 2), (256, 256, 1), (256, 256, 1),
            (256, 256, 1), (256, 256, 1),
        ]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            ) for cin, cout, stride in specs
        ])
        # Residual endpoints correspond to the six feature stages in Fig. 2.
        self.stage_ends = (1, 4, 7, 9, 11, 12)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = [x]
        residual = None
        for index, layer in enumerate(self.layers):
            output = layer(x)
            if residual is not None and residual.shape == output.shape:
                output = output + residual
            x = output
            if index in self.stage_ends:
                features.append(x)
                residual = x
        pooled = self.pool(x).flatten(1)
        embedding = self.fc2(self.fc1(pooled))
        if return_features:
            return embedding, features[:7]
        return embedding


class GaussianFeatureCompression(nn.Module):
    """Differentiable scalar-Gaussian implementation of equations (6)-(8).

    The paper does not define a tensor estimator for each intermediate map.
    Each sample/layer is therefore represented by its mean and diagonal scalar
    variance.  Equation (7) is then evaluated with d=1, avoiding the notebook's
    unregistered random projections created inside every forward pass.
    """

    def __init__(self, betas: tuple[float, ...]):
        super().__init__()
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))

    @staticmethod
    def summaries(feature: torch.Tensor):
        flattened = feature.flatten(1)
        return flattened.mean(1), flattened.var(1, unbiased=False).clamp_min(1e-6)

    @staticmethod
    def pairwise_kl(mu_p, var_p, mu_q, var_q):
        return .5 * (
            torch.log(var_q[None, :] / var_p[:, None])
            + (var_p[:, None] + (mu_p[:, None] - mu_q[None, :]).square()) / var_q[None, :]
            - 1.0
        )

    def mutual_information_bound(self, left, right):
        mu_l, var_l = self.summaries(left)
        mu_r, var_r = self.summaries(right)
        kl = self.pairwise_kl(mu_l, var_l, mu_r, var_r)
        # -1/N sum_p log(1/N sum_q exp(-KL)), equation (6).
        return -(torch.logsumexp(-kl, dim=1) - math.log(kl.shape[1])).mean()

    def forward(self, features: list[torch.Tensor]):
        terms = [
            self.betas[i] * self.mutual_information_bound(features[i], features[i + 1])
            for i in range(min(len(features) - 1, len(self.betas)))
        ]
        return torch.stack(terms).sum()


class ConstrainedTripletLoss(nn.Module):
    """Equation (12): inter-class margin plus bounded intra-class distance."""

    def __init__(self, alpha1: float, alpha2: float, eta: float):
        super().__init__()
        self.alpha1, self.alpha2, self.eta = alpha1, alpha2, eta

    def forward(self, anchor, positive, negative):
        positive_distance = torch.linalg.vector_norm(anchor - positive, dim=1)
        negative_distance = torch.linalg.vector_norm(anchor - negative, dim=1)
        inter = F.relu(positive_distance - negative_distance + self.alpha1)
        intra = F.relu(positive_distance - self.alpha2)
        return inter.mean() + self.eta * intra.mean()


class OnlineTriplets(Dataset):
    """One source-only triplet per training anchor; resampled each epoch."""

    def __init__(self, x: np.ndarray, y: np.ndarray, seed: int):
        self.x = x
        self.y = y
        self.seed = seed
        self.epoch = 0
        self.class_indices = {label: np.flatnonzero(y == label) for label in np.unique(y)}
        if any(len(indices) < 2 for indices in self.class_indices.values()):
            raise ValueError("DFE needs at least two training samples per class")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * len(self.x) + index)
        label = self.y[index]
        positive_pool = self.class_indices[label]
        positive = index
        while positive == index:
            positive = int(rng.choice(positive_pool))
        negative_label = rng.choice([item for item in self.class_indices if item != label])
        negative = int(rng.choice(self.class_indices[negative_label]))
        return (
            torch.from_numpy(self.x[index]),
            torch.from_numpy(self.x[positive]),
            torch.from_numpy(self.x[negative]),
        )


def chronological_source_split(frame: pd.DataFrame, fraction: float = .8):
    train, validation = [], []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * fraction)
        if not 1 < cut < len(group):
            raise ValueError("source capture is too short for 80/20 split")
        train.append(group.iloc[:cut])
        validation.append(group.iloc[cut:])
    return pd.concat(train, ignore_index=True), pd.concat(validation, ignore_index=True)


def fixed_tail(frame: pd.DataFrame, fraction: float, test_fraction: float):
    calibration, test, audit = [], [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        calibration_end = int(len(group) * fraction)
        test_start = int(len(group) * (1 - test_fraction))
        if calibration_end > test_start:
            raise ValueError(f"{sequence_id}: calibration overlaps fixed test")
        calibration.append(group.iloc[:calibration_end])
        test.append(group.iloc[test_start:])
        audit.append({"sequence_id": sequence_id, "rows": len(group),
                      "calibration_rows": calibration_end,
                      "unused_rows": test_start - calibration_end,
                      "test_rows": len(group) - test_start})
    return (
        pd.concat(calibration, ignore_index=True) if fraction > 0 else frame.iloc[:0].copy(),
        pd.concat(test, ignore_index=True), audit,
    )


class FlowImageTransform:
    """Train-only imputation/scaling followed by deterministic 9x9 mapping."""

    def __init__(self, fields: int = 81):
        self.fields = fields
        self.columns: list[str] = []
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler(feature_range=(0, 1), clip=True)

    def fit(self, frame: pd.DataFrame):
        numeric, _ = select_classifier_columns(frame, route_features=[])
        self.columns = sorted(column for column in numeric if column not in FORBIDDEN)[:self.fields]
        if not self.columns:
            raise ValueError("no numeric Clean-Valid input fields")
        imputed = self.imputer.fit_transform(frame[self.columns])
        self.scaler.fit(imputed)
        return self

    def transform(self, frame: pd.DataFrame):
        values = self.scaler.transform(self.imputer.transform(frame[self.columns]))
        if values.shape[1] < self.fields:
            values = np.pad(values, ((0, 0), (0, self.fields - values.shape[1])))
        return values[:, :self.fields].reshape(-1, 1, 9, 9).astype(np.float32)


def train_backbone(x_train, y_train, x_validation, y_validation, config, device):
    dataset = OnlineTriplets(x_train, y_train, config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    model = DFEBackbone(config.embedding_dim).to(device)
    compressor = GaussianFeatureCompression(config.betas).to(device)
    triplet = ConstrainedTripletLoss(config.alpha1, config.alpha2, config.eta)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    best_state, best_score, stale = None, -np.inf, 0

    for epoch in range(config.epochs):
        dataset.set_epoch(epoch)
        model.train()
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_z, features = model(anchor, return_features=True)
            positive_z, negative_z = model(positive), model(negative)
            loss = compressor(features) + config.lambda_ntc * triplet(anchor_z, positive_z, negative_z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Early stopping is source-validation only; target is never inspected.
        library_x, library_y = select_templates(x_train, y_train, config.templates_per_class, config.seed)
        predicted = template_predict(model, library_x, library_y, x_validation, 1, device)
        score = f1_score(y_validation, predicted, average="macro", zero_division=0)
        if score > best_score + 1e-8:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("DFE training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return model, {"epochs_completed": epoch + 1, "best_source_validation_macro_f1": best_score}


def select_templates(x, y, count: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        selected.extend(rng.choice(indices, min(count, len(indices)), replace=False))
    selected = np.asarray(selected, dtype=int)
    return x[selected], y[selected]


def embed(model, x, device, batch_size=512):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu())
    return torch.cat(outputs).numpy()


def vote(neighbor_labels: np.ndarray, distances: np.ndarray):
    predictions = []
    for labels, dists in zip(neighbor_labels, distances):
        values, counts = np.unique(labels, return_counts=True)
        winners = values[counts == counts.max()]
        if len(winners) == 1:
            predictions.append(winners[0])
        else:
            predictions.append(min(winners, key=lambda label: dists[labels == label].mean()))
    return np.asarray(predictions)


def template_predict(model, template_x, template_y, query_x, k, device):
    templates, queries = embed(model, template_x, device), embed(model, query_x, device)
    distances = np.linalg.norm(queries[:, None, :] - templates[None, :, :], axis=2)
    k = min(k, len(template_y))
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest, axis=1)
    return vote(template_y[nearest], nearest_distances)


def choose_k(model, template_x, template_y, validation_x, validation_y, device):
    trials = []
    for k in (1, 3, 5, 7):
        prediction = template_predict(model, template_x, template_y, validation_x, k, device)
        trials.append((f1_score(validation_y, prediction, average="macro", zero_division=0), k))
    return max(trials, key=lambda item: (item[0], -item[1]))[1], trials


def metrics(truth, prediction):
    return {
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, prediction, average="weighted", zero_division=0),
    }


def run(args):
    config = DFEConfig(epochs=args.epochs, seed=args.seed)
    seed_all(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    encoder = LabelEncoder().fit(APPLICATIONS)
    rows, audits, trained = [], {}, {}

    for scenario in args.scenarios:
        source_level, target_level = SCENARIOS[scenario]
        if source_level not in trained:
            source = data[data.congestion_level.eq(source_level)].copy()
            source_train, source_validation = chronological_source_split(source)
            transform = FlowImageTransform(config.input_fields).fit(source_train)
            x_train, x_validation = transform.transform(source_train), transform.transform(source_validation)
            y_train = encoder.transform(source_train.traffic_label)
            y_validation = encoder.transform(source_validation.traffic_label)
            started = time.perf_counter()
            model, training_audit = train_backbone(
                x_train, y_train, x_validation, y_validation, config, device
            )
            template_x, template_y = select_templates(
                x_train, y_train, config.templates_per_class, config.seed
            )
            k, k_trials = choose_k(
                model, template_x, template_y, x_validation, y_validation, device
            )
            trained[source_level] = {
                "model": model, "transform": transform,
                "source_templates_x": template_x, "source_templates_y": template_y,
                "k": k, "fit_seconds": time.perf_counter() - started,
            }
            audits[f"source:{source_level}"] = {
                **training_audit, "selected_k": k, "k_trials": k_trials,
                "features": transform.columns, "training_rows": len(source_train),
                "validation_rows": len(source_validation),
            }

        fitted = trained[source_level]
        target = data[data.congestion_level.eq(target_level)].copy()
        for fraction in args.fractions:
            calibration, test, split_audit = fixed_tail(target, fraction, args.test_fraction)
            test_x = fitted["transform"].transform(test)
            truth = encoder.transform(test.traffic_label)
            if fraction == 0:
                library_x = fitted["source_templates_x"]
                library_y = fitted["source_templates_y"]
                adaptation_templates = 0
            else:
                calibration_x = fitted["transform"].transform(calibration)
                calibration_y = encoder.transform(calibration.traffic_label)
                # Updating the template library does not retrain the backbone.
                library_x, library_y = select_templates(
                    calibration_x, calibration_y, config.templates_per_class,
                    config.seed + int(round(fraction * 10_000)),
                )
                adaptation_templates = len(library_y)
            started = time.perf_counter()
            prediction = template_predict(
                fitted["model"], library_x, library_y, test_x,
                fitted["k"], device,
            )
            rows.append({
                "adaptation_fraction": fraction, "fixed_test_fraction": args.test_fraction,
                "scenario": scenario, "source": source_level, "target": target_level,
                "method": "DFE-adapted", "n": len(test),
                "adaptation_pool_rows": len(calibration),
                "adaptation_template_rows": adaptation_templates,
                "templates_total": len(library_y), "k": fitted["k"],
                **metrics(truth, prediction),
                "inference_seconds": time.perf_counter() - started,
            })
            audits[f"scenario:{scenario}:fraction:{fraction}"] = {
                "split": split_audit,
                "calibration_test_overlap": int(len(set(zip(calibration.source_file, calibration.source_row)) &
                                                    set(zip(test.source_file, test.source_row)))),
            }

    result = pd.DataFrame(rows).sort_values(["adaptation_fraction", "scenario"])
    result.to_csv(args.output / "dfe_iust_mscn_summary.csv", index=False)
    (args.output / "audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "method": "DFE-adapted", "config": asdict(config), "device": str(device),
        "fractions": args.fractions, "test_fraction": args.test_fraction,
        "scenarios": args.scenarios,
        "paper_deviations": [
            "Clean-Valid numeric fields replace the paper-specific 81 flow fields",
            "available fields are padded/limited to a deterministic 9x9 image",
            "source chronology replaces the paper's random 80/20 split",
            "k is selected only on source validation because the paper does not report k",
        ],
        "leakage_controls": [
            "imputer and min-max scaler fit source-training only",
            "early stopping and k selection use source-validation only",
            "target fixed test is never used for training, selection, scaling, or templates",
        ],
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print_compact_results(
        result, method="DFE", calibration_column="adaptation_pool_rows",
        seconds_column="inference_seconds",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=MODULE_ROOT / "DATASETS/CDR-MLC/Clean_Valid")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/iust_mscn")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0, .01, .05, .10, .20])
    parser.add_argument("--test-fraction", type=float, default=.20)
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if any(f < 0 or f > 1 - args.test_fraction for f in args.fractions):
        parser.error("adaptation fractions must not overlap the fixed test tail")
    return args


if __name__ == "__main__":
    run(parse_args())
