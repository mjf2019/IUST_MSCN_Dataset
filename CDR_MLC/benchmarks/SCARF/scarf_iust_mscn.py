"""Independent SCARF self-supervised benchmark for scenarios 1--7."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import APPLICATIONS
from benchmarks.deep_common import (
    PROTOCOL_IDS, chronological_validation, class_weights, evaluations,
    labels, load_clean_valid, matrices, predict_batches, record_ids,
    save_run, scores, seed_everything,
)


class Encoder(nn.Module):
    def __init__(self, features, hidden, embedding, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding),
        )

    def forward(self, x):
        return self.network(x)


class SCARFPretrainer(nn.Module):
    def __init__(self, features, hidden, embedding, projection, dropout):
        super().__init__()
        self.encoder = Encoder(features, hidden, embedding, dropout)
        self.projector = nn.Sequential(
            nn.Linear(embedding, projection), nn.ReLU(),
            nn.Linear(projection, projection),
        )

    def forward(self, x):
        return self.projector(self.encoder(x))


class SCARFClassifier(nn.Module):
    def __init__(self, encoder, embedding, classes):
        super().__init__()
        self.encoder = encoder
        self.output = nn.Linear(embedding, classes)

    def forward(self, x):
        return self.output(self.encoder(x))


def supervised_loader(x, y, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    drop_last = bool(shuffle and len(x) > 1 and len(x) % batch == 1)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch, shuffle=shuffle, drop_last=drop_last,
        generator=generator,
    )


def corruption(x, probability):
    donor = x[torch.randperm(len(x), device=x.device)]
    mask = torch.rand_like(x) < probability
    return torch.where(mask, donor, x)


def nt_xent(clean, corrupted, temperature):
    clean = F.normalize(clean, dim=1)
    corrupted = F.normalize(corrupted, dim=1)
    z = torch.cat([clean, corrupted], dim=0)
    logits = z @ z.T / temperature
    count = len(clean)
    logits.fill_diagonal_(-1e9)
    targets = torch.cat([
        torch.arange(count, 2 * count, device=z.device),
        torch.arange(0, count, device=z.device),
    ])
    return F.cross_entropy(logits, targets)


def pretrain(train_x, args, device):
    seed_everything(args.seed)
    model = SCARFPretrainer(
        train_x.shape[1], args.hidden, args.embedding,
        args.projection, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=generator,
    )
    model.train()
    for _ in range(args.pretrain_epochs):
        for (x,) in data_loader:
            if len(x) < 2:
                continue
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nt_xent(
                model(x), model(corruption(x, args.corruption)),
                args.temperature,
            )
            loss.backward()
            optimizer.step()
    return model.encoder


def validation_loss(model, loader, criterion, device):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total += float(loss.item()) * len(y)
            count += len(y)
    return total / max(count, 1)


def finetune(encoder, train_x, train_y, valid_x, valid_y, args, device):
    model = SCARFClassifier(encoder, args.embedding, len(APPLICATIONS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )
    train_loader = supervised_loader(
        train_x, train_y, args.batch_size, True, args.seed
    )
    valid_loader = supervised_loader(
        valid_x, valid_y, args.batch_size, False, args.seed
    )
    best_loss, best_state, stale, completed = float("inf"), None, 0, 0
    for epoch in range(args.finetune_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        current = validation_loss(model, valid_loader, criterion, device)
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
        raise RuntimeError("SCARF produced no valid supervised checkpoint")
    model.load_state_dict(best_state)
    return model, completed, best_loss


def run(args):
    if not 0 < args.corruption < 1:
        raise ValueError("--corruption must be in (0,1)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data, input_audit = load_clean_valid(args.data_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(args.output / "input_audit.csv", index=False)
    rows, audits = [], {}

    for development, test, definition in evaluations(
        data, args.scenarios, args.train_fraction
    ):
        train, valid = chronological_validation(development, args.validation_fraction)
        features, arrays = matrices(train, [valid, test])
        train_x, valid_x, test_x = arrays
        train_y, valid_y, test_y = labels(train), labels(valid), labels(test)

        seed_everything(args.seed)
        started = time.perf_counter()
        encoder = pretrain(train_x, args, device)
        model, epochs, best_loss = finetune(
            encoder, train_x, train_y, valid_x, valid_y, args, device
        )
        fit_seconds = time.perf_counter() - started
        prediction, predict_seconds = predict_batches(
            model, test_x, args.batch_size, device
        )
        rows.append({
            **definition, "seed": args.seed, "method": "SCARF",
            "train_n": len(train), "validation_n": len(valid), "n": len(test),
            "features": len(features), "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": epochs, "best_validation_loss": best_loss,
            **scores(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
        })
        audits[definition["protocol"]] = {
            "development_n": len(development), "train_n": len(train),
            "validation_n": len(valid), "test_n": len(test),
            "feature_count": len(features), "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_run(args.output, "SCARF", rows, audits, {
        "data_dir": str(args.data_dir), "scenarios": args.scenarios,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed, "device": str(device),
        "architecture": {
            "hidden": args.hidden, "embedding": args.embedding,
            "projection": args.projection, "dropout": args.dropout,
        },
        "self_supervision": {
            "pretrain_epochs": args.pretrain_epochs,
            "corruption": args.corruption, "temperature": args.temperature,
        },
        "protocol": "SCARF pretraining uses train features only; immutable test",
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid")
    p.add_argument("--output", type=Path, default=HERE / "outputs")
    p.add_argument("--scenarios", nargs="+", choices=PROTOCOL_IDS,
                   default=list(PROTOCOL_IDS))
    p.add_argument("--train-fraction", type=float, default=.80)
    p.add_argument("--validation-fraction", type=float, default=.10)
    p.add_argument("--pretrain-epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--projection", type=int, default=128)
    p.add_argument("--dropout", type=float, default=.10)
    p.add_argument("--corruption", type=float, default=.60)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
