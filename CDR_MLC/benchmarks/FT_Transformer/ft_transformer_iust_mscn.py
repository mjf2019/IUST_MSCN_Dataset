"""Independent FT-Transformer benchmark for IUST_MSCN scenarios 1--7."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class NumericalTokenizer(nn.Module):
    def __init__(self, features: int, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, width))
        self.bias = nn.Parameter(torch.empty(features, width))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, features, classes, width, heads, layers, ffn, dropout):
        super().__init__()
        if width % heads:
            raise ValueError("--d-token must be divisible by --heads")
        self.tokenizer = NumericalTokenizer(features, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, classes)
        nn.init.normal_(self.cls, std=.02)

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls.expand(len(x), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(encoded[:, 0]))


def loader(x, y, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch, shuffle=shuffle, generator=generator,
    )


def loss_on(model, data_loader, criterion, device):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total += float(loss.item()) * len(y)
            count += len(y)
    return total / max(count, 1)


def fit(train_x, train_y, valid_x, valid_y, args, device):
    seed_everything(args.seed)
    model = FTTransformer(
        train_x.shape[1], len(APPLICATIONS), args.d_token, args.heads,
        args.layers, args.ffn, args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_loader = loader(train_x, train_y, args.batch_size, True, args.seed)
    valid_loader = loader(valid_x, valid_y, args.batch_size, False, args.seed)
    best_loss, best_state, stale, completed = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        current = loss_on(model, valid_loader, criterion, device)
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
        raise RuntimeError("FT-Transformer produced no valid checkpoint")
    model.load_state_dict(best_state)
    return model, completed, best_loss


def run(args):
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

        started = time.perf_counter()
        model, epochs, best_loss = fit(
            train_x, train_y, valid_x, valid_y, args, device
        )
        fit_seconds = time.perf_counter() - started
        prediction, predict_seconds = predict_batches(
            model, test_x, args.batch_size, device
        )
        rows.append({
            **definition, "seed": args.seed, "method": "FT-Transformer",
            "train_n": len(train), "validation_n": len(valid), "n": len(test),
            "features": len(features), "epochs": epochs,
            "best_validation_loss": best_loss,
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

    save_run(args.output, "FT-Transformer", rows, audits, {
        "data_dir": str(args.data_dir), "scenarios": args.scenarios,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed, "device": str(device),
        "architecture": {
            "d_token": args.d_token, "heads": args.heads,
            "layers": args.layers, "ffn": args.ffn, "dropout": args.dropout,
        },
        "optimizer": {"name": "AdamW", "lr": args.lr,
                      "weight_decay": args.weight_decay},
        "protocol": "chronological validation; immutable test; preprocessing fit on train",
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-token", type=int, default=64)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--ffn", type=int, default=128)
    p.add_argument("--dropout", type=float, default=.10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
