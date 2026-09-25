"""Independent causal GraphSAGE benchmark for IUST_MSCN scenarios 1--7.

Each flow record is a node.  A node receives messages only from preceding
records in the same capture sequence, so test-graph construction is causal and
never creates train/test edges.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import APPLICATIONS
from benchmarks.console_output import ResourceMonitor, resource_values
from benchmarks.deep_common import (
    PROTOCOL_IDS, chronological_validation, class_weights, evaluations,
    labels, load_clean_valid, matrices, record_ids, save_run, scores,
    seed_everything,
)


def causal_adjacency(frame: pd.DataFrame, neighbors: int, device):
    if neighbors < 1:
        raise ValueError("--neighbors must be positive")
    frame = frame.reset_index(drop=True)
    rows, cols = [np.arange(len(frame))], [np.arange(len(frame))]
    for _, group in frame.groupby("sequence_id", sort=False):
        ordered = group.sort_values(["timestamp", "source_row"], kind="stable")
        positions = ordered.index.to_numpy(dtype=np.int64)
        for offset in range(1, neighbors + 1):
            if len(positions) > offset:
                rows.append(positions[offset:])
                cols.append(positions[:-offset])
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    degree = np.bincount(row, minlength=len(frame)).astype(np.float32)
    values = 1.0 / degree[row]
    indices = torch.tensor(np.vstack([row, col]), dtype=torch.long, device=device)
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, weights, (len(frame), len(frame)), device=device
    ).coalesce()


class SAGEConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(2 * input_dim, output_dim)

    def forward(self, x, adjacency):
        neighborhood = torch.sparse.mm(adjacency, x)
        return self.linear(torch.cat([x, neighborhood], dim=1))


class GraphSAGE(nn.Module):
    def __init__(self, features, hidden, classes, layers, dropout):
        super().__init__()
        dimensions = [features] + [hidden] * layers
        self.layers = nn.ModuleList([
            SAGEConv(dimensions[i], dimensions[i + 1])
            for i in range(layers)
        ])
        self.output = nn.Linear(hidden, classes)
        self.dropout = dropout

    def forward(self, x, adjacency):
        for layer in self.layers:
            x = F.relu(layer(x, adjacency))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output(x)


def validation_loss(model, x, y, adjacency, criterion):
    model.eval()
    with torch.no_grad():
        return float(criterion(model(x, adjacency), y).item())


def fit(train_x, train_y, train_frame, valid_x, valid_y, valid_frame, args, device):
    seed_everything(args.seed)
    train_tensor = torch.from_numpy(train_x).to(device)
    valid_tensor = torch.from_numpy(valid_x).to(device)
    train_labels = torch.from_numpy(train_y).to(device)
    valid_labels = torch.from_numpy(valid_y).to(device)
    train_adj = causal_adjacency(train_frame, args.neighbors, device)
    valid_adj = causal_adjacency(valid_frame, args.neighbors, device)

    model = GraphSAGE(
        train_x.shape[1], args.hidden, len(APPLICATIONS),
        args.layers, args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_y, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss, best_state, stale, completed = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_tensor, train_adj), train_labels)
        loss.backward()
        optimizer.step()
        current = validation_loss(
            model, valid_tensor, valid_labels, valid_adj, criterion
        )
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
        raise RuntimeError("GraphSAGE produced no valid checkpoint")
    model.load_state_dict(best_state)
    return model, completed, best_loss


def predict(model, x, frame, neighbors, device):
    started = time.perf_counter()
    adjacency = causal_adjacency(frame, neighbors, device)
    values = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(values, adjacency).argmax(1).cpu().numpy()
    return prediction, time.perf_counter() - started


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
        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)
        features, arrays = matrices(train, [valid, test])
        train_x, valid_x, test_x = arrays
        train_y, valid_y, test_y = labels(train), labels(valid), labels(test)

        with ResourceMonitor(device) as fit_mem:
            started = time.perf_counter()
            model, epochs, best_loss = fit(
                train_x, train_y, train, valid_x, valid_y, valid,
                args, device,
            )
            fit_seconds = time.perf_counter() - started
        with ResourceMonitor(device) as infer_mem:
            prediction, predict_seconds = predict(
                model, test_x, test, args.neighbors, device
            )
        rows.append({
            **definition, "seed": args.seed, "method": "GraphSAGE",
            "train_n": len(train), "validation_n": len(valid), "n": len(test),
            "features": len(features), "epochs": epochs,
            "best_validation_loss": best_loss, "neighbors": args.neighbors,
            **scores(test_y, prediction),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "inference_us_per_row": 1e6 * predict_seconds / len(test),
            "throughput_rows_per_second": len(test) / predict_seconds,
            "parameters": sum(p.numel() for p in model.parameters()),
            **resource_values(fit_mem, infer_mem),
        })
        audits[definition["protocol"]] = {
            "development_n": len(development), "train_n": len(train),
            "validation_n": len(valid), "test_n": len(test),
            "feature_count": len(features), "features": features,
            "development_test_overlap": len(record_ids(development) & record_ids(test)),
            "graph_policy": (
                f"self plus {args.neighbors} preceding records per sequence; "
                "separate train, validation, and test graphs"
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_run(args.output, "GraphSAGE", rows, audits, {
        "data_dir": str(args.data_dir), "scenarios": args.scenarios,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed, "device": str(device),
        "architecture": {
            "hidden": args.hidden, "layers": args.layers,
            "dropout": args.dropout, "causal_neighbors": args.neighbors,
        },
        "optimizer": {"name": "AdamW", "lr": args.lr,
                      "weight_decay": args.weight_decay},
        "protocol": "inductive causal graphs built independently per partition",
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=3)
    p.add_argument("--dropout", type=float, default=.20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
