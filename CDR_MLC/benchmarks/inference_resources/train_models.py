"""Train Low-level models and save local artifacts for CPU inference tests.

Training may use CUDA.  Saved PyTorch modules are moved to CPU first, and the
artifact directory is gitignored.  No Medium test row is used for model
selection.  AF is the sole exception that intrinsically requires labeled
target adaptation; it receives an early, disjoint Medium calibration prefix.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import APPLICATIONS, DEFAULT_CANDIDATES, load_dataset  # noqa: E402
from benchmarks.deep_common import chronological_validation, labels  # noqa: E402
from common import (  # noqa: E402
    dump_artifact, fit_tabular_preprocessor, frame_sha256, write_json,
)
from compare_clean_valid import fit_rf  # noqa: E402
from confirmatory_rf_cdr_mlc import fit_original_cdr  # noqa: E402
from congestion_feature_cdr_mlc import DEFAULT_CONGESTION_FEATURES  # noqa: E402
from meta_stacked_cdr_mlc_leakage_safe import (  # noqa: E402
    MetaStackConfig, fit_meta_stacker,
)


METHODS = (
    "rf", "original-cdr", "mf-cdr", "1d-cnn", "ft-transformer",
    "graphsage", "scarf", "dfe", "af",
)


def _early_target_calibration(medium: pd.DataFrame, fraction: float):
    parts = []
    for sequence_id, group in medium.groupby("sequence_id", sort=False):
        ordered = group.sort_values(["timestamp", "source_row"], kind="stable")
        count = max(1, int(np.floor(len(ordered) * fraction)))
        if count >= len(ordered):
            raise ValueError(f"{sequence_id}: AF calibration consumes sequence")
        parts.append(ordered.iloc[:count].copy())
    return pd.concat(parts, ignore_index=True)


def _deep_preprocessing(low, validation_fraction):
    train, validation = chronological_validation(low, validation_fraction)
    features, imputer, scaler, arrays = fit_tabular_preprocessor(
        train, [validation]
    )
    return train, validation, features, imputer, scaler, arrays[0], arrays[1]


def _save(output: Path, method: str, artifact: dict, metadata: dict):
    path = output / f"{method}.joblib"
    dump_artifact(path, artifact)
    return {
        "method": method,
        "artifact": str(path),
        "artifact_bytes": int(path.stat().st_size),
        **metadata,
    }


def train_one(method, low, medium, args, device):
    started = time.perf_counter()
    base = {
        "method": method,
        "seed": args.seed,
        "train_level": "Low",
        "train_identity_sha256": frame_sha256(low),
    }

    if method == "rf":
        model = fit_rf(low, (), args.seed, args.rf_trees)
        artifact = {**base, "kind": method, "model": model}

    elif method in {"original-cdr", "mf-cdr"}:
        config = MetaStackConfig(
            window=args.window,
            congestion_window=args.congestion_window,
            congestion_features=tuple(args.congestion_features),
            expert_trees=args.expert_trees,
            utility_trees=args.utility_trees,
            meta_trees=args.meta_trees,
            random_state=args.seed,
        ).validate()
        model = (
            fit_original_cdr(low, config)
            if method == "original-cdr" else fit_meta_stacker(low, config)
        )
        artifact = {
            **base, "kind": method, "model": model,
            "config": asdict(config),
        }

    elif method == "1d-cnn":
        module = importlib.import_module(
            "benchmarks.1D_CNN.standard_1d_cnn_iust_mscn"
        )
        train, valid, features, imputer, scaler, train_x, valid_x = (
            _deep_preprocessing(low, args.validation_fraction)
        )
        namespace = SimpleNamespace(
            seed=args.seed, batch_size=args.batch_size,
            epochs=args.epochs, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, patience=args.patience,
            min_delta=0.0,
        )
        model, epochs, best_loss = module.fit_model(
            train_x, labels(train), valid_x, labels(valid), namespace, device
        )
        artifact = {
            **base, "kind": method, "model": model.cpu(),
            "features": features, "imputer": imputer, "scaler": scaler,
            "batch_size": args.batch_size, "epochs": epochs,
            "best_validation_loss": best_loss,
        }

    elif method == "ft-transformer":
        module = importlib.import_module(
            "benchmarks.FT_Transformer.ft_transformer_iust_mscn"
        )
        train, valid, features, imputer, scaler, train_x, valid_x = (
            _deep_preprocessing(low, args.validation_fraction)
        )
        namespace = SimpleNamespace(
            seed=args.seed, batch_size=args.batch_size, epochs=args.epochs,
            d_token=args.d_token, heads=args.heads, layers=args.layers,
            ffn=args.ffn, dropout=args.dropout, lr=args.learning_rate,
            weight_decay=args.weight_decay, patience=args.patience,
            min_delta=0.0,
        )
        model, epochs, best_loss = module.fit(
            train_x, labels(train), valid_x, labels(valid), namespace, device
        )
        artifact = {
            **base, "kind": method, "model": model.cpu(),
            "features": features, "imputer": imputer, "scaler": scaler,
            "batch_size": args.batch_size, "epochs": epochs,
            "best_validation_loss": best_loss,
        }

    elif method == "graphsage":
        module = importlib.import_module(
            "benchmarks.GraphSAGE.graphsage_iust_mscn"
        )
        train, valid, features, imputer, scaler, train_x, valid_x = (
            _deep_preprocessing(low, args.validation_fraction)
        )
        train, valid = train.reset_index(drop=True), valid.reset_index(drop=True)
        namespace = SimpleNamespace(
            seed=args.seed, hidden=args.graph_hidden,
            layers=args.graph_layers, neighbors=args.graph_neighbors,
            dropout=args.graph_dropout, lr=args.learning_rate,
            weight_decay=args.weight_decay, epochs=args.graph_epochs,
            patience=args.graph_patience, min_delta=0.0,
        )
        model, epochs, best_loss = module.fit(
            train_x, labels(train), train,
            valid_x, labels(valid), valid, namespace, device,
        )
        artifact = {
            **base, "kind": method, "model": model.cpu(),
            "features": features, "imputer": imputer, "scaler": scaler,
            "neighbors": args.graph_neighbors, "epochs": epochs,
            "best_validation_loss": best_loss,
        }

    elif method == "scarf":
        module = importlib.import_module("benchmarks.SCARF.scarf_iust_mscn")
        train, valid, features, imputer, scaler, train_x, valid_x = (
            _deep_preprocessing(low, args.validation_fraction)
        )
        namespace = SimpleNamespace(
            seed=args.seed, hidden=args.scarf_hidden,
            embedding=args.scarf_embedding,
            projection=args.scarf_projection,
            dropout=args.dropout, pretrain_lr=args.learning_rate,
            finetune_lr=args.learning_rate, weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            pretrain_epochs=args.scarf_pretrain_epochs,
            finetune_epochs=args.epochs, corruption=args.scarf_corruption,
            temperature=args.scarf_temperature, patience=args.patience,
            min_delta=0.0,
        )
        encoder = module.pretrain(train_x, namespace, device)
        model, epochs, best_loss = module.finetune(
            encoder, train_x, labels(train), valid_x, labels(valid),
            namespace, device,
        )
        artifact = {
            **base, "kind": method, "model": model.cpu(),
            "features": features, "imputer": imputer, "scaler": scaler,
            "batch_size": args.batch_size, "epochs": epochs,
            "best_validation_loss": best_loss,
        }

    elif method == "dfe":
        module = importlib.import_module("benchmarks.DFE.dfe_iust_mscn")
        config = replace(module.DFEConfig(), epochs=args.dfe_epochs, seed=args.seed)
        train, valid = module.chronological_source_split(low)
        transform = module.FlowImageTransform(config.input_fields).fit(train)
        train_x, valid_x = transform.transform(train), transform.transform(valid)
        encoder = LabelEncoder().fit(APPLICATIONS)
        train_y = encoder.transform(train.traffic_label.astype(str))
        valid_y = encoder.transform(valid.traffic_label.astype(str))
        model, training_audit = module.train_backbone(
            train_x, train_y, valid_x, valid_y, config, device
        )
        template_x, template_y = module.select_templates(
            train_x, train_y, config.templates_per_class, args.seed
        )
        k, trials = module.choose_k(
            model, template_x, template_y, valid_x, valid_y, device
        )
        template_z = module.embed(model, template_x, device)
        artifact = {
            **base, "kind": method, "model": model.cpu(),
            "transform": transform, "label_encoder": encoder,
            "template_embeddings": template_z,
            "template_labels": template_y, "k": k,
            "training_audit": training_audit, "k_trials": trials,
        }

    elif method == "af":
        module = importlib.import_module("benchmarks.AF.af_single_source")
        calibration = _early_target_calibration(
            medium, args.af_calibration_fraction
        )
        features, imputer, scaler, arrays = fit_tabular_preprocessor(
            low, [calibration]
        )
        source_x, calibration_x = arrays
        encoder = LabelEncoder().fit(low.traffic_label.astype(str))
        source_y = encoder.transform(low.traffic_label.astype(str))
        calibration_y = encoder.transform(calibration.traffic_label.astype(str))
        config = replace(
            module.AFConfig(), folds=1, seed=args.seed,
            pretrain_epochs=args.af_epochs, batch_size=args.af_batch_size,
        )
        extractor = module.fit_domain_network(
            source_x, source_y, calibration_x, "tabular", config, device
        )
        calibration_z = module.embeddings(extractor, calibration_x, device)
        counts = pd.Series(calibration_y).value_counts()
        if len(counts) != len(encoder.classes_):
            raise ValueError("AF calibration prefix does not cover every class")
        neighbors = int(counts.min())
        classifier = KNeighborsClassifier(
            n_neighbors=neighbors, metric="euclidean"
        ).fit(calibration_z, calibration_y)
        artifact = {
            **base, "kind": method, "model": extractor.cpu(),
            "classifier": classifier, "label_encoder": encoder,
            "features": features, "imputer": imputer, "scaler": scaler,
            "calibration_fraction": args.af_calibration_fraction,
            "calibration_rows": len(calibration),
            "calibration_identity_sha256": frame_sha256(calibration),
            "neighbors": neighbors,
        }
    else:
        raise ValueError(f"unsupported method: {method}")

    seconds = time.perf_counter() - started
    metadata = _save(
        args.output, method, artifact,
        {"training_seconds": seconds, "training_device": str(device)},
    )
    print(f"saved {method}: {metadata['artifact']} ({seconds:.2f}s)", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=CDR_MLC / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/models")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--congestion-window", type=int, default=50)
    parser.add_argument("--congestion-features", nargs="+", default=list(DEFAULT_CONGESTION_FEATURES))
    parser.add_argument("--expert-trees", type=int, default=20)
    parser.add_argument("--utility-trees", type=int, default=10)
    parser.add_argument("--meta-trees", type=int, default=20)
    parser.add_argument("--rf-trees", type=int, default=110)
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.10)
    parser.add_argument("--graph-hidden", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-neighbors", type=int, default=3)
    parser.add_argument("--graph-dropout", type=float, default=.20)
    parser.add_argument("--graph-epochs", type=int, default=200)
    parser.add_argument("--graph-patience", type=int, default=20)
    parser.add_argument("--scarf-hidden", type=int, default=256)
    parser.add_argument("--scarf-embedding", type=int, default=128)
    parser.add_argument("--scarf-projection", type=int, default=128)
    parser.add_argument("--scarf-pretrain-epochs", type=int, default=100)
    parser.add_argument("--scarf-corruption", type=float, default=.60)
    parser.add_argument("--scarf-temperature", type=float, default=1.0)
    parser.add_argument("--dfe-epochs", type=int, default=200)
    parser.add_argument("--af-epochs", type=int, default=30)
    parser.add_argument("--af-batch-size", type=int, default=32)
    parser.add_argument("--af-calibration-fraction", type=float, default=.01)
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    data, audit = load_dataset(args.data_dir, tuple(DEFAULT_CANDIDATES))
    low = data[data.congestion_level.eq("Low")].copy().reset_index(drop=True)
    medium = data[data.congestion_level.eq("Medium")].copy().reset_index(drop=True)
    if low.empty or medium.empty:
        raise ValueError("Low training or Medium AF calibration data is empty")
    args.output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output / "input_audit.csv", index=False)

    entries = []
    for method in args.methods:
        entries.append(train_one(method, low, medium, args, device))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    index_path = args.output / "index.json"
    if index_path.is_file():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        retained = {
            item["method"]: item for item in previous.get("methods", [])
            if item["method"] not in set(args.methods)
        }
        retained.update({item["method"]: item for item in entries})
        entries = [retained[name] for name in METHODS if name in retained]
    write_json(index_path, {
        "data_dir": str(args.data_dir),
        "training_level": "Low",
        "training_rows": len(low),
        "training_identity_sha256": frame_sha256(low),
        "seed": args.seed,
        "models_committed_to_git": False,
        "methods": entries,
    })


if __name__ == "__main__":
    main()
