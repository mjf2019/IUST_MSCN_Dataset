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
    select_classifier_columns, trend_frame,
)
from compare_clean_valid import SCENARIOS, TIMING
from congestion_feature_cdr_mlc import (
    CongestionRouterConfig, congestion_feature_frame,
)
from mixed_level_protocols_leakage_safe import PROTOCOLS, build_protocol
from benchmarks.console_output import print_compact_results
from quicext25_common import (
    MONTHS as QUIC_MONTHS,
    build_protocol as build_quic_protocol,
    load_class_spec as load_quic_class_spec,
    load_months as load_quic_months,
    numeric_model_features as quic_numeric_features,
    record_identity as quic_record_identity,
)

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


def mf_context_eligible_test(
    frame: pd.DataFrame, window: int = 3, congestion_window: int = 50
) -> pd.DataFrame:
    """Return the exact causal test rows eligible for MF-CDR-MLC."""
    base = trend_frame(frame, TIMING, window)
    context, _ = congestion_feature_frame(
        frame,
        CongestionRouterConfig(
            window=congestion_window,
            features=tuple(TIMING),
            expert_trees=1,
        ).validate(),
    )
    eligible = base.index[base.index.isin(context.index)]
    result = frame.loc[eligible].copy()
    if result.empty:
        raise ValueError("no MF-CDR-MLC context-eligible test rows")
    return result


def _ordered_target_tail(frame: pd.DataFrame, test_fraction: float) -> pd.DataFrame:
    if not 0 < test_fraction <= 1:
        raise ValueError("target_test_fraction must be in (0,1]")
    if np.isclose(test_fraction, 1.0):
        return frame.copy().reset_index(drop=True)
    parts = []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        start = int(len(group) * (1.0 - test_fraction))
        if not 0 < start < len(group):
            raise ValueError(f"{sequence_id}: insufficient ordered target tail")
        parts.append(group.iloc[start:].copy())
    return pd.concat(parts, ignore_index=True)


def evaluations(
    data: pd.DataFrame, ids, train_fraction: float = .80,
    target_test_fraction: float = .20,
):
    result = []
    is_quic = {"record_id", "period"}.issubset(data.columns)
    if is_quic:
        months = {
            month: data.loc[data.period.astype(str).eq(month)].copy()
            for month in QUIC_MONTHS
        }
        classes = tuple(APPLICATIONS)
        for protocol_id in ids:
            development, test, definition = build_quic_protocol(
                months, protocol_id, classes, train_fraction
            )
            # Score every zero-shot method on the exact rows that are causally
            # eligible for the published MF-CDR-MLC windows (3 and 50).
            base = trend_frame(test, TIMING, 3)
            context, _ = congestion_feature_frame(
                test,
                CongestionRouterConfig(
                    window=50, features=tuple(TIMING), expert_trees=1
                ).validate(),
            )
            eligible = base.index[base.index.isin(context.index)]
            definition["test_rows_before_context_filter"] = len(test)
            definition["test_rows"] = len(eligible)
            definition["context_eligibility_window"] = 50
            definition["test_identity_before_context_filter"] = definition["test_identity"]
            test = test.loc[eligible].copy()
            if test.empty:
                raise ValueError(f"S{protocol_id}: no context-eligible test rows")
            definition["test_identity"] = quic_record_identity(test)
            result.append((development, test, definition))
        return result
    external_dataset = None
    if "benchmark_dataset" in data.columns:
        names = data["benchmark_dataset"].astype(str).unique().tolist()
        if len(names) == 1 and names[0] in {
            "SDNCampus", "ISCX-Tor", "ISCX-VPN", "UNSW-IoT"
        }:
            external_dataset = names[0]
    if external_dataset is not None:
        from sdncampus_rf_cdr_mf_comparison import split_80_20

        development, raw_test, split_audit = split_80_20(
            data, train_fraction, split_mode="auto", split_seed=42
        )
        split_mode = split_audit[0]["split_mode"]
        protocol_name = (
            f"{external_dataset}-Stratified-80-20"
            if split_mode == "stratified"
            else f"{external_dataset}-Ordered-80-20"
        )
        # Match the MF-CDR-MLC comparison's common eligibility for its
        # congestion window of 20: the first 19 rows of every independent
        # test capture are causal cold-start rows.
        test_parts = []
        for _, group in raw_test.groupby("sequence_id", sort=False):
            ordered = group.sort_values(
                ["timestamp", "source_row"], kind="stable"
            )
            test_parts.append(ordered.iloc[19:].copy())
        test = pd.concat(test_parts, ignore_index=True)
        if test.empty:
            raise ValueError(
                f"{external_dataset}: no context-eligible test rows"
            )
        overlap = record_ids(development) & record_ids(test)
        if overlap:
            raise RuntimeError(
                f"{external_dataset}: {len(overlap)} development/test overlaps"
            )
        identity_payload = development[[
            "source_file", "source_row"
        ]].to_csv(index=False).encode("utf-8")
        import hashlib
        development_identity = hashlib.sha256(identity_payload).hexdigest()
        return [(
            development.reset_index(drop=True),
            test.reset_index(drop=True),
            {
                "protocol": protocol_name,
                "source": (
                    "stratified-80-percent"
                    if split_mode == "stratified" else "first-80-percent"
                ),
                "target": (
                    "stratified-20-percent"
                    if split_mode == "stratified" else "last-20-percent"
                ),
                "kind": (
                    "fixed within-class stratified holdout"
                    if split_mode == "stratified"
                    else "within-capture ordered holdout"
                ),
                "split_mode": split_mode,
                "split_seed": 42 if split_mode == "stratified" else None,
                "development_identity": development_identity,
                "test_rows_before_context_filter": len(raw_test),
                "test_rows": len(test),
                "context_eligibility_window": 20,
                "split_audit": split_audit,
            },
        )]
    for protocol_id in ids:
        if protocol_id in SCENARIOS:
            source, target = SCENARIOS[protocol_id]
            development = data[data.congestion_level.eq(source)].copy().reset_index(drop=True)
            target_all = data[data.congestion_level.eq(target)].copy().reset_index(drop=True)
            test = _ordered_target_tail(target_all, target_test_fraction)
            definition = {
                "protocol": f"S{protocol_id}", "source": source, "target": target,
                "kind": "ordered target-tail transfer",
                "target_test_fraction": target_test_fraction,
                "target_rows_before_tail": len(target_all),
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
        definition["test_rows_before_context_filter"] = len(test)
        test = mf_context_eligible_test(test)
        definition["test_rows"] = len(test)
        definition["context_eligibility_window"] = 50
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
    if {"record_id", "period"}.issubset(train.columns):
        features = quic_numeric_features(train)
    else:
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
    if data_dir.is_file() and data_dir.suffix.lower() == ".csv":
        from sdncampus_rf_cdr_mf_comparison import load_sdncampus

        data, raw_audit = load_sdncampus(data_dir)
        classes = sorted(data.traffic_label.astype(str).unique().tolist())
        # Mutate the shared ontology list so every already-imported benchmark
        # module observes the exact external-dataset class set.
        APPLICATIONS[:] = classes
        dataset_name = raw_audit.get("dataset", "External")
        data["benchmark_dataset"] = dataset_name
        audit = pd.DataFrame([{
            "dataset": dataset_name,
            "input": str(data_dir),
            "rows": len(data),
            "classes": "|".join(classes),
            "ordering_basis": raw_audit["ordering_basis"],
            "congestion_labels_available": False,
        }])
        return data, audit
    if all((data_dir / f"{month}.parquet").is_file() for month in QUIC_MONTHS):
        spec = load_quic_class_spec(data_dir.parent / "quicext25_classes.json")
        # Mutate the shared list object so benchmark modules that imported
        # APPLICATIONS by value observe the same immutable 20-class ontology.
        APPLICATIONS[:] = spec["classes"]
        months, audit = load_quic_months(data_dir)
        data = pd.concat(
            [months[month] for month in QUIC_MONTHS], ignore_index=True
        )
        audit["dataset"] = "CESNET-QUICEXT-25"
        return data, audit
    return load_dataset(data_dir, tuple(DEFAULT_CANDIDATES))
