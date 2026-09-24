"""Minimal learned-router extension for fixed CDR-MLC.

MiniBatchKMeans, trend features, and the three RF experts are preserved. A small RF
gate learns the oracle expert choice from a chronological source holdout. Gate
targets are produced by experts that did not train on those holdout records.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from adaptive_cdr_mlc import (
    APPLICATIONS, aligned_probabilities, make_preprocessor,
    select_classifier_columns, trend_frame,
)
from compare_clean_valid import TIMING, fit_fixed_cdr, rf_config


@dataclass(frozen=True)
class LearnedRouterConfig:
    window: int = 3
    expert_trees: int = 20
    gate_trees: int = 50
    gate_max_depth: int = 8
    gate_min_samples_leaf: int = 20
    gate_min_confidence: float = .45
    router_train_fraction: float = .70
    random_state: int = 42

    def validate(self):
        if self.window < 1 or self.expert_trees < 1 or self.gate_trees < 1:
            raise ValueError("window and tree counts must be positive")
        if not 0 < self.router_train_fraction < 1:
            raise ValueError("router_train_fraction must be in (0,1)")
        if not 0 <= self.gate_min_confidence <= 1:
            raise ValueError("gate_min_confidence must be in [0,1]")
        return self


def chronological_split(frame: pd.DataFrame, fraction: float):
    train, gate = [], []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        cut = int(len(group) * fraction)
        if not 0 < cut < len(group):
            raise ValueError("capture too short for learned-router split")
        train.append(group.iloc[:cut].copy())
        gate.append(group.iloc[cut:].copy())
    return pd.concat(train), pd.concat(gate)


def _expert_outputs(model, frame):
    view = trend_frame(frame, TIMING, model["window"])
    raw = frame.loc[view.index]
    z = model["scaler"].transform(view[model["trend_columns"]])
    distances = model["router"].transform(z)
    minibatch_route = model["router"].predict(z)
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(raw[columns])
    probabilities, predictions = [], []
    for cluster in range(3):
        probability = aligned_probabilities(
            model["experts"][cluster], x, APPLICATIONS
        )
        probabilities.append(probability)
        predictions.append(np.asarray(APPLICATIONS)[probability.argmax(axis=1)])
    probabilities = np.stack(probabilities, axis=1)
    predictions = np.stack(predictions, axis=1)
    return raw, z, distances, minibatch_route, probabilities, predictions


def _gate_features(distances, minibatch_route, probabilities):
    maximum = probabilities.max(axis=2)
    ordered = np.sort(probabilities, axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=2)
    one_hot = np.eye(3)[minibatch_route]
    return np.column_stack([
        distances,
        probabilities.reshape(len(probabilities), -1),
        maximum,
        margin,
        entropy,
        one_hot,
    ])


def _oracle_route(truth, probabilities, predictions):
    correct = predictions == truth[:, None]
    true_index = np.array([APPLICATIONS.index(label) for label in truth])
    true_probability = probabilities[
        np.arange(len(truth))[:, None],
        np.arange(3)[None, :],
        true_index[:, None],
    ]
    # Correct experts always outrank incorrect experts. Probability breaks ties.
    return np.where(correct, true_probability + 2.0, true_probability).argmax(axis=1)


def _refit_experts_with_frozen_router(initial, full_source, config):
    view = trend_frame(full_source, TIMING, initial["window"])
    raw = full_source.loc[view.index]
    z = initial["scaler"].transform(view[initial["trend_columns"]])
    routes = initial["router"].predict(z)
    numeric, categorical = select_classifier_columns(raw, TIMING)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    labels = raw.traffic_label.to_numpy()
    experts, counts = {}, []
    for cluster in range(3):
        mask = routes == cluster
        if not mask.any():
            raise ValueError(f"empty full-source cluster {cluster}")
        experts[cluster] = RandomForestClassifier(
            **rf_config(config.random_state + cluster, config.expert_trees)
        ).fit(x[mask], labels[mask])
        counts.append({"cluster": cluster, "rows": int(mask.sum())})
    final = dict(initial)
    final.update({
        "preprocessor": preprocessor,
        "numeric": numeric,
        "categorical": categorical,
        "experts": experts,
        "full_source_rows": len(raw),
        "source_eligible_index": view.index.tolist(),
        "full_source_cluster_counts": counts,
    })
    return final


def fit_learned_router(source: pd.DataFrame, config: LearnedRouterConfig):
    config.validate()
    expert_train, gate_holdout = chronological_split(
        source, config.router_train_fraction
    )
    initial = fit_fixed_cdr(
        expert_train, config.window, config.random_state, config.expert_trees
    )
    raw, _, distances, minibatch_route, probabilities, predictions = _expert_outputs(
        initial, gate_holdout
    )
    truth = raw.traffic_label.to_numpy()
    oracle_route = _oracle_route(truth, probabilities, predictions)
    features = _gate_features(distances, minibatch_route, probabilities)
    unique = np.unique(oracle_route)
    if len(unique) == 1:
        gate = None
        constant_route = int(unique[0])
    else:
        gate = RandomForestClassifier(
            n_estimators=config.gate_trees,
            max_depth=config.gate_max_depth,
            min_samples_leaf=config.gate_min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_state,
        ).fit(features, oracle_route)
        constant_route = None
    final = _refit_experts_with_frozen_router(initial, source, config)
    final.update({
        "learned_gate": gate,
        "constant_gate_route": constant_route,
        "learned_router_config": config,
        "gate_training_rows": len(raw),
        "gate_target_counts": {
            str(cluster): int((oracle_route == cluster).sum()) for cluster in range(3)
        },
        "gate_minibatch_kmeans_agreement": float(
            np.mean(oracle_route == minibatch_route)
        ),
    })
    return final


def predict_all(model, frame):
    raw, _, distances, minibatch_route, probabilities, predictions = _expert_outputs(
        model, frame
    )
    gate_features = _gate_features(distances, minibatch_route, probabilities)
    gate = model["learned_gate"]
    if gate is None:
        learned_route = np.full(len(raw), model["constant_gate_route"], dtype=int)
        learned_confidence = np.ones(len(raw))
    else:
        learned_route = gate.predict(gate_features).astype(int)
        learned_confidence = gate.predict_proba(gate_features).max(axis=1)
    fallback = learned_confidence < model["learned_router_config"].gate_min_confidence
    learned_route[fallback] = minibatch_route[fallback]
    truth = raw.traffic_label.to_numpy()
    oracle_route = _oracle_route(truth, probabilities, predictions)
    row = np.arange(len(raw))
    report = pd.DataFrame({
        "minibatch_kmeans_route": minibatch_route,
        "learned_route": learned_route,
        "oracle_route": oracle_route,
        "learned_confidence": learned_confidence,
        "used_minibatch_kmeans_fallback": fallback,
        "learned_matches_oracle": learned_route == oracle_route,
        "minibatch_kmeans_matches_oracle": minibatch_route == oracle_route,
    }, index=raw.index)
    return {
        "observed": raw,
        "CDR_MLC_actual_router": predictions[row, minibatch_route],
        "CDR_MLC_learned_router": predictions[row, learned_route],
        "CDR_MLC_oracle_router": predictions[row, oracle_route],
        "routes": report,
    }
