"""Production inference paths for Original CDR-MLC and MF-CDR-MLC.

The published diagnostic ``predict_all`` routine also computes an oracle route
from the ground-truth label.  That diagnostic must not be included in a
deployment latency measurement.  The functions here return only predictions
that are available at inference time and expose non-overlapping stage timings.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from time import perf_counter

import numpy as np
import pandas as pd

from adaptive_cdr_mlc import APPLICATIONS, aligned_probabilities, trend_values
from compare_clean_valid import TIMING
from congestion_feature_cdr_mlc import congestion_feature_values
from learned_router_cdr_mlc import _gate_features
from meta_stacked_cdr_mlc_leakage_safe import (
    _aligned_meta_probabilities,
    _hybrid_prediction,
    _meta_features,
)
from utility_router_cdr_mlc import _utility_routes


def _elapsed(start: float) -> float:
    return perf_counter() - start


def _expert_probability(expert, values):
    return aligned_probabilities(expert, values, APPLICATIONS)


def _expert_bank(model, values, workers: int):
    clusters = tuple(sorted(model["experts"]))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(clusters))) as pool:
            probabilities = list(pool.map(
                lambda cluster: _expert_probability(
                    model["experts"][cluster], values
                ),
                clusters,
            ))
    else:
        probabilities = [
            _expert_probability(model["experts"][cluster], values)
            for cluster in clusters
        ]
    probability = np.stack(probabilities, axis=1)
    prediction = np.asarray(APPLICATIONS)[probability.argmax(axis=2)]
    return probability, prediction


def _utility_bank(model, features, workers: int):
    utilities = np.zeros((len(features), 3), dtype=float)

    def one(expert):
        estimator = model["utility_models"][expert]
        if estimator is None:
            value = float(model["utility_constants"][expert] == 1)
            return expert, np.full(len(features), value, dtype=float)
        classes = list(estimator.classes_)
        if 1 not in classes:
            return expert, np.zeros(len(features), dtype=float)
        return expert, estimator.predict_proba(features)[:, classes.index(1)]

    experts = tuple(range(3))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
            outputs = list(pool.map(one, experts))
    else:
        outputs = [one(expert) for expert in experts]
    for expert, values in outputs:
        utilities[:, expert] = values
    return utilities


def _current_trend(values):
    """Feature-major TTFEF values for one complete streaming window."""
    block = np.asarray(values, dtype=np.float64)
    return np.stack([
        block.mean(axis=0), block.max(axis=0), np.median(block, axis=0),
        block.min(axis=0), block.std(axis=0, ddof=0),
    ], axis=1).reshape(1, -1)


def _current_congestion(values, epsilon):
    """Feature-major congestion context for one streaming endpoint."""
    block = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(block, (.25, .50, .75), axis=0)
    mean = block.mean(axis=0)
    std = block.std(axis=0, ddof=0)
    scale = np.abs(median) + epsilon
    x = np.arange(len(block), dtype=np.float64)
    x -= x.mean()
    slope = np.dot(x, block) / np.dot(x, x)
    return np.stack([
        np.log1p(np.maximum(median, 0.0)),
        std / (np.abs(mean) + epsilon),
        (q75 - q25) / scale,
        (block.max(axis=0) - block.min(axis=0)) / scale,
        (block[-1] - block[0]) / scale,
        slope / scale,
    ], axis=1).reshape(1, -1)


class StreamingCDRPredictor:
    """Stateful one-record-at-a-time Original/MF-CDR-MLC inference."""

    def __init__(self, model, mf=False, branch_workers=1):
        self.model = model
        self.mf = mf
        self.branch_workers = branch_workers
        self.trend_features = tuple(TIMING)
        self.trend_buffers = defaultdict(
            lambda: deque(maxlen=int(model["window"]))
        )
        if mf:
            config = model["congestion_config"]
            self.context_features = tuple(
                feature for feature in config.features
                if feature in model.get("congestion_source_columns", config.features)
            )
            self.context_buffers = defaultdict(
                lambda: deque(maxlen=int(config.window))
            )
            self.context_epsilon = float(config.epsilon)
        self.pool = (
            ThreadPoolExecutor(max_workers=min(branch_workers, 3))
            if mf and branch_workers > 1 else None
        )

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    @staticmethod
    def _row_values(row, features):
        values = pd.to_numeric(
            row.loc[:, list(features)].iloc[0], errors="coerce"
        ).to_numpy(dtype=np.float64)
        return values

    def update(self, row, predict=True):
        sequence = str(row.sequence_id.iloc[0])
        trend_values = self._row_values(row, self.trend_features)
        trend_buffer = self.trend_buffers[sequence]
        if np.isfinite(trend_values).all() and (trend_values >= 0).all():
            trend_buffer.append(trend_values)
        else:
            trend_buffer.clear()

        context_buffer = None
        if self.mf:
            context_values = self._row_values(row, self.context_features)
            context_buffer = self.context_buffers[sequence]
            if np.isfinite(context_values).all() and (context_values >= 0).all():
                context_buffer.append(context_values)
            else:
                context_buffer.clear()

        if not predict:
            return None, {}
        if len(trend_buffer) < self.model["window"]:
            return None, {}
        if self.mf and len(context_buffer) < self.model["congestion_config"].window:
            return None, {}

        stages = {}
        started = perf_counter()
        trend = _current_trend(trend_buffer)
        stages["ttfef_seconds"] = _elapsed(started)

        started = perf_counter()
        scaled = self.model["scaler"].transform(trend)
        distances = self.model["router"].transform(scaled)
        route = self.model["router"].predict(scaled)
        columns = self.model["numeric"] + self.model["categorical"]
        values = self.model["preprocessor"].transform(row[columns])
        stages["router_preprocess_seconds"] = _elapsed(started)

        if not self.mf:
            started = perf_counter()
            prediction = self.model["experts"][int(route[0])].predict(values)[0]
            stages["expert_seconds"] = _elapsed(started)
            return prediction, stages

        started = perf_counter()
        clusters = tuple(sorted(self.model["experts"]))
        if self.pool is None:
            banks = [
                aligned_probabilities(
                    self.model["experts"][cluster], values, APPLICATIONS
                ) for cluster in clusters
            ]
        else:
            futures = [
                self.pool.submit(
                    aligned_probabilities,
                    self.model["experts"][cluster], values, APPLICATIONS,
                ) for cluster in clusters
            ]
            banks = [future.result() for future in futures]
        probability = np.stack(banks, axis=1)
        prediction = np.asarray(APPLICATIONS)[probability.argmax(axis=2)]
        stages["expert_seconds"] = _elapsed(started)

        started = perf_counter()
        context = _current_congestion(
            context_buffer, self.context_epsilon
        )
        base = np.column_stack([
            _gate_features(distances, route, probability), context
        ])
        stages["congestion_context_seconds"] = _elapsed(started)

        started = perf_counter()
        if self.pool is None:
            utility = _utility_bank(self.model, base, workers=1)
        else:
            utility = np.zeros((1, 3), dtype=float)

            def utility_one(expert):
                estimator = self.model["utility_models"][expert]
                if estimator is None:
                    return expert, float(
                        self.model["utility_constants"][expert] == 1
                    )
                classes = list(estimator.classes_)
                if 1 not in classes:
                    return expert, 0.0
                return expert, float(
                    estimator.predict_proba(base)[0, classes.index(1)]
                )

            futures = [self.pool.submit(utility_one, expert) for expert in range(3)]
            for future in futures:
                expert, value = future.result()
                utility[0, expert] = value
        stages["utility_seconds"] = _elapsed(started)

        started = perf_counter()
        utility_route, _, _, _ = _utility_routes(route, utility, 0.0)
        hard_prediction = prediction[0, utility_route[0]]
        meta_probability = _aligned_meta_probabilities(
            self.model["meta_models"][self.model["selected_meta_variant"]],
            _meta_features(base, utility),
        )
        stacked, _, _, _ = _hybrid_prediction(
            meta_probability,
            np.asarray([hard_prediction]),
            self.model["selected_meta_confidence"],
        )
        stages["meta_fusion_seconds"] = _elapsed(started)
        return stacked[0], stages


def _base_state(model, frame, workers: int):
    stages = {}

    started = perf_counter()
    trend_index, trend_columns, trend_matrix = trend_values(
        frame, TIMING, model["window"]
    )
    stages["ttfef_seconds"] = _elapsed(started)

    started = perf_counter()
    raw = frame.loc[trend_index]
    if trend_columns != list(model["trend_columns"]):
        positions = [trend_columns.index(column) for column in model["trend_columns"]]
        trend_matrix = trend_matrix[:, positions]
    scaled = model["scaler"].transform(trend_matrix)
    distances = model["router"].transform(scaled)
    route = model["router"].predict(scaled)
    columns = model["numeric"] + model["categorical"]
    values = model["preprocessor"].transform(raw[columns])
    stages["router_preprocess_seconds"] = _elapsed(started)

    started = perf_counter()
    probability, prediction = _expert_bank(model, values, workers)
    stages["expert_seconds"] = _elapsed(started)
    return raw, distances, route, probability, prediction, stages


def predict_original_profile(model, frame):
    """Predict Original CDR-MLC without any label-dependent diagnostics."""
    started_total = perf_counter()
    stages = {}
    started = perf_counter()
    trend_index, trend_columns, trend_matrix = trend_values(
        frame, TIMING, model["window"]
    )
    stages["ttfef_seconds"] = _elapsed(started)
    started = perf_counter()
    raw = frame.loc[trend_index]
    if trend_columns != list(model["trend_columns"]):
        positions = [trend_columns.index(column) for column in model["trend_columns"]]
        trend_matrix = trend_matrix[:, positions]
    scaled = model["scaler"].transform(trend_matrix)
    route = model["router"].predict(scaled)
    columns = model["numeric"] + model["categorical"]
    values = model["preprocessor"].transform(raw[columns])
    stages["router_preprocess_seconds"] = _elapsed(started)
    started = perf_counter()
    output = np.empty(len(raw), dtype=object)
    for cluster, expert in sorted(model["experts"].items()):
        selected = route == cluster
        if selected.any():
            output[selected] = expert.predict(values[selected])
    stages["expert_seconds"] = _elapsed(started)
    stages["end_to_end_seconds"] = _elapsed(started_total)
    return raw, output, stages


def predict_mf_profile(model, frame, branch_workers: int = 1):
    """Predict MF-CDR-MLC with sequential or parallel expert/utility banks.

    ``branch_workers=1`` is the reference production path.  Values above one
    execute the three independent experts concurrently and, after their
    outputs are available, execute the three independent utility estimators
    concurrently.  The dependency order and predictions are unchanged.
    """
    started_total = perf_counter()
    raw, distances, route, probability, prediction, stages = _base_state(
        model, frame, workers=branch_workers
    )

    started = perf_counter()
    congestion_index, _, congestion_matrix = congestion_feature_values(
        frame, model["congestion_config"]
    )
    common = raw.index[raw.index.isin(congestion_index)]
    if len(common) == 0:
        raise ValueError("no common base/congestion windows")
    positions = raw.index.get_indexer(common)
    raw = raw.loc[common]
    distances = distances[positions]
    route = route[positions]
    probability = probability[positions]
    prediction = prediction[positions]
    congestion_positions = congestion_index.get_indexer(common)
    congestion_values = congestion_matrix[congestion_positions]
    base = np.column_stack([
        _gate_features(distances, route, probability), congestion_values
    ])
    stages["congestion_context_seconds"] = _elapsed(started)

    started = perf_counter()
    utility = _utility_bank(model, base, workers=branch_workers)
    stages["utility_seconds"] = _elapsed(started)

    started = perf_counter()
    utility_route, _, _, _ = _utility_routes(route, utility, 0.0)
    rows = np.arange(len(raw))
    hard_prediction = prediction[rows, utility_route]
    meta_values = _meta_features(base, utility)
    meta_probability = _aligned_meta_probabilities(
        model["meta_models"][model["selected_meta_variant"]], meta_values
    )
    stacked, _, _, _ = _hybrid_prediction(
        meta_probability,
        hard_prediction,
        model["selected_meta_confidence"],
    )
    stages["meta_fusion_seconds"] = _elapsed(started)
    stages["end_to_end_seconds"] = _elapsed(started_total)
    return raw, stacked, stages


def predict_mf_pipelined(model, frame):
    """Overlap MF stages across independent capture sequences.

    The three workers implement the deployment pipeline proposed in the paper:
    (1) causal preprocessing/router/context, (2) expert bank, and (3)
    utility/meta fusion.  At steady state, different sequences occupy different
    stages concurrently.  Records inside a sequence remain ordered, so moving
    windows are causal.  Each tree estimator must be configured with
    ``n_jobs=1`` by the caller to keep the total CPU budget bounded.
    """
    sentinel = object()
    prepared_queue = Queue(maxsize=2)
    expert_queue = Queue(maxsize=2)
    completed_queue = Queue()
    groups = [
        (position, group.copy())
        for position, (_, group) in enumerate(
            frame.groupby("sequence_id", sort=False)
        )
    ]
    if not groups:
        raise ValueError("empty MF inference frame")

    def prepare_worker():
        try:
            for position, group in groups:
                timings = {}
                started = perf_counter()
                trend_index, trend_columns, trend_matrix = trend_values(
                    group, TIMING, model["window"]
                )
                timings["ttfef_seconds"] = _elapsed(started)

                started = perf_counter()
                raw = group.loc[trend_index]
                if trend_columns != list(model["trend_columns"]):
                    positions = [
                        trend_columns.index(column)
                        for column in model["trend_columns"]
                    ]
                    trend_matrix = trend_matrix[:, positions]
                scaled = model["scaler"].transform(trend_matrix)
                distances = model["router"].transform(scaled)
                route = model["router"].predict(scaled)
                columns = model["numeric"] + model["categorical"]
                values = model["preprocessor"].transform(raw[columns])
                timings["router_preprocess_seconds"] = _elapsed(started)

                started = perf_counter()
                congestion_index, _, congestion_matrix = congestion_feature_values(
                    group, model["congestion_config"]
                )
                common = raw.index[raw.index.isin(congestion_index)]
                if len(common) == 0:
                    raise ValueError("no common base/congestion windows")
                offsets = raw.index.get_indexer(common)
                congestion_positions = congestion_index.get_indexer(common)
                congestion_values = congestion_matrix[congestion_positions]
                timings["congestion_context_seconds"] = _elapsed(started)
                prepared_queue.put((
                    position, raw.loc[common], distances[offsets],
                    route[offsets], values[offsets], congestion_values,
                    timings,
                ))
        except BaseException as error:
            completed_queue.put(("error", error))
        finally:
            prepared_queue.put(sentinel)

    def expert_worker():
        try:
            while True:
                item = prepared_queue.get()
                if item is sentinel:
                    break
                (position, raw, distances, route, values,
                 congestion_values, timings) = item
                started = perf_counter()
                probability, prediction = _expert_bank(
                    model, values, workers=1
                )
                base = np.column_stack([
                    _gate_features(distances, route, probability),
                    congestion_values,
                ])
                timings["expert_seconds"] = _elapsed(started)
                expert_queue.put((
                    position, raw, route, prediction, base, timings
                ))
        except BaseException as error:
            completed_queue.put(("error", error))
        finally:
            expert_queue.put(sentinel)

    def fusion_worker():
        try:
            while True:
                item = expert_queue.get()
                if item is sentinel:
                    break
                position, raw, route, prediction, base, timings = item
                started = perf_counter()
                utility = _utility_bank(model, base, workers=1)
                timings["utility_seconds"] = _elapsed(started)

                started = perf_counter()
                utility_route, _, _, _ = _utility_routes(route, utility, 0.0)
                rows = np.arange(len(raw))
                hard_prediction = prediction[rows, utility_route]
                meta_probability = _aligned_meta_probabilities(
                    model["meta_models"][model["selected_meta_variant"]],
                    _meta_features(base, utility),
                )
                stacked, _, _, _ = _hybrid_prediction(
                    meta_probability, hard_prediction,
                    model["selected_meta_confidence"],
                )
                timings["meta_fusion_seconds"] = _elapsed(started)
                completed_queue.put((
                    "result", position, raw, stacked, timings
                ))
        except BaseException as error:
            completed_queue.put(("error", error))

    started_total = perf_counter()
    workers = [
        Thread(target=prepare_worker, daemon=True),
        Thread(target=expert_worker, daemon=True),
        Thread(target=fusion_worker, daemon=True),
    ]
    for worker in workers:
        worker.start()

    outputs = []
    while len(outputs) < len(groups):
        item = completed_queue.get()
        if item[0] == "error":
            raise item[1]
        outputs.append(item[1:])
    for worker in workers:
        worker.join()
    outputs.sort(key=lambda item: item[0])
    raw = pd.concat([item[1] for item in outputs])
    prediction = np.concatenate([item[2] for item in outputs])
    stage_names = {
        name for item in outputs for name in item[3]
    }
    timings = {
        name: float(sum(item[3].get(name, 0.0) for item in outputs))
        for name in stage_names
    }
    timings["end_to_end_seconds"] = _elapsed(started_total)
    return raw, prediction, timings
