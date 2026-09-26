"""Load one saved model and measure CPU-only inference on the fixed sample."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import platform
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
CDR_MLC = HERE.parents[1]
if str(CDR_MLC) not in sys.path:
    sys.path.insert(0, str(CDR_MLC))

from adaptive_cdr_mlc import APPLICATIONS  # noqa: E402
from common import (  # noqa: E402
    ProcessSampler, configure_cpu, cpu_thread_limit, frame_sha256,
    load_artifact, metric_values, set_estimator_jobs, transform_tabular,
    write_json,
)
from mf_runtime import (  # noqa: E402
    StreamingCDRPredictor, predict_mf_pipelined, predict_mf_profile,
    predict_original_profile,
)


def _torch_predict(model, values, batch_size, forward=None):
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size])
            logits = forward(model, batch) if forward else model(batch)
            output.append(logits.argmax(1).numpy())
    return np.concatenate(output)


def _predict_batch(artifact, frame, variant, threads):
    kind = artifact["kind"]
    stages = {}
    started_total = time.perf_counter()

    if kind == "rf":
        started = time.perf_counter()
        model = artifact["model"]
        columns = model["numeric"] + model["categorical"]
        values = model["preprocessor"].transform(frame[columns])
        stages["preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model["model"].predict(values)
        stages["classifier_seconds"] = time.perf_counter() - started
        observed = frame

    elif kind == "original-cdr":
        observed, prediction, stages = predict_original_profile(
            artifact["model"], frame
        )

    elif kind == "mf-cdr":
        if variant == "pipelined":
            observed, prediction, stages = predict_mf_pipelined(
                artifact["model"], frame
            )
        else:
            workers = threads if variant == "parallel" else 1
            observed, prediction, stages = predict_mf_profile(
                artifact["model"], frame, branch_workers=workers
            )

    elif kind in {"1d-cnn", "ft-transformer", "scarf"}:
        started = time.perf_counter()
        values = transform_tabular(artifact, frame)
        stages["preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        forward = (
            (lambda model, batch: model(batch.unsqueeze(1)))
            if kind == "1d-cnn" else None
        )
        encoded = _torch_predict(
            artifact["model"], values, artifact["batch_size"], forward
        )
        prediction = np.asarray(APPLICATIONS)[encoded]
        stages["classifier_seconds"] = time.perf_counter() - started
        observed = frame

    elif kind == "graphsage":
        module = importlib.import_module(
            "benchmarks.GraphSAGE.graphsage_iust_mscn"
        )
        started = time.perf_counter()
        values = transform_tabular(artifact, frame)
        adjacency = module.causal_adjacency(
            frame.reset_index(drop=True), artifact["neighbors"],
            torch.device("cpu"),
        )
        stages["preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        artifact["model"].eval()
        with torch.inference_mode():
            encoded = artifact["model"](
                torch.from_numpy(values), adjacency
            ).argmax(1).numpy()
        prediction = np.asarray(APPLICATIONS)[encoded]
        stages["classifier_seconds"] = time.perf_counter() - started
        observed = frame

    elif kind == "dfe":
        module = importlib.import_module("benchmarks.DFE.dfe_iust_mscn")
        started = time.perf_counter()
        values = artifact["transform"].transform(frame)
        stages["preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        query = module.embed(
            artifact["model"], values, torch.device("cpu")
        )
        stages["embedding_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        templates = artifact["template_embeddings"]
        distances = np.linalg.norm(
            query[:, None, :] - templates[None, :, :], axis=2
        )
        k = min(int(artifact["k"]), len(artifact["template_labels"]))
        nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        nearest_distance = np.take_along_axis(distances, nearest, axis=1)
        encoded = module.vote(
            artifact["template_labels"][nearest], nearest_distance
        )
        prediction = artifact["label_encoder"].inverse_transform(encoded)
        stages["template_match_seconds"] = time.perf_counter() - started
        observed = frame

    elif kind == "af":
        started = time.perf_counter()
        values = transform_tabular(artifact, frame)
        stages["preprocess_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        artifact["model"].eval()
        with torch.inference_mode():
            embedding = artifact["model"](torch.from_numpy(values)).numpy()
        embedding = normalize(embedding, norm="l2")
        stages["embedding_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        encoded = artifact["classifier"].predict(embedding)
        prediction = artifact["label_encoder"].inverse_transform(encoded)
        stages["knn_seconds"] = time.perf_counter() - started
        observed = frame

    else:
        raise ValueError(f"unsupported artifact kind: {kind}")

    stages["end_to_end_seconds"] = time.perf_counter() - started_total
    return observed, np.asarray(prediction), stages


class _StreamingPredictor:
    """One-record-at-a-time CPU inference with persistent causal state."""

    def __init__(self, artifact, variant, threads):
        self.artifact = artifact
        self.kind = artifact["kind"]
        self.variant = variant
        self.threads = threads
        self.graph_history = defaultdict(deque)
        self.graph_module = None
        self.cdr = None
        if self.kind == "original-cdr":
            self.cdr = StreamingCDRPredictor(artifact["model"], mf=False)
        elif self.kind == "mf-cdr":
            workers = threads if variant == "parallel" else 1
            self.cdr = StreamingCDRPredictor(
                artifact["model"], mf=True, branch_workers=workers
            )
        elif self.kind == "graphsage":
            self.graph_module = importlib.import_module(
                "benchmarks.GraphSAGE.graphsage_iust_mscn"
            )

    def close(self):
        if self.cdr is not None:
            self.cdr.close()

    def warm(self, row):
        if self.cdr is not None:
            self.cdr.update(row, predict=False)
        elif self.kind == "graphsage":
            self._graph_predict(row)

    def _graph_predict(self, row):
        started = time.perf_counter()
        values = transform_tabular(self.artifact, row)[0]
        sequence = str(row.sequence_id.iloc[0])
        history = self.graph_history[sequence]
        layer_count = len(self.artifact["model"].layers)
        capacity = 1 + int(self.artifact["neighbors"]) * layer_count
        history.append(values)
        while len(history) > capacity:
            history.popleft()
        matrix = np.asarray(history, dtype=np.float32)
        context = pd.DataFrame({
            "sequence_id": [sequence] * len(matrix),
            "timestamp": np.arange(len(matrix)),
            "source_row": np.arange(len(matrix)),
        })
        adjacency = self.graph_module.causal_adjacency(
            context, self.artifact["neighbors"], torch.device("cpu")
        )
        preprocess = time.perf_counter() - started
        started = time.perf_counter()
        self.artifact["model"].eval()
        with torch.inference_mode():
            encoded = int(self.artifact["model"](
                torch.from_numpy(matrix), adjacency
            )[-1].argmax().item())
        return np.asarray(APPLICATIONS)[encoded], {
            "preprocess_seconds": preprocess,
            "classifier_seconds": time.perf_counter() - started,
        }

    def predict(self, row):
        if self.cdr is not None:
            return self.cdr.update(row, predict=True)

        kind = self.kind
        stages = {}
        if kind == "rf":
            started = time.perf_counter()
            model = self.artifact["model"]
            columns = model["numeric"] + model["categorical"]
            values = model["preprocessor"].transform(row[columns])
            stages["preprocess_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            prediction = model["model"].predict(values)[0]
            stages["classifier_seconds"] = time.perf_counter() - started
            return prediction, stages

        if kind in {"1d-cnn", "ft-transformer", "scarf"}:
            started = time.perf_counter()
            values = transform_tabular(self.artifact, row)
            stages["preprocess_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            batch = torch.from_numpy(values)
            self.artifact["model"].eval()
            with torch.inference_mode():
                logits = (
                    self.artifact["model"](batch.unsqueeze(1))
                    if kind == "1d-cnn" else self.artifact["model"](batch)
                )
                encoded = int(logits[0].argmax().item())
            stages["classifier_seconds"] = time.perf_counter() - started
            return np.asarray(APPLICATIONS)[encoded], stages

        if kind == "graphsage":
            return self._graph_predict(row)

        if kind == "dfe":
            module = importlib.import_module("benchmarks.DFE.dfe_iust_mscn")
            started = time.perf_counter()
            values = self.artifact["transform"].transform(row)
            stages["preprocess_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            query = module.embed(
                self.artifact["model"], values, torch.device("cpu")
            )
            stages["embedding_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            distances = np.linalg.norm(
                query[:, None, :] - self.artifact["template_embeddings"][None, :, :],
                axis=2,
            )
            k = min(
                int(self.artifact["k"]),
                len(self.artifact["template_labels"]),
            )
            nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            nearest_distance = np.take_along_axis(distances, nearest, axis=1)
            encoded = module.vote(
                self.artifact["template_labels"][nearest], nearest_distance
            )
            prediction = self.artifact["label_encoder"].inverse_transform(encoded)[0]
            stages["template_match_seconds"] = time.perf_counter() - started
            return prediction, stages

        if kind == "af":
            started = time.perf_counter()
            values = transform_tabular(self.artifact, row)
            stages["preprocess_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            self.artifact["model"].eval()
            with torch.inference_mode():
                embedding = self.artifact["model"](
                    torch.from_numpy(values)
                ).numpy()
            embedding = normalize(embedding, norm="l2")
            stages["embedding_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            encoded = self.artifact["classifier"].predict(embedding)
            prediction = self.artifact["label_encoder"].inverse_transform(encoded)[0]
            stages["knn_seconds"] = time.perf_counter() - started
            return prediction, stages
        raise ValueError(f"unsupported artifact kind: {kind}")


def _predict_stream(artifact, frame, variant, threads):
    predictor = _StreamingPredictor(artifact, variant, threads)
    predictions, truth, latencies = [], [], []
    stage_totals = defaultdict(float)
    try:
        for position in range(len(frame)):
            row = frame.iloc[[position]]
            scored = bool(row.benchmark_scored.iloc[0])
            if not scored:
                predictor.warm(row)
                continue
            started = time.perf_counter()
            prediction, stages = predictor.predict(row)
            latency = time.perf_counter() - started
            if prediction is None:
                raise RuntimeError("scored row lacks sufficient causal context")
            predictions.append(prediction)
            truth.append(str(row.traffic_label.iloc[0]))
            latencies.append(latency)
            for name, seconds in stages.items():
                stage_totals[name] += float(seconds)
    finally:
        predictor.close()
    stage_totals["end_to_end_seconds"] = float(sum(latencies))
    return (
        np.asarray(truth), np.asarray(predictions),
        np.asarray(latencies, dtype=float), dict(stage_totals),
    )


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("streaming", "batch"), default="streaming",
        help="streaming measures one arriving record at a time; batch is offline throughput",
    )
    parser.add_argument("--cpu-threads", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    configure_cpu(args.cpu_threads)
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(args.sample)
    if "benchmark_scored" not in frame:
        raise ValueError("sample lacks benchmark_scored marker")
    gc.collect()
    with ProcessSampler() as load_monitor:
        started = time.perf_counter()
        artifact = load_artifact(args.model)
        load_seconds = time.perf_counter() - started
    if "model" in artifact and hasattr(artifact["model"], "cpu"):
        artifact["model"] = artifact["model"].cpu().eval()

    if artifact["kind"] == "mf-cdr":
        variants = (
            ("MF-Sequential", "sequential"),
            ("MF-Parallel", "parallel"),
        ) if args.mode == "streaming" else (
            ("MF-Sequential", "sequential"),
            ("MF-Parallel", "parallel"),
            ("MF-Pipelined", "pipelined"),
        )
    else:
        variants = ((artifact["kind"], "default"),)

    summaries, repeat_rows = [], []
    reference = {}
    mf_reference_prediction = None
    for method_name, variant in variants:
        # The parallel MF bank uses one thread per branch.  All other methods
        # may use the same total CPU-thread budget internally.
        # For one-row streaming requests, sklearn's per-forest joblib
        # parallelism costs much more than it saves.  Keep every individual
        # forest single-threaded; MF-Parallel alone uses the shared CPU budget
        # by executing independent expert/utility branches in its persistent
        # ThreadPool.  Batch mode retains internal estimator parallelism.
        estimator_jobs = (
            1 if args.mode == "streaming"
            or variant in {"parallel", "pipelined"}
            else args.cpu_threads
        )
        set_estimator_jobs(artifact, estimator_jobs)
        with cpu_thread_limit(args.cpu_threads):
            for _ in range(args.warmup_runs):
                if args.mode == "streaming":
                    _predict_stream(artifact, frame, variant, args.cpu_threads)
                else:
                    _predict_batch(artifact, frame, variant, args.cpu_threads)

            runs = []
            all_record_latencies = []
            for repeat in range(args.repeats):
                with ProcessSampler() as monitor:
                    if args.mode == "streaming":
                        truth, scored_prediction, record_latencies, stages = (
                            _predict_stream(
                                artifact, frame, variant, args.cpu_threads
                            )
                        )
                        all_record_latencies.extend(record_latencies.tolist())
                        evaluated_rows = len(scored_prediction)
                    else:
                        observed, prediction, stages = _predict_batch(
                            artifact, frame, variant, args.cpu_threads
                        )
                        scored = observed.benchmark_scored.astype(bool).to_numpy()
                        truth = observed.loc[
                            scored, "traffic_label"
                        ].astype(str).to_numpy()
                        scored_prediction = prediction[scored]
                        record_latencies = np.asarray([], dtype=float)
                        evaluated_rows = int(scored.sum())
                if method_name in reference:
                    if not np.array_equal(reference[method_name], scored_prediction):
                        raise RuntimeError(
                            f"{method_name}: predictions changed across repeats"
                        )
                else:
                    reference[method_name] = scored_prediction.copy()
                    if artifact["kind"] == "mf-cdr":
                        if mf_reference_prediction is None:
                            mf_reference_prediction = scored_prediction.copy()
                        elif not np.array_equal(
                            mf_reference_prediction, scored_prediction
                        ):
                            raise RuntimeError(
                                f"{method_name}: optimized MF predictions differ "
                                "from MF-Sequential"
                            )
                row = {
                    "method": method_name, "repeat": repeat,
                    "mode": args.mode,
                    "input_rows": len(frame), "evaluated_rows": evaluated_rows,
                    "record_latency_mean_seconds": (
                        float(record_latencies.mean())
                        if len(record_latencies) else np.nan
                    ),
                    "record_latency_p50_seconds": (
                        _percentile(record_latencies, 50)
                        if len(record_latencies) else np.nan
                    ),
                    "record_latency_p95_seconds": (
                        _percentile(record_latencies, 95)
                        if len(record_latencies) else np.nan
                    ),
                    "record_latency_p99_seconds": (
                        _percentile(record_latencies, 99)
                        if len(record_latencies) else np.nan
                    ),
                    **monitor.values(), **stages,
                }
                repeat_rows.append(row)
                runs.append(row)

        end_to_end = [row["end_to_end_seconds"] for row in runs]
        seconds_mean = float(np.mean(end_to_end))
        metrics = metric_values(truth, reference[method_name])
        summary = {
            "method": method_name,
            "artifact_kind": artifact["kind"],
            "mode": args.mode,
            "cpu_threads": args.cpu_threads,
            "input_rows": len(frame),
            "evaluated_rows": evaluated_rows,
            **metrics,
            "model_bytes": int(args.model.stat().st_size),
            "model_load_seconds": load_seconds,
            "model_load_peak_rss_delta_mb": load_monitor.values()[
                "peak_rss_delta_mb"
            ],
            "context_engine": (
                getattr(
                    artifact["model"].get("congestion_config"),
                    "context_engine",
                    "vectorized",
                )
                if artifact["kind"] == "mf-cdr" else "-"
            ),
            "latency_batch_mean_seconds": seconds_mean,
            "latency_batch_p50_seconds": _percentile(end_to_end, 50),
            "latency_batch_p95_seconds": _percentile(end_to_end, 95),
            "latency_batch_p99_seconds": _percentile(end_to_end, 99),
            "latency_run_mean_seconds": seconds_mean,
            "latency_run_p50_seconds": _percentile(end_to_end, 50),
            "latency_run_p95_seconds": _percentile(end_to_end, 95),
            "latency_run_p99_seconds": _percentile(end_to_end, 99),
            "latency_record_mean_us": (
                1e6 * float(np.mean(all_record_latencies))
                if all_record_latencies else np.nan
            ),
            "latency_record_p50_us": (
                1e6 * _percentile(all_record_latencies, 50)
                if all_record_latencies else np.nan
            ),
            "latency_record_p95_us": (
                1e6 * _percentile(all_record_latencies, 95)
                if all_record_latencies else np.nan
            ),
            "latency_record_p99_us": (
                1e6 * _percentile(all_record_latencies, 99)
                if all_record_latencies else np.nan
            ),
            "amortized_latency_p50_us_per_evaluated_row": (
                1e6 * _percentile(end_to_end, 50) / evaluated_rows
            ),
            "amortized_latency_p95_us_per_evaluated_row": (
                1e6 * _percentile(end_to_end, 95) / evaluated_rows
            ),
            "amortized_latency_p99_us_per_evaluated_row": (
                1e6 * _percentile(end_to_end, 99) / evaluated_rows
            ),
            "mean_us_per_input_row": 1e6 * seconds_mean / (
                evaluated_rows if args.mode == "streaming" else len(frame)
            ),
            "mean_us_per_evaluated_row": 1e6 * seconds_mean / evaluated_rows,
            "throughput_input_rows_per_second": (
                evaluated_rows if args.mode == "streaming" else len(frame)
            ) / seconds_mean,
            "throughput_evaluated_rows_per_second": evaluated_rows / seconds_mean,
            "peak_rss_mb": max(row["peak_rss_mb"] for row in runs),
            "peak_rss_delta_mb": max(row["peak_rss_delta_mb"] for row in runs),
            "mean_cpu_core_equivalents": float(np.mean([
                row["cpu_core_equivalents"] for row in runs
            ])),
        }
        stage_names = sorted({
            key for row in runs for key in row
            if key.endswith("_seconds")
            and not key.startswith("record_latency_")
            and key not in {
                "wall_seconds", "process_cpu_seconds", "end_to_end_seconds"
            }
        })
        for stage in stage_names:
            values = [row.get(stage, 0.0) for row in runs]
            summary[f"mean_{stage}"] = float(np.mean(values))
            summary[f"mean_{stage.removesuffix('_seconds')}_us_per_input_row"] = (
                1e6 * float(np.mean(values)) / (
                    evaluated_rows if args.mode == "streaming" else len(frame)
                )
            )
        summaries.append(summary)

    repeats = pd.DataFrame(repeat_rows)
    summary = pd.DataFrame(summaries)
    repeats.to_csv(args.output / "repeats.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    write_json(args.output / "manifest.json", {
        "model": str(args.model), "sample": str(args.sample),
        "sample_identity_sha256": frame_sha256(frame),
        "cpu_only": True, "cpu_threads": args.cpu_threads,
        "inference_mode": args.mode,
        "warmup_runs": args.warmup_runs, "measured_repeats": args.repeats,
        "model_loading_excluded_from_inference": True,
        "ground_truth_not_used_by_inference": True,
        "streaming_policy": (
            "records arrive in capture order; batch size is one; causal state "
            "is retained per sequence; warm-up rows initialize state and are "
            "excluded from scored latency"
        ),
        "hardware": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "total_ram_mb": psutil.virtual_memory().total / (1024.0 ** 2),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
        },
        "mf_parallel_policy": (
            "three expert branches, then three utility branches; dependency "
            "order unchanged and each branch estimator uses one thread"
        ),
        "mf_pipeline_policy": (
            "three bounded stages overlap causal preprocessing, expert-bank, "
            "and utility/meta work across independent capture sequences"
        ),
    })
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
