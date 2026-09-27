"""CDR-MLC with a congestion-oriented, label-free router feature space.

The three experts and hard MiniBatchKMeans routing of fixed CDR-MLC are preserved.
Only the observations supplied to MiniBatchKMeans are changed. Every engineered
quantity is computed from a causal trailing window and is available at test
time without reading traffic_label or congestion_level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from adaptive_cdr_mlc import APPLICATIONS, aligned_probabilities, make_preprocessor, select_classifier_columns
from compare_clean_valid import TIMING, rf_config
from minibatch_clustering import make_minibatch_kmeans, minibatch_kmeans_audit


DEFAULT_CONGESTION_FEATURES = (
    "TcpRtt", "SynAck", "AckDat",
    "SrcLoad", "DstLoad", "Load",
    "SrcRate", "DstRate", "Rate",
    "pLoss", "SrcLoss", "DstLoss", "Loss",
)


@dataclass(frozen=True)
class CongestionRouterConfig:
    window: int = 10
    features: tuple[str, ...] = DEFAULT_CONGESTION_FEATURES
    expert_trees: int = 20
    random_state: int = 42
    n_clusters: int = 3
    batch_size: int = 1024
    n_init: int = 20
    max_iter: int = 200
    epsilon: float = 1e-9
    # ``vectorized`` is numerically equivalent to the original pandas
    # implementation, but evaluates every feature in a segment together and
    # removes the Python callback used for each rolling slope.  ``pandas`` is
    # retained as a reproducibility/reference engine.
    context_engine: str = "vectorized"
    context_block_rows: int = 16384

    def validate(self):
        if self.window < 3:
            raise ValueError("window must be at least 3 for slope extraction")
        if len(set(self.features)) < 3:
            raise ValueError("at least three router features are required")
        if self.expert_trees < 1 or self.n_clusters != 3:
            raise ValueError("positive expert_trees and exactly three clusters are required")
        if self.context_engine not in {"vectorized", "pandas"}:
            raise ValueError("context_engine must be 'vectorized' or 'pandas'")
        if self.context_block_rows < 1:
            raise ValueError("context_block_rows must be positive")
        return self


def _rolling_slope(values: np.ndarray) -> float:
    """Least-squares slope with a fixed, local time axis."""
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    denominator = np.dot(x, x)
    return float(np.dot(x, y - y.mean()) / denominator) if denominator else 0.0


def _context_schema(frame: pd.DataFrame, config: CongestionRouterConfig):
    available = [feature for feature in config.features if feature in frame.columns]
    if len(available) < 3:
        raise ValueError(f"only {len(available)} requested congestion features exist")
    stats = ("log_median", "cv", "iqr_ratio", "range_ratio", "delta_ratio", "slope_ratio")
    columns = [f"router_{feature}_{stat}" for feature in available for stat in stats]
    return available, columns


def _linear_quartiles(windows: np.ndarray):
    """Return q25, q50 and q75 using NumPy/pandas linear interpolation.

    A single partial partition replaces three independent rolling quantile
    operations.  Only the six order statistics needed for linear
    interpolation are materialised.
    """
    width = windows.shape[-1]
    positions = np.asarray((.25, .50, .75), dtype=float) * (width - 1)
    lower = np.floor(positions).astype(np.intp)
    upper = np.ceil(positions).astype(np.intp)
    kth = np.unique(np.concatenate([lower, upper]))
    partitioned = np.partition(windows, kth=kth, axis=-1)
    fraction = positions - lower
    quantiles = []
    for lo, hi, weight in zip(lower, upper, fraction):
        low_value = partitioned[..., lo]
        high_value = partitioned[..., hi]
        quantiles.append(low_value + (high_value - low_value) * weight)
    return quantiles


def _vectorized_window_descriptors(
    values: np.ndarray,
    window: int,
    epsilon: float,
    block_rows: int,
):
    """Yield feature-major congestion descriptors for one valid segment.

    The sliding-window view itself is zero-copy.  Work is bounded to
    ``block_rows`` output rows so the partial partition used for quartiles
    cannot create an unbounded temporary array on long captures.
    """
    count = len(values) - window + 1
    if count <= 0:
        return
    windows = np.lib.stride_tricks.sliding_window_view(
        np.ascontiguousarray(values, dtype=np.float64),
        window_shape=window,
        axis=0,
    )
    # Shape is (output rows, features, window).  The fixed centred time axis
    # makes least-squares slopes a single batched dot product.
    x = np.arange(window, dtype=np.float64)
    x -= x.mean()
    slope_denominator = float(np.dot(x, x))

    for start in range(0, count, block_rows):
        stop = min(start + block_rows, count)
        block = windows[start:stop]
        q25, median, q75 = _linear_quartiles(block)
        mean = block.mean(axis=-1)
        std = block.std(axis=-1, ddof=0)
        minimum = block.min(axis=-1)
        maximum = block.max(axis=-1)
        scale = np.abs(median) + epsilon
        slope = np.einsum("bfw,w->bf", block, x, optimize=True)
        slope /= slope_denominator

        output = np.empty(
            (len(block), values.shape[1], 6), dtype=np.float64
        )
        output[..., 0] = np.log1p(np.maximum(median, 0.0))
        output[..., 1] = std / (np.abs(mean) + epsilon)
        output[..., 2] = (q75 - q25) / scale
        output[..., 3] = (maximum - minimum) / scale
        output[..., 4] = (
            block[..., -1] - block[..., 0]
        ) / scale
        output[..., 5] = slope / scale
        yield start, stop, output.reshape(len(block), -1)


def _vectorized_context_values(
    frame: pd.DataFrame,
    config: CongestionRouterConfig,
    available: list[str],
    columns: list[str],
):
    """Return sorted eligible indices and their context feature matrix."""
    # Numeric conversion is performed once for the complete frame rather than
    # once per sequence and again per valid segment.
    selected = frame[available]
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in selected.dtypes):
        numeric = selected
    else:
        numeric = selected.apply(pd.to_numeric, errors="coerce")
    index_chunks = []
    value_chunks = []
    block_rows = int(getattr(config, "context_block_rows", 16384))
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        group_numeric = numeric.loc[group.index]
        values = group_numeric.to_numpy(dtype=np.float64, copy=False)
        valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)

        # Locate maximal contiguous valid runs.  Invalid rows split windows,
        # exactly matching the reference pandas implementation.
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for run_start, run_stop in edges.reshape(-1, 2):
            if run_stop - run_start < config.window:
                continue
            segment_values = values[run_start:run_stop]
            first_output = run_start + config.window - 1
            for start, stop, descriptors in _vectorized_window_descriptors(
                segment_values,
                config.window,
                config.epsilon,
                block_rows,
            ):
                finite = np.isfinite(descriptors).all(axis=1)
                if not finite.any():
                    continue
                row_start = first_output + start
                row_stop = first_output + stop
                selected_index = group.iloc[row_start:row_stop].index[finite]
                index_chunks.append(selected_index.to_numpy(copy=True))
                value_chunks.append(descriptors[finite])

    if not index_chunks:
        return frame.index[:0], np.empty((0, len(columns)), dtype=np.float64)
    context = pd.DataFrame(
        np.concatenate(value_chunks, axis=0),
        index=pd.Index(np.concatenate(index_chunks)),
        columns=columns,
    ).sort_index(kind="stable")
    return context.index, context.to_numpy(dtype=np.float64, copy=False)


def _congestion_feature_frame_vectorized(
    frame: pd.DataFrame,
    config: CongestionRouterConfig,
    available: list[str],
    columns: list[str],
):
    """Vectorized, memory-bounded implementation of causal context features."""
    index, values = _vectorized_context_values(
        frame, config, available, columns
    )
    if len(index) == 0:
        empty = frame.iloc[:0].copy()
        return empty.assign(**{
            column: pd.Series(dtype=float) for column in columns
        }), available
    context = pd.DataFrame(values, index=index, columns=columns)
    return pd.concat([frame.loc[index].copy(), context], axis=1), available


def _congestion_feature_frame_pandas(
    frame: pd.DataFrame,
    config: CongestionRouterConfig,
    available: list[str],
    columns: list[str],
):
    """Original implementation kept as an equivalence reference/fallback."""
    chunks = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable").copy()
        numeric = group[available].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(numeric).all(axis=1) & numeric.ge(0).all(axis=1)
        segments = (~valid).cumsum()
        for _, segment in group.loc[valid].groupby(segments[valid], sort=False):
            values = segment[available].apply(pd.to_numeric, errors="coerce")
            output = {}
            for feature in available:
                series = values[feature]
                rolling = series.rolling(config.window, min_periods=config.window)
                median = rolling.median()
                mean = rolling.mean()
                std = rolling.std(ddof=0)
                minimum = rolling.min()
                maximum = rolling.max()
                q25 = rolling.quantile(.25)
                q75 = rolling.quantile(.75)
                first = series.shift(config.window - 1)
                scale = median.abs() + config.epsilon
                output[f"router_{feature}_log_median"] = np.log1p(median.clip(lower=0))
                output[f"router_{feature}_cv"] = std / (mean.abs() + config.epsilon)
                output[f"router_{feature}_iqr_ratio"] = (q75 - q25) / scale
                output[f"router_{feature}_range_ratio"] = (maximum - minimum) / scale
                output[f"router_{feature}_delta_ratio"] = (series - first) / scale
                slope = rolling.apply(_rolling_slope, raw=True)
                output[f"router_{feature}_slope_ratio"] = slope / scale
            matrix = pd.DataFrame(output, index=segment.index)
            matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna()
            if len(matrix):
                joined = segment.loc[matrix.index].copy()
                joined[columns] = matrix[columns]
                chunks.append(joined)
    if not chunks:
        empty = frame.iloc[:0].copy()
        return empty.assign(**{column: pd.Series(dtype=float) for column in columns}), available
    return pd.concat(chunks).sort_index(kind="stable"), available


def congestion_feature_frame(frame: pd.DataFrame, config: CongestionRouterConfig):
    """Build causal absolute + relative congestion descriptors per capture.

    ``log_median`` retains congestion magnitude.  The remaining fields describe
    short-term dynamics while dividing by the local magnitude, which reduces
    application/flow-scale bias without using the unknown application label.
    Invalid observations break a window rather than being imputed across time.
    """
    available, columns = _context_schema(frame, config)
    engine = getattr(config, "context_engine", "vectorized")
    if engine == "pandas":
        return _congestion_feature_frame_pandas(
            frame, config, available, columns
        )
    return _congestion_feature_frame_vectorized(
        frame, config, available, columns
    )


def congestion_feature_values(frame: pd.DataFrame, config: CongestionRouterConfig):
    """Return only index/columns/values for latency-sensitive inference.

    Unlike :func:`congestion_feature_frame`, this API does not copy all raw
    input columns into an intermediate DataFrame.  The original pandas engine
    remains supported for exact reference comparisons.
    """
    available, columns = _context_schema(frame, config)
    engine = getattr(config, "context_engine", "vectorized")
    if engine == "pandas":
        view, _ = _congestion_feature_frame_pandas(
            frame, config, available, columns
        )
        return view.index, columns, view[columns].to_numpy(dtype=float)
    index, values = _vectorized_context_values(
        frame, config, available, columns
    )
    return index, columns, values


def fit_congestion_cdr(source: pd.DataFrame, config: CongestionRouterConfig):
    config.validate()
    view, available = congestion_feature_frame(source, config)
    router_columns = [column for column in view.columns if column.startswith("router_")]
    if len(view) < config.n_clusters:
        raise ValueError("fewer than three complete router windows")
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(view[router_columns])
    z = scaler.transform(view[router_columns])
    router = make_minibatch_kmeans(
        n_clusters=config.n_clusters, batch_size=config.batch_size,
        n_init=config.n_init, max_iter=config.max_iter,
        random_state=config.random_state,
    ).fit(z)
    routes = router.predict(z)
    if len(np.unique(routes)) != config.n_clusters:
        raise ValueError("congestion router produced fewer than three clusters")

    raw = source.loc[view.index]
    # Keep expert inputs identical to fixed CDR-MLC.  This makes the experiment
    # isolate the router feature-space change instead of silently weakening the
    # experts whenever more congestion candidates are added.
    numeric, categorical = select_classifier_columns(raw, TIMING)
    preprocessor = make_preprocessor(numeric, categorical)
    x = preprocessor.fit_transform(raw[numeric + categorical])
    labels = raw.traffic_label.to_numpy()
    experts, cluster_counts = {}, []
    for cluster in range(config.n_clusters):
        mask = routes == cluster
        experts[cluster] = RandomForestClassifier(
            **rf_config(config.random_state + cluster, config.expert_trees)
        ).fit(x[mask], labels[mask])
        counts = pd.Series(labels[mask]).value_counts().to_dict()
        levels = pd.Series(raw.congestion_level.to_numpy()[mask]).value_counts().to_dict()
        cluster_counts.append({
            "cluster": cluster, "rows": int(mask.sum()),
            **{f"class_{label}": int(counts.get(label, 0)) for label in APPLICATIONS},
            **{f"level_{level}": int(levels.get(level, 0)) for level in ("Low", "Medium", "High")},
        })
    return {
        "config": config, "window": config.window, "route_features": available,
        "router_columns": router_columns, "scaler": scaler, "router": router,
        "router_audit": minibatch_kmeans_audit(router),
        "preprocessor": preprocessor, "numeric": numeric, "categorical": categorical,
        "experts": experts, "source_eligible_index": view.index.tolist(),
        "cluster_counts": cluster_counts,
    }


def expert_outputs(model: dict, frame: pd.DataFrame):
    view, _ = congestion_feature_frame(frame, model["config"])
    raw = frame.loc[view.index]
    z = model["scaler"].transform(view[model["router_columns"]])
    distances = model["router"].transform(z)
    routes = model["router"].predict(z)
    columns = model["numeric"] + model["categorical"]
    x = model["preprocessor"].transform(raw[columns])
    probabilities, predictions = [], []
    for cluster in range(3):
        probability = aligned_probabilities(model["experts"][cluster], x, APPLICATIONS)
        probabilities.append(probability)
        predictions.append(np.asarray(APPLICATIONS)[probability.argmax(axis=1)])
    return (
        raw, distances, routes,
        np.stack(probabilities, axis=1), np.stack(predictions, axis=1),
    )


def oracle_route(truth, probabilities, predictions):
    correct = predictions == truth[:, None]
    true_index = np.array([APPLICATIONS.index(label) for label in truth])
    true_probability = probabilities[
        np.arange(len(truth))[:, None], np.arange(3)[None, :], true_index[:, None]
    ]
    return np.where(correct, true_probability + 2.0, true_probability).argmax(axis=1)


def predict_all(model: dict, frame: pd.DataFrame):
    raw, distances, routes, probabilities, predictions = expert_outputs(model, frame)
    truth = raw.traffic_label.to_numpy()
    oracle = oracle_route(truth, probabilities, predictions)
    row = np.arange(len(raw))
    report = pd.DataFrame({
        "congestion_route": routes,
        "oracle_route": oracle,
        "router_matches_oracle": routes == oracle,
        "nearest_distance": distances.min(axis=1),
        "distance_margin": np.partition(distances, 1, axis=1)[:, 1] - distances.min(axis=1),
    }, index=raw.index)
    return {
        "observed": raw,
        "CDR_MLC_congestion_router": predictions[row, routes],
        "CDR_MLC_oracle_router": predictions[row, oracle],
        "routes": report,
    }
