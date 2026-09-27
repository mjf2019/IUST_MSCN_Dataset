"""Compare clustering algorithms on the paper-aligned TTFEF router input.

The benchmark is development-only and leakage-safe:
- each capture is truncated to its chronological development prefix;
- causal TTFEF windows are built independently inside each capture;
- exactly 1,000 rows are sampled from every congestion-level/application pair
  when --samples-per-level=5000 (15,000 rows in total);
- congestion labels are never passed to fit or hyperparameter selection;
- labels are used only after fitting for ARI, NMI, and Hungarian-mapped accuracy.

No classifier is trained by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from adaptive_cdr_mlc import APPLICATIONS, LEVELS, STATS, load_dataset, trend_frame
from compare_clean_valid import TIMING
from minibatch_clustering import make_minibatch_kmeans


METHOD_NAMES = {
    "MiniBatchKMeans": "MBK",
    "DBSCAN": "DBS",
    "SpectralClustering": "SPC",
    "GaussianMixture": "GMM",
    "AgglomerativeClustering": "AGG",
}


def development_prefix(data: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return the chronological prefix of every capture."""
    if not 0 < fraction < 1:
        raise ValueError("--development-fraction must be in (0, 1)")
    chunks = []
    for sequence_id, group in data.groupby("sequence_id", sort=False):
        group = group.sort_values(["timestamp", "source_row"], kind="stable")
        stop = int(len(group) * fraction)
        if stop < 3:
            raise ValueError(f"{sequence_id}: development prefix is too short")
        chunks.append(group.iloc[:stop].copy())
    return pd.concat(chunks, axis=0).sort_index(kind="stable")


def causal_trends(development: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build TTFEF without allowing a window to cross a capture boundary."""
    chunks = []
    for _, group in development.groupby("sequence_id", sort=False):
        ordered = group.sort_values(["timestamp", "source_row"], kind="stable")
        view = trend_frame(ordered, TIMING, window)
        if len(view):
            chunks.append(view)
    if not chunks:
        raise ValueError("no complete TTFEF windows were produced")
    result = pd.concat(chunks, axis=0).sort_index(kind="stable")
    columns = [f"{feature}_{stat}" for feature in TIMING for stat in STATS]
    finite = np.isfinite(result[columns].to_numpy(dtype=float)).all(axis=1)
    return result.loc[finite].copy()


def balanced_sample(
    trends: pd.DataFrame,
    samples_per_level: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample equally across all 3 congestion levels and 5 applications."""
    if samples_per_level % len(APPLICATIONS):
        raise ValueError(
            "--samples-per-level must be divisible by the five applications"
        )
    per_pair = samples_per_level // len(APPLICATIONS)
    chunks, audit = [], []
    for level_index, level in enumerate(LEVELS):
        for app_index, application in enumerate(APPLICATIONS):
            group = trends[
                trends.congestion_level.eq(level)
                & trends.traffic_label.eq(application)
            ]
            if len(group) < per_pair:
                raise ValueError(
                    f"{level}/{application}: requested {per_pair}, "
                    f"but only {len(group)} development windows are available"
                )
            random_state = seed + level_index * 100 + app_index
            selected = group.sample(n=per_pair, random_state=random_state)
            chunks.append(selected)
            audit.append({
                "congestion_level": level,
                "application": application,
                "available_development_windows": int(len(group)),
                "sampled_rows": int(len(selected)),
            })
    sample = pd.concat(chunks, axis=0)
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sample, pd.DataFrame(audit)


def cluster_count(labels: np.ndarray) -> int:
    return int(len(set(np.asarray(labels).tolist()) - {-1}))


def internal_metrics(
    x: np.ndarray,
    labels: np.ndarray,
    silhouette_sample_size: int,
    seed: int,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    retained = labels != -1
    observed = np.unique(labels[retained])
    result = {
        "clusters": int(len(observed)),
        "noise_fraction": float((~retained).mean()),
        "coverage": float(retained.mean()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if len(observed) < 2 or retained.sum() <= len(observed):
        return result
    x_valid, labels_valid = x[retained], labels[retained]
    result["silhouette"] = float(silhouette_score(
        x_valid,
        labels_valid,
        sample_size=min(silhouette_sample_size, len(x_valid)),
        random_state=seed,
    ))
    result["davies_bouldin"] = float(
        davies_bouldin_score(x_valid, labels_valid)
    )
    result["calinski_harabasz"] = float(
        calinski_harabasz_score(x_valid, labels_valid)
    )
    return result


def mapped_accuracy(
    truth: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, str]]:
    """Map non-noise clusters to levels with Hungarian matching.

    Noise rows remain incorrect, so DBSCAN cannot improve its score by dropping
    difficult records.
    """
    truth = np.asarray(truth, dtype=object)
    labels = np.asarray(labels, dtype=int)
    clusters = sorted(set(labels.tolist()) - {-1})
    if not clusters:
        return 0.0, {}
    matrix = np.zeros((len(clusters), len(LEVELS)), dtype=np.int64)
    for row, cluster in enumerate(clusters):
        mask = labels == cluster
        for column, level in enumerate(LEVELS):
            matrix[row, column] = int(np.sum(mask & (truth == level)))
    rows, columns = linear_sum_assignment(-matrix)
    correct = int(matrix[rows, columns].sum())
    mapping = {
        str(clusters[row]): LEVELS[column]
        for row, column in zip(rows, columns)
    }
    return float(correct / len(truth)), mapping


def evaluate(
    method: str,
    x: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    fit_seconds: float,
    native_predict: bool,
    params: dict,
    silhouette_sample_size: int,
    seed: int,
) -> dict:
    internal = internal_metrics(
        x, labels, silhouette_sample_size, seed
    )
    accuracy, mapping = mapped_accuracy(truth, labels)
    return {
        "method": method,
        "abbreviation": METHOD_NAMES[method],
        "n": int(len(x)),
        **internal,
        "adjusted_rand": float(adjusted_rand_score(truth, labels)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(truth, labels)
        ),
        "mapped_accuracy": accuracy,
        "fit_seconds": float(fit_seconds),
        "native_out_of_sample_predict": bool(native_predict),
        "cluster_to_level_mapping": json.dumps(mapping, sort_keys=True),
        "selected_parameters": json.dumps(params, sort_keys=True),
    }


def timed_fit_predict(model, x: np.ndarray, threads: int):
    with threadpool_limits(limits=threads):
        start = perf_counter()
        if isinstance(model, GaussianMixture):
            model.fit(x)
            labels = model.predict(x)
        else:
            labels = model.fit_predict(x)
        seconds = perf_counter() - start
    return np.asarray(labels, dtype=int), float(seconds)


def select_mbk(
    x: np.ndarray,
    batch_grid: list[int],
    n_init_grid: list[int],
    max_iter_grid: list[int],
    reassignment_grid: list[float],
    silhouette_sample_size: int,
    seed: int,
    threads: int,
) -> tuple[dict, pd.DataFrame, float]:
    """Tune MBK using internal, label-free development metrics only."""
    rows = []
    search_start = perf_counter()
    for batch_size in batch_grid:
        for n_init in n_init_grid:
            for max_iter in max_iter_grid:
                for reassignment_ratio in reassignment_grid:
                    model = MiniBatchKMeans(
                        n_clusters=3,
                        init="k-means++",
                        batch_size=batch_size,
                        n_init=n_init,
                        max_iter=max_iter,
                        reassignment_ratio=reassignment_ratio,
                        random_state=seed,
                    )
                    labels, fit_seconds = timed_fit_predict(
                        model, x, threads
                    )
                    metrics = internal_metrics(
                        x, labels, silhouette_sample_size, seed
                    )
                    rows.append({
                        "batch_size": int(batch_size),
                        "n_init": int(n_init),
                        "max_iter": int(max_iter),
                        "reassignment_ratio": float(reassignment_ratio),
                        "fit_seconds": fit_seconds,
                        **metrics,
                    })
    search_seconds = perf_counter() - search_start
    grid = pd.DataFrame(rows)
    valid = grid[
        grid.clusters.eq(3)
        & grid.silhouette.notna()
        & grid.davies_bouldin.notna()
        & grid.calinski_harabasz.notna()
    ].copy()
    if valid.empty:
        raise ValueError("MBK grid produced no valid three-cluster solution")
    selected = valid.sort_values(
        [
            "silhouette", "davies_bouldin",
            "calinski_harabasz", "fit_seconds",
        ],
        ascending=[False, True, False, True],
        kind="stable",
    ).iloc[0]
    mask = (
        grid.batch_size.eq(selected.batch_size)
        & grid.n_init.eq(selected.n_init)
        & grid.max_iter.eq(selected.max_iter)
        & np.isclose(
            grid.reassignment_ratio,
            selected.reassignment_ratio,
        )
    )
    grid["selected"] = mask
    params = {
        "batch_size": int(selected.batch_size),
        "n_init": int(selected.n_init),
        "max_iter": int(selected.max_iter),
        "reassignment_ratio": float(selected.reassignment_ratio),
    }
    return params, grid, float(search_seconds)


def select_dbscan(
    x: np.ndarray,
    eps_grid: list[float],
    min_samples: int,
    silhouette_sample_size: int,
    seed: int,
    threads: int,
) -> tuple[float, pd.DataFrame, float]:
    """Select DBSCAN with label-free development metrics only."""
    rows = []
    search_start = perf_counter()
    for eps in eps_grid:
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=threads)
        labels, fit_seconds = timed_fit_predict(model, x, threads)
        metrics = internal_metrics(
            x, labels, silhouette_sample_size, seed
        )
        rows.append({
            "eps": float(eps),
            "min_samples": int(min_samples),
            "fit_seconds": fit_seconds,
            **metrics,
        })
    search_seconds = perf_counter() - search_start
    grid = pd.DataFrame(rows)
    valid = grid[
        grid.clusters.between(2, 10)
        & grid.coverage.ge(0.50)
        & grid.silhouette.notna()
    ].copy()
    if valid.empty:
        raise ValueError(
            "DBSCAN grid produced no configuration with 2-10 clusters and "
            "at least 50% coverage; extend --dbscan-eps-grid"
        )
    exactly_three = valid[valid.clusters.eq(3)]
    pool = exactly_three if len(exactly_three) else valid
    selected = pool.sort_values(
        ["silhouette", "noise_fraction", "fit_seconds"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]
    grid["selected"] = np.isclose(grid.eps, selected.eps)
    return float(selected.eps), grid, float(search_seconds)


def print_results(results: pd.DataFrame):
    print("\nColumn abbreviations")
    meanings = [
        ("M", "Clustering method"),
        ("N", "Evaluated development rows"),
        ("K", "Detected non-noise clusters"),
        ("Noise", "Noise fraction"),
        ("Sil", "Silhouette score (higher is better)"),
        ("DB", "Davies-Bouldin index (lower is better)"),
        ("CH", "Calinski-Harabasz score (higher is better)"),
        ("ARI", "Adjusted Rand index against held-back level labels"),
        ("NMI", "Normalized mutual information"),
        ("MAcc", "Hungarian-mapped level accuracy; noise is incorrect"),
        ("FitS", "Fit and assignment time in seconds"),
        ("Pred", "Native out-of-sample prediction support"),
    ]
    print(pd.DataFrame(meanings, columns=["Key", "Meaning"]).to_string(index=False))
    view = results[[
        "abbreviation", "n", "clusters", "noise_fraction", "silhouette",
        "davies_bouldin", "calinski_harabasz", "adjusted_rand",
        "normalized_mutual_info", "mapped_accuracy", "fit_seconds",
        "native_out_of_sample_predict",
    ]].copy()
    view.columns = [
        "M", "N", "K", "Noise", "Sil", "DB", "CH", "ARI",
        "NMI", "MAcc", "FitS", "Pred",
    ]
    for column in ["Noise", "Sil", "DB", "ARI", "NMI", "MAcc"]:
        view[column] = view[column].map(
            lambda value: "-" if pd.isna(value) else f"{value:.4f}"
        )
    view["CH"] = view["CH"].map(
        lambda value: "-" if pd.isna(value) else f"{value:.1f}"
    )
    view["FitS"] = view["FitS"].map(lambda value: f"{value:.2f}")
    view["Pred"] = view["Pred"].map({True: "Yes", False: "No"})
    print("\nResults")
    print(view.to_string(index=False))


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare five clustering methods on identical TTFEF inputs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "DATASETS/CDR-MLC/Clean_Valid",
    )
    parser.add_argument("--development-fraction", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--samples-per-level", type=int, default=5000)
    parser.add_argument(
        "--mbk-batch-grid",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--mbk-n-init-grid",
        type=int,
        nargs="+",
        default=[10, 20, 50],
    )
    parser.add_argument(
        "--mbk-max-iter-grid",
        type=int,
        nargs="+",
        default=[100, 200],
    )
    parser.add_argument(
        "--mbk-reassignment-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.01],
    )
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument(
        "--dbscan-eps-grid",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00],
    )
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--spectral-neighbors", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs/clustering_algorithm_comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")
    if args.samples_per_level < len(APPLICATIONS):
        raise ValueError("--samples-per-level is too small")
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    data, load_audit = load_dataset(args.data_dir, tuple(TIMING))
    development = development_prefix(data, args.development_fraction)
    trends = causal_trends(development, args.window)
    sample, sample_audit = balanced_sample(
        trends, args.samples_per_level, args.seed
    )
    trend_columns = [
        f"{feature}_{stat}" for feature in TIMING for stat in STATS
    ]
    scaler = StandardScaler()
    x = scaler.fit_transform(sample[trend_columns])
    truth = sample.congestion_level.astype(str).to_numpy()

    results = []

    selected_mbk, mbk_grid, mbk_search_seconds = select_mbk(
        x=x,
        batch_grid=args.mbk_batch_grid,
        n_init_grid=args.mbk_n_init_grid,
        max_iter_grid=args.mbk_max_iter_grid,
        reassignment_grid=args.mbk_reassignment_grid,
        silhouette_sample_size=min(
            2000, args.silhouette_sample_size
        ),
        seed=args.seed,
        threads=args.threads,
    )
    mbk = MiniBatchKMeans(
        n_clusters=3,
        init="k-means++",
        random_state=args.seed,
        **selected_mbk,
    )
    labels, seconds = timed_fit_predict(mbk, x, args.threads)
    results.append(evaluate(
        "MiniBatchKMeans", x, truth, labels, seconds, True,
        {
            **selected_mbk,
            "selection": (
                "label-free silhouette, then Davies-Bouldin, "
                "Calinski-Harabasz, and fit time"
            ),
        },
        args.silhouette_sample_size,
        args.seed,
    ))

    selected_eps, dbscan_grid, dbscan_search_seconds = select_dbscan(
        x=x,
        eps_grid=args.dbscan_eps_grid,
        min_samples=args.dbscan_min_samples,
        silhouette_sample_size=min(2000, args.silhouette_sample_size),
        seed=args.seed,
        threads=args.threads,
    )
    dbscan = DBSCAN(
        eps=selected_eps,
        min_samples=args.dbscan_min_samples,
        n_jobs=args.threads,
    )
    labels, seconds = timed_fit_predict(dbscan, x, args.threads)
    results.append(evaluate(
        "DBSCAN", x, truth, labels, seconds, False,
        {
            "eps": selected_eps,
            "min_samples": args.dbscan_min_samples,
            "selection": "best label-free silhouette; prefer exactly 3 clusters",
        },
        args.silhouette_sample_size,
        args.seed,
    ))

    spectral = SpectralClustering(
        n_clusters=3,
        affinity="nearest_neighbors",
        n_neighbors=args.spectral_neighbors,
        assign_labels="kmeans",
        n_init=10,
        random_state=args.seed,
        n_jobs=args.threads,
    )
    labels, seconds = timed_fit_predict(spectral, x, args.threads)
    results.append(evaluate(
        "SpectralClustering", x, truth, labels, seconds, False,
        spectral.get_params(deep=False),
        args.silhouette_sample_size,
        args.seed,
    ))

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=5,
        reg_covar=1e-6,
        random_state=args.seed,
    )
    labels, seconds = timed_fit_predict(gmm, x, args.threads)
    results.append(evaluate(
        "GaussianMixture", x, truth, labels, seconds, True,
        gmm.get_params(deep=False),
        args.silhouette_sample_size,
        args.seed,
    ))

    agglomerative = AgglomerativeClustering(
        n_clusters=3,
        linkage="ward",
    )
    labels, seconds = timed_fit_predict(agglomerative, x, args.threads)
    results.append(evaluate(
        "AgglomerativeClustering", x, truth, labels, seconds, False,
        agglomerative.get_params(deep=False),
        args.silhouette_sample_size,
        args.seed,
    ))

    result_frame = pd.DataFrame(results)
    ordered = [
        "MiniBatchKMeans", "DBSCAN", "SpectralClustering",
        "GaussianMixture", "AgglomerativeClustering",
    ]
    result_frame["method_order"] = result_frame.method.map(
        {method: index for index, method in enumerate(ordered)}
    )
    result_frame = result_frame.sort_values("method_order").drop(
        columns="method_order"
    )

    args.output.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(
        args.output / "clustering_comparison.csv", index=False
    )
    mbk_grid.to_csv(args.output / "mbk_grid.csv", index=False)
    dbscan_grid.to_csv(args.output / "dbscan_grid.csv", index=False)
    sample_audit.to_csv(args.output / "sample_audit.csv", index=False)

    manifest = {
        "data_dir": str(args.data_dir),
        "development_fraction": args.development_fraction,
        "window": args.window,
        "raw_router_features": list(TIMING),
        "trend_statistics": list(STATS),
        "trend_dimensions": len(trend_columns),
        "trend_columns": trend_columns,
        "samples_per_level": args.samples_per_level,
        "samples_per_application_per_level": (
            args.samples_per_level // len(APPLICATIONS)
        ),
        "total_sampled_rows": len(sample),
        "seed": args.seed,
        "threads": args.threads,
        "labels_used_for_fitting": False,
        "labels_used_for_dbscan_selection": False,
        "labels_used_for_mbk_selection": False,
        "mbk_search_seconds": mbk_search_seconds,
        "mbk_selected_parameters": selected_mbk,
        "labels_used_only_for_external_evaluation": True,
        "scaler_fit_scope": "sampled development rows only",
        "dbscan_search_seconds": dbscan_search_seconds,
        "dbscan_selected_eps": selected_eps,
        "load_audit": load_audit.to_dict(orient="records"),
    }
    (args.output / "clustering_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print_results(result_frame)
    print(f"\nSaved outputs to {args.output}")


if __name__ == "__main__":
    main()
