"""DBSCAN router adapter for controlled CDR-MLC ablation experiments.

DBSCAN has no native out-of-sample ``predict`` operation.  This adapter fits
DBSCAN on development-only TTFEF vectors, selects a label-free three-cluster
solution, and assigns later records through their nearest fitted core sample.
Records outside ``eps`` use the same nearest-core rule as an explicit fallback;
this guarantees that every record is routed to one of the three experts.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


class DBSCANRouter:
    def __init__(
        self,
        *,
        n_clusters=3,
        batch_size=1024,
        n_init=10,
        max_iter=100,
        random_state=42,
        min_samples=10,
        eps_grid=(0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00),
    ):
        if n_clusters != 3:
            raise ValueError("DBSCAN CDR-MLC ablation requires exactly 3 experts")
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.min_samples = int(min_samples)
        self.eps_grid = tuple(float(value) for value in eps_grid)

    def _candidate_eps(self, x):
        neighbors = NearestNeighbors(n_neighbors=min(self.min_samples, len(x)))
        distances, _ = neighbors.fit(x).kneighbors(x)
        k_distance = distances[:, -1]
        derived = np.quantile(
            k_distance,
            [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        )
        return sorted(set([*self.eps_grid, *np.round(derived, 6).tolist()]))

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if len(x) <= self.min_samples:
            raise ValueError("insufficient rows for DBSCAN router")
        candidates = []
        for eps in self._candidate_eps(x):
            model = DBSCAN(eps=eps, min_samples=self.min_samples, n_jobs=1)
            labels = model.fit_predict(x)
            retained = labels >= 0
            clusters = sorted(set(labels[retained].tolist()))
            if len(clusters) != self.n_clusters or retained.sum() <= 3:
                continue
            score_sample = min(5000, int(retained.sum()))
            silhouette = silhouette_score(
                x[retained],
                labels[retained],
                sample_size=score_sample,
                random_state=self.random_state,
            )
            candidates.append((
                float(silhouette),
                float(retained.mean()),
                -float(eps),
                model,
                labels,
            ))
        if not candidates:
            raise ValueError(
                "DBSCAN found no label-free three-cluster solution on the "
                "development partition"
            )
        _, coverage, _, model, labels = max(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        raw_clusters = sorted(set(labels.tolist()) - {-1})
        remap = {cluster: index for index, cluster in enumerate(raw_clusters)}
        core_raw_labels = labels[model.core_sample_indices_]
        keep = core_raw_labels >= 0
        self.core_samples_ = x[model.core_sample_indices_][keep]
        self.core_labels_ = np.asarray(
            [remap[int(label)] for label in core_raw_labels[keep]], dtype=int
        )
        if len(self.core_samples_) == 0:
            raise ValueError("DBSCAN router has no usable core samples")
        self.neighbor_index_ = NearestNeighbors(n_neighbors=1).fit(
            self.core_samples_
        )
        self.cluster_centers_ = np.vstack([
            self.core_samples_[self.core_labels_ == cluster].mean(axis=0)
            for cluster in range(self.n_clusters)
        ])
        self.eps_ = float(model.eps)
        self.coverage_ = float(coverage)
        self.noise_fraction_ = 1.0 - self.coverage_
        self.silhouette_ = float(max(candidates, key=lambda item: (
            item[0], item[1], item[2]
        ))[0])
        self.labels_ = self.predict(x)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        _, indices = self.neighbor_index_.kneighbors(x)
        return self.core_labels_[indices[:, 0]]

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return np.linalg.norm(
            x[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )


def make_dbscan_router(**kwargs):
    return DBSCANRouter(**kwargs)


def dbscan_router_audit(router):
    if not isinstance(router, DBSCANRouter):
        raise TypeError(f"expected DBSCANRouter, received {type(router).__name__}")
    return {
        "algorithm": "DBSCAN",
        "out_of_sample_assignment": "nearest fitted core sample",
        "outside_eps_fallback": "nearest fitted core sample",
        "selection": "development-only silhouette; exactly three clusters",
        "eps": router.eps_,
        "min_samples": router.min_samples,
        "silhouette": router.silhouette_,
        "coverage_before_fallback": router.coverage_,
        "noise_fraction_before_fallback": router.noise_fraction_,
        "random_state": router.random_state,
    }
