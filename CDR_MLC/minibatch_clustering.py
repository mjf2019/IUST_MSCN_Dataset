"""Canonical MiniBatchKMeans construction for CDR-MLC routers.

The paper uses Mini-Batch K-Means (MBK), not full-batch KMeans.  Keeping its
construction in one module prevents experimental extensions from silently
changing the clustering algorithm or its initialization policy.
"""

from __future__ import annotations

from sklearn.cluster import MiniBatchKMeans


def make_minibatch_kmeans(
    *,
    n_clusters: int = 3,
    batch_size: int = 1024,
    n_init: int = 10,
    max_iter: int = 100,
    random_state: int = 42,
) -> MiniBatchKMeans:
    """Return the paper-aligned MiniBatchKMeans router."""
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
        reassignment_ratio=.01,
        random_state=random_state,
    )


def minibatch_kmeans_audit(router: MiniBatchKMeans) -> dict:
    """Validate and serialize the fitted router's relevant configuration."""
    if not isinstance(router, MiniBatchKMeans):
        raise TypeError(
            "CDR-MLC clustering router must be sklearn MiniBatchKMeans; "
            f"received {type(router).__name__}"
        )
    params = router.get_params(deep=False)
    return {
        "algorithm": "MiniBatchKMeans",
        "n_clusters": int(params["n_clusters"]),
        "init": params["init"],
        "batch_size": int(params["batch_size"]),
        "n_init": int(params["n_init"]),
        "max_iter": int(params["max_iter"]),
        "reassignment_ratio": float(params["reassignment_ratio"]),
        "random_state": int(params["random_state"]),
    }
