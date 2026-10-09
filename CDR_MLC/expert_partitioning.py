"""Independent seeded RNG partitions for the no-clustering control."""
from __future__ import annotations
from copy import deepcopy
import numpy as np


def random_expert_assignments(frame, seed: int, n_experts: int = 3):
    """One independent uniform draw per row, using only row count and seed.

    No filenames, row IDs, features, labels, stratification, balancing or
    rejection/redrawing. Use random_expert_partition to save draws for refits.
    """
    if n_experts < 1:
        raise ValueError("n_experts must be positive")
    return np.random.default_rng(seed).integers(
        0, n_experts, size=len(frame), dtype=np.int64)


def random_expert_partition(frame, seed: int, n_experts: int = 3, previous=None):
    """Retain existing draws; continue the RNG for new development rows.

    Integer DataFrame indices are lookup keys only, never inputs to the RNG.
    No inference/test rows are passed to this training-only function.
    """
    if n_experts < 1:
        raise ValueError("n_experts must be positive")
    if not frame.index.is_unique or any(
        not isinstance(value, (int, np.integer)) for value in frame.index):
        raise ValueError("Random partition requires unique integer row indices")
    keys = [int(value) for value in frame.index]
    rng = np.random.default_rng(seed)
    assignments = {}
    if previous is not None:
        if previous["seed"] != seed or previous["n_experts"] != n_experts:
            raise ValueError("Saved random partition seed/expert count differs")
        assignments = dict(zip(previous["row_indices"], previous["assignments"]))
        if not set(assignments).issubset(keys):
            raise ValueError("Refit must contain all previously assigned rows")
        rng.bit_generator.state = deepcopy(previous["rng_state"])
    new_keys = [key for key in keys if key not in assignments]
    draws = rng.integers(0, n_experts, size=len(new_keys), dtype=np.int64)
    assignments.update(zip(new_keys, map(int, draws)))
    routes = np.asarray([assignments[key] for key in keys], dtype=np.int64)
    state = {
        "seed": int(seed), "n_experts": int(n_experts),
        "row_indices": keys, "assignments": routes.tolist(),
        "rng_state": deepcopy(rng.bit_generator.state),
    }
    return routes, state
