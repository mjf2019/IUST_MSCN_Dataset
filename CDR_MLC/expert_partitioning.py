"""Label-free, reproducible development partitions for the no-clustering control."""
from __future__ import annotations

import hashlib

import numpy as np


def random_expert_assignments(frame, seed: int, n_experts: int = 3):
    """Hash opaque record identities; never use features or class/level labels.

    Assignments are independent of row order and stay fixed when preliminary
    experts are refitted on the larger authorized development partition. They
    are approximately balanced, not stratified by application or congestion.
    This function is used for training only; inference evaluates every expert.
    """
    if n_experts < 1:
        raise ValueError("n_experts must be positive")
    if frame[["source_file", "source_row"]].isna().any().any():
        raise ValueError("No-clustering partition requires valid record identities")
    if frame.duplicated(["source_file", "source_row"]).any():
        raise ValueError("No-clustering partition requires unique record identities")
    assignments = []
    for name, row in frame[["source_file", "source_row"]].itertuples(index=False, name=None):
        # Length-prefixed source identity avoids ambiguous string concatenation.
        name_bytes = str(name).encode("utf-8")
        identity = (str(seed).encode() + b":" + str(len(name_bytes)).encode()
                    + b":" + name_bytes + b":" + str(int(row)).encode())
        digest = hashlib.sha256(identity).digest()
        assignments.append(int.from_bytes(digest[:8], "big") % n_experts)
    return np.asarray(assignments, dtype=np.int64)
