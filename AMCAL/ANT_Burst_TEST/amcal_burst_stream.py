"""Offline stream interleaving; never used as a model/selection input."""
from collections import deque
import hashlib
import random


def stream_permutation(labels, order="class-interleaved", seed=42):
    """Shuffle class slots, consuming each class's original FIFO queue.

    Counts and selected rows are preserved. Only order changes. Independent RNG
    keeps exploration/query RNG untouched. Same label sequence + seed => same order.
    """
    if order == "input":
        return list(range(len(labels)))
    if order != "class-interleaved":
        raise ValueError("Unsupported stream order.")
    queues = {}
    slots = []
    for index, label in enumerate(labels):
        queues.setdefault(label, deque()).append(index)
        slots.append(label)
    random.Random(seed).shuffle(slots)
    return [queues[label].popleft() for label in slots]


def permutation_sha256(indices):
    return hashlib.sha256(",".join(map(str, indices)).encode("ascii")).hexdigest()


class ClassInterleavedSampler:
    """One pass over all existing rows; a reproducible fresh merge each epoch."""
    def __init__(self, labels, seed=42):
        self.labels = list(labels)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        indices = stream_permutation(self.labels, seed=self.seed + self.epoch)
        self.epoch += 1
        return iter(indices)
