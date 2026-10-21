"""Causal source / historical / recent reference memory with fixed total capacity."""
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from baselines.balanced_reference import balanced_indices


@dataclass
class ReferenceBatch:
    x: torch.Tensor
    y: torch.Tensor
    segments: np.ndarray

    def protected(self, count, rng):
        eligible = np.flatnonzero(self.segments != 2)
        if count <= 0 or not len(eligible): return None
        ids = balanced_indices(self.y.detach().cpu().numpy(), eligible, count, rng)
        at = torch.as_tensor(ids, device=self.x.device)
        return self.x[at], self.y[at]


def macro(values, labels):
    return float(np.mean([values[labels == c].mean() for c in np.unique(labels)])) if len(labels) else 0.


def transfer(before, after, batch, gain_weight, retention_weight, cost, config):
    if batch is None: return -float(cost), 0., 0.
    delta = (np.asarray(before) - np.asarray(after)) / (np.asarray(before) + np.asarray(after) + 1e-8)
    labels = batch.y.detach().cpu().numpy()
    recent = batch.segments == 2
    old = ~recent
    # The first update has source anchors but no previous target feedback.
    gain_mask = recent if recent.any() else np.ones(len(delta), dtype=bool)
    harm_mask = old if old.any() else np.ones(len(delta), dtype=bool)
    gain = macro(np.maximum(delta[gain_mask], 0), labels[gain_mask])
    harm = macro(np.maximum(-delta[harm_mask], 0), labels[harm_mask])
    excess = max(0., harm - config.damage_limit)
    return gain_weight * gain - retention_weight * harm - cost - config.damage_weight * excess, gain, harm


class ProtectedMemory:
    """Anchors are TRAIN-only; other rows are inserted strictly after paid feedback.

    FIFO evictions enter a separate per-class reservoir. Thus target rows are
    not duplicated in recent and historical storage. Sampling also deduplicates
    feature-identical rows across all partitions and excludes the current input.
    """
    def __init__(self, anchors, classes, config, seed):
        self.config, self.classes = config, classes
        self.rng = np.random.default_rng(seed)
        self.replay_rng = np.random.default_rng(seed + 1729)
        self.source = []
        self.recent = deque()
        self.historical = [[] for _ in range(classes)]
        self.seen = np.zeros(classes, dtype=np.int64)
        x, y = anchors
        y = np.asarray(y, dtype=np.int64)
        if len(x) != len(y): raise ValueError('Anchor features and labels must align.')
        cpu_x = x.detach().cpu().contiguous()
        unique = []; known = set()
        for i, features in enumerate(cpu_x):
            key = features.numpy().tobytes()
            if key not in known: unique.append(i); known.add(key)
        ids = balanced_indices(y, unique, config.source_capacity, self.rng)
        self.source = [self._row(cpu_x[int(i):int(i)+1], int(y[int(i)])) for i in ids]

    def _row(self, x, y):
        if not 0 <= y < self.classes: raise ValueError('Label is outside the source class encoding.')
        x = x.detach().cpu().clone().contiguous()
        return x, y, x.numpy().tobytes()

    def _archive(self, row):
        label = row[1]
        capacity = self.config.historical_capacity // self.classes + int(label < self.config.historical_capacity % self.classes)
        group = self.historical[label]
        if capacity == 0 or any(old[2] == row[2] for old in group): return
        self.seen[label] += 1
        if len(group) < capacity: group.append(row)
        else:
            index = int(self.rng.integers(self.seen[label]))
            if index < capacity: group[index] = row

    def add(self, x, y):
        row = self._row(x, int(torch.as_tensor(y).reshape(-1)[0]))
        if any(old[2] == row[2] for old in self.source): return
        for old in self.recent:
            if old[2] == row[2]: return
        for group in self.historical:
            group[:] = [old for old in group if old[2] != row[2]]
        if self.config.recent_capacity:
            self.recent.append(row)
            if len(self.recent) > self.config.recent_capacity: self._archive(self.recent.popleft())
        else: self._archive(row)

    def sample(self, current):
        current_key = current.detach().cpu().contiguous().numpy().tobytes()
        pools = [self.source, [row for group in self.historical for row in group], list(self.recent)]
        eligible, seen = [], {current_key}
        for segment, pool in enumerate(pools):
            for row in pool:
                if row[2] not in seen:
                    eligible.append((row, segment)); seen.add(row[2])
        if not eligible: return None
        labels = np.array([row[0][1] for row in eligible])
        segments = np.array([row[1] for row in eligible])
        count = self.config.reference_size
        quotas = [count // 4, count // 4, count - 2 * (count // 4)]
        chosen = []
        for segment, quota in enumerate(quotas):
            chosen.extend(balanced_indices(labels, np.flatnonzero(segments == segment), quota, self.rng).tolist())
        remaining = np.setdiff1d(np.arange(len(eligible)), chosen)
        chosen.extend(balanced_indices(labels, remaining, count - len(chosen), self.rng).tolist())
        return ReferenceBatch(torch.cat([eligible[i][0][0] for i in chosen]).to(current.device),
                              torch.tensor(labels[chosen], device=current.device), segments[chosen])

    def counts(self):
        return {'source': len(self.source), 'historical': sum(map(len, self.historical)),
                'recent': len(self.recent),
                'historical_per_class': [len(group) for group in self.historical]}
