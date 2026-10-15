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


def split_class_budget(total, num_classes):
    if total < 0 or num_classes < 1:
        raise ValueError("Invalid class budget.")
    share, remainder = divmod(total, num_classes)
    return {c: share + int(c < remainder) for c in range(num_classes)}


def class_budget_available(mode, class_id, used, quotas):
    return mode != "per-class" or used[class_id] < quotas[class_id]


def class_budget_report(rows, quotas):
    """Observer-only pre-update accuracy curves; never feeds the learner."""
    counts = {c: [0, 0, 0] for c in quotas}
    curve = []
    for row in rows:
        c = int(row["true_label"])
        counts[c][0] += 1
        counts[c][1] += int(int(row["prediction_before_update"]) == c)
        counts[c][2] += int(bool(row["updated"]))
        n, correct, used = counts[c]
        curve.append(dict(step=int(row["step"]), class_id=c, class_samples_seen=n,
                          class_labels_used=used, accuracy_before_update=correct/n))
    summary = [dict(class_id=c, samples=v[0], labels_used=v[2],
                    label_budget=quotas[c], accuracy_before_update=v[1]/v[0] if v[0] else None)
               for c, v in counts.items()]
    return summary, curve


def fractional_class_budget(labels, num_classes, fraction):
    """Hard cap floor(fraction * class count), computed on adaptation rows only."""
    import math
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or num_classes < 1:
        raise ValueError("Invalid per-class budget fraction.")
    counts = {c: 0 for c in range(num_classes)}
    for label in labels:
        counts[int(label)] += 1
    return {c: math.floor(fraction * count) for c, count in counts.items()}
