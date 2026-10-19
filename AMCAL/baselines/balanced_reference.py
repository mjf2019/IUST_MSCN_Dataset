"""Class-balanced reward probes; no change to label acquisition or gradients."""
import numpy as np


def balanced_indices(labels, eligible, size, rng):
    """Draw unique eligible rows round-robin over observed classes only."""
    eligible = np.asarray(eligible, dtype=np.int64)
    if size <= 0 or not len(eligible):
        return np.empty(0, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1)
    groups = [rng.permutation(eligible[labels[eligible] == c]).tolist()
              for c in np.unique(labels[eligible])]
    chosen = []
    limit = min(size, len(eligible))
    while len(chosen) < limit:
        for group_id in rng.permutation(len(groups)):
            if groups[group_id]:
                chosen.append(groups[group_id].pop())
                if len(chosen) == limit:
                    break
    return np.asarray(chosen, dtype=np.int64)


def macro_transfer(before, after, labels, gain_weight, retention_weight, cost):
    """Equal class weight after computing positive gain/damage per example."""
    if before is None:
        return -float(cost), 0.0, 0.0
    before, after = np.asarray(before), np.asarray(after)
    labels = np.asarray(labels).reshape(-1)
    if before.shape != after.shape or before.shape != labels.shape or not len(labels):
        raise ValueError('Reward losses and labels must be nonempty matching vectors.')
    delta = (before - after) / (before + after + 1e-8)
    classes = np.unique(labels)
    gain = float(np.mean([np.maximum(delta[labels == c], 0).mean() for c in classes]))
    harm = float(np.mean([np.maximum(-delta[labels == c], 0).mean() for c in classes]))
    return gain_weight * gain - retention_weight * harm - cost, gain, harm
