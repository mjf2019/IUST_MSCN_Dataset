"""TRAIN-only effect of one real Context update on disjoint fixed probes."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def keys(x):
    return {row.tobytes() for row in x.detach().cpu().contiguous().numpy()}


def signed_update_reward(before, after, forgetting_weight, cost):
    values = [*before, *after, forgetting_weight, cost]
    if len(before) != 2 or len(after) != 2 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Reward measurements must be two finite probe losses.')
    if min(forgetting_weight, cost) < 0:
        raise ValueError('Forgetting weight and request cost must be nonnegative.')
    gain = float(before[0]-after[0])
    harm = max(0., float(after[1]-before[1]))
    reward = gain-forgetting_weight*harm-cost
    return reward, dict(probe_gain=gain, probe_harm=harm,
        weighted_probe_harm=forgetting_weight*harm, request_cost=float(cost),
        generalization_loss_before=float(before[0]), generalization_loss_after=float(after[0]),
        retention_loss_before=float(before[1]), retention_loss_after=float(after[1]))


class FixedProbeReward:
    def __init__(self, generalization, retention, forgetting_weight=1., cost=.001):
        if not generalization or not all(len(d[1]) for d in [*generalization, retention]):
            raise ValueError('Nonempty fixed TRAIN probes are required.')
        signed_update_reward((0., 0.), (0., 0.), forgetting_weight, cost)
        self.generalization, self.retention = generalization, retention
        self.forgetting_weight, self.cost = forgetting_weight, cost
        self.blocked_keys = set().union(*(keys(d[0]) for d in [*generalization, retention]))
        self.calls = 0

    @staticmethod
    def macro_loss(cnn, context, data):
        x, y = data[:2]
        truth = torch.as_tensor(y, dtype=torch.long, device=x.device)
        losses = F.cross_entropy(cnn(context(x)), truth, reduction='none')
        return torch.stack([losses[truth == label].mean() for label in truth.unique()]).mean()

    @torch.no_grad()
    def measure(self, cnn, context):
        if cnn.training or context.training:
            raise ValueError('Probe observation requires eval mode and frozen running statistics.')
        generalization = torch.stack([self.macro_loss(cnn, context, d) for d in self.generalization]).mean()
        retention = self.macro_loss(cnn, context, self.retention)
        result = float(generalization), float(retention)
        if not all(math.isfinite(v) for v in result): raise ValueError('Nonfinite TRAIN probe loss.')
        self.calls += 1
        return result

    def finish(self, cnn, context, before):
        return signed_update_reward(before, self.measure(cnn, context), self.forgetting_weight, self.cost)

    def assert_disjoint_update(self, current, paid_rows, replay):
        batches = [current, *paid_rows]
        if replay is not None: batches.append(replay)
        if any(keys(x).intersection(self.blocked_keys) for x, _ in batches):
            raise ValueError('A TRAIN reward-probe feature entered the actual update batch.')


def prepare_train_probes(experiment, api, rows=24, retention_rows=32):
    """Keep sampled streams unchanged; reserve probes outside all sampled streams.

    Probe IDs and feature duplicates are excluded across TRAIN intensities and
    removed from TRAIN anchors. Evaluation keeps its original anchors and rows.
    Probes may have appeared in historical base pretraining; they are held out
    from the current experiment's adaptive updates, not from all past training.
    """
    a = experiment.a
    if rows < 3*len(a.classes) or retention_rows < 3*len(a.classes):
        raise ValueError('Probe limits must retain up to three examples per class.')
    stream_ids = {int(i) for d in experiment.train.values() for i in d[2]}
    stream_keys = set().union(*(keys(d[0]) for d in experiment.train.values()))
    reference_ids = set(map(int, experiment.reference[2]))
    reference_keys = keys(experiment.reference[0])
    blocked_ids = stream_ids | reference_ids
    blocked_keys = stream_keys | reference_keys
    probes, probe_ids, probe_keys = [], set(), set()
    manifest = {}
    for level in (0, 1, 3):
        full = api.base.legacy.read_domain(a.burst_dir, experiment.artifact, level, 'train_ids')
        feature_keys = api.feature_keys(full[0])
        eligible = np.asarray([int(i) not in blocked_ids and key not in blocked_keys
            for i, key in zip(full[2], feature_keys)])
        pool = api.subset(full, eligible)
        if len(pool[1]) < rows or set(np.unique(pool[1])) != set(range(len(a.classes))):
            raise ValueError(f'TRAIN level {level}: insufficient independent probe rows/classes; '
                'no validation/test or stream-row fallback is permitted.')
        at = api.pilot_indices(pool[1], rows, a.seed+8401+level)
        selected = np.zeros(len(pool[1]), dtype=bool); selected[at] = True
        probe = api.subset(pool, selected)
        # Duplicate rows within one probe are removed without changing the streams.
        unique, seen = [], set()
        for i, key in enumerate(api.feature_keys(probe[0])):
            if key not in seen: unique.append(i); seen.add(key)
        unique_mask = np.zeros(len(probe[1]), dtype=bool); unique_mask[unique] = True
        probe = api.subset(probe, unique_mask)
        if set(np.unique(probe[1])) != set(range(len(a.classes))):
            raise ValueError('Probe deduplication removed a class; use another explicit seed/probe size.')
        ids, feature_set = set(map(int, probe[2])), keys(probe[0])
        probes.append(probe); probe_ids.update(ids); probe_keys.update(feature_set)
        blocked_ids.update(ids); blocked_keys.update(feature_set)
        manifest[str(level)] = dict(ids=list(map(int, probe[2])), rows=len(probe[1]))
    at = api.pilot_indices(experiment.reference[1], retention_rows, a.seed+8402)
    selected = np.zeros(len(experiment.reference[1]), dtype=bool); selected[at] = True
    retention = api.subset(experiment.reference, selected)
    excluded_ids = probe_ids | reference_ids
    excluded_keys = probe_keys | reference_keys
    keep = np.asarray([int(i) not in excluded_ids and key not in excluded_keys
        for i, key in zip(experiment.anchors[2], api.feature_keys(experiment.anchors[0]))])
    anchors = api.subset(experiment.anchors, keep)
    if not len(anchors[1]) or set(np.unique(anchors[1])) != set(range(len(a.classes))):
        raise ValueError('Independent probe reservation leaves insufficient source anchors/classes.')
    for data in experiment.train.values():
        if set(map(int, data[2])).intersection(excluded_ids) or keys(data[0]).intersection(excluded_keys):
            raise ValueError('A TRAIN reward probe overlaps the adaptive stream.')
    return probes, retention, anchors, dict(generalization=manifest,
        retention=dict(ids=list(map(int, retention[2])), rows=len(retention[1])),
        excluded_anchor_rows=len(experiment.anchors[1])-len(anchors[1]),
        sampling='fixed TRAIN-only; IDs/features disjoint across probes and streams; macro CE within each level',
        independence='held out from this experiment adaptive updates; not necessarily from historical pretraining')
