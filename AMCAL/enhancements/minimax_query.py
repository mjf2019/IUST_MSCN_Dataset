"""Experimental query maximizer; no bootstrap and no online query-head updates.

The actor directly ascends E[p(query|s) * detached_score]. A separate value
output regresses the score for diagnostics; regression is not called the max
step. Routing, query estimation and the query actor have disjoint parameters.
"""
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F


class QueryMaximizer(nn.Module):
    def __init__(self, width):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.actor = head()
        self.value = head()
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, states):
        return self.actor(states).squeeze(-1), self.value(states).squeeze(-1)


def maximize_step(head, optimizer, states, scores):
    """Exact gradient of a two-action, full-information frozen-round surrogate."""
    logits, predicted = head(states.detach())
    target = scores.detach().reshape_as(predicted)
    if not torch.isfinite(target).all(): raise ValueError('Nonfinite counterfactual score.')
    actor_loss = -(logits.sigmoid() * target).mean()
    value_loss = F.smooth_l1_loss(predicted, target)
    optimizer.zero_grad(set_to_none=True)
    (actor_loss + value_loss).backward()
    nn.utils.clip_grad_norm_(head.parameters(), 1.)
    optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()))


def context_min_step(context, optimizer, cnn, current, paid_rows, replay, replay_weight):
    """One batch5 forward/backward: current gap plus prior-feedback/source replay.

    Frozen CE is a detached reference. The four prior paid rows and protected
    TRAIN replay regularize retention; they do not enter the independent probes.
    Unlike the legacy nonlinear alpha loss, this explicit loss differentiates
    the same current-row gap used by the query maximizer.
    """
    rows = [current, *paid_rows]
    if replay is not None: rows.append(replay)
    x, y = torch.cat([r[0] for r in rows]), torch.cat([r[1] for r in rows])
    optimizer.zero_grad(set_to_none=True)
    losses = F.cross_entropy(cnn(context(x)), y, reduction='none')
    with torch.no_grad(): frozen_loss = F.cross_entropy(cnn(current[0]), current[1])
    gap = losses[0] - frozen_loss
    # Equal weight per retained training row, including up to four paid labels.
    retention = losses[1:].mean() if len(losses) > 1 else losses[0] * 0
    loss = gap + replay_weight * retention
    loss.backward()
    nn.utils.clip_grad_norm_(context.parameters(), 1.)
    optimizer.step()
    return float(gap.detach()), float(retention.detach())


@torch.no_grad()
def row_losses(cnn, context, data):
    x, y = data
    return F.cross_entropy(cnn(context(x)), y, reduction='none')


def counterfactual_step(context, optimizer, cnn, current, paid_rows, replay,
                        future, reference, replay_weight, gain_weight,
                        forgetting_weight, cost):
    """A copied Context/Adam update; live Context, Adam and RNG stay untouched.

    Both probes are held fixed. The skip branch is the unmodified live model;
    neither branch adapts within the future block. No memory is inserted or
    sampled here, so the caller controls exactly which labels are accessible.
    """
    devices = [current[0].device.index] if current[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        candidate, candidate_optimizer = deepcopy((context, optimizer))
        before_future = row_losses(cnn, context, future)
        before_old = row_losses(cnn, context, reference)
        gap, retention = context_min_step(candidate, candidate_optimizer, cnn,
            current, paid_rows, replay, replay_weight)
        after_future = row_losses(cnn, candidate, future)
        after_old = row_losses(cnn, candidate, reference)
        # Signed generalization gain; positive-only old-domain harm.
        gain = float((before_future - after_future).mean())
        harm = float((after_old - before_old).clamp_min(0).mean())
        score = gap + gain_weight * gain - forgetting_weight * harm - cost
    return candidate, candidate_optimizer, dict(gap=gap, gain=gain, harm=harm,
        score=score, replay_loss=retention)
