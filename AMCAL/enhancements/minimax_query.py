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


def query_target_allowed(training, used, cap, position, count, horizon):
    return training and used < cap and position + horizon < count


def maximize_step(head, optimizer, states, scores):
    """Exact gradient of a two-action, full-information frozen-round surrogate."""
    logits, predicted = head(states.detach())
    target = scores.detach().reshape_as(predicted)
    if not torch.isfinite(target).all(): raise ValueError('Nonfinite counterfactual score.')
    actor_loss = -(logits.sigmoid() * target).mean()
    value_loss = F.smooth_l1_loss(predicted, target)
    if not torch.isfinite(actor_loss + value_loss): raise ValueError('Nonfinite query-head loss.')
    optimizer.zero_grad(set_to_none=True)
    (actor_loss + value_loss).backward()
    nn.utils.clip_grad_norm_(head.actor.parameters(), 1., error_if_nonfinite=True)
    nn.utils.clip_grad_norm_(head.value.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()))


def context_min_step(context, optimizer, cnn, current, paid_rows, replay, replay_weight):
    """One batch5 forward/backward, equally weighting all available paid rows.

    Frozen CE is detached. The actor uses the same batch-mean gap; protected
    replay is a separate retention regularizer, disjoint from the probes.
    """
    if context.training or cnn.training:
        raise ValueError('Context updates must keep Context and Frozen in eval mode.')
    if any(p.requires_grad for p in cnn.parameters()):
        raise ValueError('Frozen classifier parameters must not require gradients.')
    if len(current[1]) != 1 or any(len(y) != 1 for _, y in paid_rows):
        raise ValueError('batch5 expects one current row and individual previous paid rows.')
    if len(paid_rows) > 4: raise ValueError('batch5 permits at most four prior paid rows.')
    rows = [current, *paid_rows]
    count = len(rows)
    if replay is not None: rows.append(replay)
    x, y = torch.cat([r[0] for r in rows]), torch.cat([r[1] for r in rows])
    optimizer.zero_grad(set_to_none=True)
    losses = F.cross_entropy(cnn(context(x)), y, reduction='none')
    with torch.no_grad(): frozen_loss = F.cross_entropy(cnn(x[:count]), y[:count])
    gap = losses[:count].mean() - frozen_loss
    retention = losses[count:].mean() if len(losses) > count else losses[0] * 0
    loss = gap + replay_weight * retention
    if not torch.isfinite(loss): raise ValueError('Nonfinite Context loss.')
    loss.backward()
    nn.utils.clip_grad_norm_(context.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(gap.detach()), float(retention.detach())


@torch.no_grad()
def row_losses(cnn, context, data):
    if cnn.training or context.training:
        raise ValueError('Probes require eval mode; running statistics must not change.')
    x, y = data
    return F.cross_entropy(cnn(context(x)), y, reduction='none')


class CounterfactualWorkspace:
    """Reuse a shadow module; clone Adam tensors to avoid storage aliasing."""
    def __init__(self):
        self.context = self.optimizer = None

    def copy_from(self, context, optimizer):
        if self.context is None:
            self.context, self.optimizer = deepcopy((context, optimizer))
        else:
            if self.context is context: raise RuntimeError('Shadow aliases the live Context.')
            self.context.load_state_dict(context.state_dict())
            self.context.train(context.training)
            self.optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        return self.context, self.optimizer

    def recycle(self, old_context, old_optimizer):
        self.context, self.optimizer = old_context, old_optimizer


def counterfactual_step(context, optimizer, cnn, current, paid_rows, replay,
                        future, reference, replay_weight, gain_weight,
                        forgetting_weight, cost, workspace=None):
    """A copied Context/Adam update; live Context, Adam and RNG stay untouched.

    Both probes are held fixed. The skip branch is the unmodified live model;
    neither branch adapts within the future block. No memory is inserted or
    sampled here, so the caller controls exactly which labels are accessible.
    """
    devices = [current[0].device.index] if current[0].is_cuda else []
    with torch.random.fork_rng(devices=devices):
        candidate, candidate_optimizer = (deepcopy((context, optimizer)) if workspace is None
            else workspace.copy_from(context, optimizer))
        if len(future[1]) == 0 or len(reference[1]) == 0:
            raise ValueError('Future and forgetting probes must be nonempty.')
        probe = torch.cat([future[0], reference[0]]), torch.cat([future[1], reference[1]])
        before = row_losses(cnn, context, probe)
        gap, retention = context_min_step(candidate, candidate_optimizer, cnn,
            current, paid_rows, replay, replay_weight)
        after = row_losses(cnn, candidate, probe)
        before_future, before_old = before[:len(future[1])], before[len(future[1]):]
        after_future, after_old = after[:len(future[1])], after[len(future[1]):]
        # Signed generalization gain; positive-only old-domain harm.
        gain = float((before_future - after_future).mean())
        harm = float((after_old - before_old).clamp_min(0).mean())
        score = gap + gain_weight * gain - forgetting_weight * harm - cost
    return candidate, candidate_optimizer, dict(gap=gap, gain=gain, harm=harm,
        score=score, replay_loss=retention)
