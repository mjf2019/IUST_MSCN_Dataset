"""Exact minibatch gradient of the existing max term under a probability quota.

Maximize mean(p*S), subject to mean(p)=quota. S is detached and unchanged.
The common offset is differentiated implicitly; it is not a label bonus.
This is a constrained local surrogate, not a claim of exact multistep min-max.
"""
import torch
from torch import nn


class _QuotaProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, quota):
        if logits.ndim != 1 or len(logits) < 2 or not torch.isfinite(logits).all() or not 0 < quota < 1:
            raise ValueError('Quota fitting requires finite vector logits and quota in (0,1).')
        lo, hi = logits.min()-60., logits.max()+60.
        for _ in range(80):
            mid = (lo+hi)/2
            if torch.sigmoid(logits-mid).mean() > quota: lo = mid
            else: hi = mid
        probabilities = torch.sigmoid(logits-(lo+hi)/2)
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx, gradient):
        (probabilities,) = ctx.saved_tensors
        weights = probabilities*(1-probabilities)
        denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        # Removes only the gradient of the shared offset; preserves ranking.
        centered = gradient-(gradient*weights).sum()/denominator
        return weights*centered, None


def quota_probabilities(logits, quota):
    return _QuotaProbabilities.apply(logits, float(quota))


def quota_max_step(head, optimizer, states, scores, quota):
    logits, _ = head(states.detach())
    targets = scores.detach().reshape_as(logits)
    if not torch.isfinite(targets).all(): raise ValueError('Nonfinite signed TRAIN scores.')
    probability = quota_probabilities(logits, quota)
    objective = (probability*targets).mean()
    optimizer.zero_grad(set_to_none=True)
    (-objective).backward()
    nn.utils.clip_grad_norm_(head.actor.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(objective=float(objective.detach()), mean_probability=float(probability.detach().mean()))
