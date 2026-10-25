"""Opt-in TRAIN-fitted Actor preprocessing; routing and reward stay unchanged."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .minimax_query import QueryMaximizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_train_spec(samples, seen):
    """Recover natural-prevalence moments from a per-sign TRAIN reservoir."""
    groups = [[r for r in samples if r['score'] > 0], [r for r in samples if r['score'] < 0]]
    counts = [int(seen.get('positive', 0)), int(seen.get('negative', 0))]
    if min(counts) < 1 or any(len(g) < 8 for g in groups):
        raise ValueError('Actor preprocessing requires at least eight cached TRAIN rows of each sign.')
    states = torch.stack([r['state'].detach().cpu().reshape(-1) for g in groups for r in g]).double()
    if not torch.isfinite(states).all(): raise ValueError('Nonfinite TRAIN Actor states.')
    prevalence = counts[0]/sum(counts)
    weights = torch.tensor([prevalence/len(groups[0])]*len(groups[0])+
        [(1-prevalence)/len(groups[1])]*len(groups[1]), dtype=torch.float64)
    mean = (weights[:, None]*states).sum(0)
    std = (weights[:, None]*(states-mean).square()).sum(0).sqrt()
    scale = torch.where(std < 1e-6, torch.ones_like(std), std)
    return dict(mean=mean.tolist(), scale=scale.tolist(), positive_fraction=prevalence,
        dimensions=states.shape[1], cached_rows=len(states), full_nonzero_target_counts=counts,
        near_constant_dimensions=int((std < 1e-6).sum()),
        fitting_scope='TRAIN cache only; natural-prevalence weighted per-sign reservoir moments; nonzero targets')


def prepare_experiment(a):
    root = Path(a.actor_train_cache)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    completion = json.loads((root/'collection_complete.json').read_text(encoding='utf-8'))
    if protocol != completion.get('protocol') or not completion.get('completed'):
        raise ValueError('Actor TRAIN cache provenance is incomplete or inconsistent.')
    if protocol['dataset'] != a.dataset or protocol['seed'] != a.seed or protocol.get('test_used_for_targets') is not False:
        raise ValueError('Actor normalizer requires a TRAIN-only cache for the same dataset and seed.')
    if protocol['source_checkpoint_sha256'] != sha256(a.models/'final_enhancement.pth'):
        raise ValueError('Actor normalizer and training source checkpoints differ.')
    cache = root/'TRAIN'/'actor_cache.pt'
    if sha256(cache) != completion['cache_sha256']:
        raise ValueError('Actor TRAIN cache checksum mismatch.')
    cached = torch.load(cache, map_location='cpu', weights_only=False)
    if cached.get('version') != 1: raise ValueError('Unsupported Actor TRAIN cache version.')
    spec = fit_train_spec(cached['samples'], cached['seen'])
    return dict(version=1, control=a.actor_control, normalizer=spec, cache_sha256=sha256(cache),
        protocol=protocol, reward='original signed S, unchanged', query_threshold=.5,
        routing='unchanged; original raw-state routing, paid-feedback updates',
        balance_scope='Actor loss only; auxiliary control, not the original shared min-max objective'
            if a.actor_control == 'balanced' else 'natural TRAIN weighting; original signed Actor objective')


class StandardizedQueryMaximizer(QueryMaximizer):
    def __init__(self, width, spec):
        super().__init__(width)
        mean, scale = torch.tensor(spec['mean'], dtype=torch.float32), torch.tensor(spec['scale'], dtype=torch.float32)
        if mean.shape != (width,) or scale.shape != (width,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid TRAIN Actor normalizer.')
        self.register_buffer('actor_mean', mean)
        self.register_buffer('actor_scale', scale)

    def forward(self, states):
        # Only Actor inputs change. The diagnostic value head keeps raw states.
        normalized = (states-self.actor_mean)/self.actor_scale
        return self.actor(normalized).squeeze(-1), self.value(states).squeeze(-1)


def regularized_actor_step(head, optimizer, states, scores, coefficient):
    """Original signed objective plus a coercive, TRAIN-only logit penalty.

    Loss = -mean(sigmoid(z)*stopgrad(S)) + coefficient/2 * mean(z**2).
    The penalty gradient coefficient*z restores finite logits even when the
    sigmoid derivative is tiny. This is an explicitly regularized max problem,
    not an algebraically equivalent implementation of the unregularized loss.
    """
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Actor logit penalty must be finite and nonnegative.')
    logits, predicted = head(states.detach())
    target = scores.detach().reshape_as(predicted)
    if not torch.isfinite(target).all(): raise ValueError('Nonfinite counterfactual score.')
    score_loss = -(logits.sigmoid()*target).mean()
    penalty = .5*coefficient*logits.square().mean()
    actor_loss = score_loss+penalty
    value_loss = F.smooth_l1_loss(predicted, target)
    if not torch.isfinite(actor_loss+value_loss): raise ValueError('Nonfinite regularized Actor loss.')
    optimizer.zero_grad(set_to_none=True)
    (actor_loss+value_loss).backward()
    nn.utils.clip_grad_norm_(head.actor.parameters(), 1., error_if_nonfinite=True)
    nn.utils.clip_grad_norm_(head.value.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
        actor_score_loss=float(score_loss.detach()), actor_logit_penalty_loss=float(penalty.detach()),
        actor_logit=float(logits.detach().mean()))


def balanced_actor_step(head, optimizer, states, scores, positive_fraction):
    """Auxiliary inverse-frequency Actor control; value regression remains raw."""
    if not 0 < positive_fraction < 1: raise ValueError('Invalid TRAIN sign prevalence.')
    logits, predicted = head(states.detach())
    targets = scores.detach().reshape_as(predicted)
    if not torch.isfinite(targets).all(): raise ValueError('Nonfinite counterfactual score.')
    weights = torch.where(targets > 0, .5/positive_fraction, .5/(1-positive_fraction))
    actor_loss = -(logits.sigmoid()*targets*weights).mean()
    value_loss = F.smooth_l1_loss(predicted, targets)
    if not torch.isfinite(actor_loss+value_loss): raise ValueError('Nonfinite Actor control loss.')
    optimizer.zero_grad(set_to_none=True)
    (actor_loss+value_loss).backward()
    nn.utils.clip_grad_norm_(head.actor.parameters(), 1., error_if_nonfinite=True)
    nn.utils.clip_grad_norm_(head.value.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()))

