"""Compact causal query policy for the shared-gap frozen-round min-max.

This is a two-action contextual-bandit policy, not a sequential DDQN. Target
regression is diagnostic only; the actor directly maximizes the signed score.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def query_features(outputs, embedding, paid_embeddings, paid_labels, used, cap,
                   position, count, last_update):
    logits, distortion = outputs
    if logits.shape[0] != 1 or logits.shape[1] != 2 or count < 1:
        raise ValueError('One row and two prediction paths are required.')
    if not 0 <= used <= cap <= count or not 0 <= position <= count:
        raise ValueError('Invalid causal budget state.')
    probs = logits.softmax(-1)[0]
    classes = probs.shape[-1]
    top = probs.topk(2, dim=-1).values
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1) / math.log(classes)
    mean = probs.mean(0)
    js = (probs * (probs.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(-1).mean()
    agreement = (probs[0].argmax() == probs[1].argmax()).float()
    if paid_embeddings:
        bank = torch.cat(paid_embeddings[-64:]).to(embedding)
        distances = 1-F.cosine_similarity(embedding, bank, dim=-1)
        nearest, average = distances.min().clamp(0, 2)/2, distances.mean().clamp(0, 2)/2
    else:
        nearest = average = probs.new_tensor(1.)
    support = torch.bincount(torch.tensor(paid_labels, device=probs.device, dtype=torch.long),
                             minlength=classes).to(probs)
    coverage = (support > 0).float().mean()
    predicted_support = support[probs.argmax(-1)] / max(used, 1)
    scalars = torch.stack([agreement, js, nearest, average, coverage])
    temporal = probs.new_tensor([(cap-used)/max(cap, 1), position/count,
                                 (position-last_update)/count])
    result = torch.cat([probs.flatten(), top[:, 0], top[:, 0]-top[:, 1], entropy,
                        distortion[0].reshape(-1), scalars, predicted_support, temporal])
    if result.numel() != 2*classes+18 or not torch.isfinite(result).all():
        raise ValueError('Invalid compact query features.')
    return result.unsqueeze(0)


class CompetitiveSelector(nn.Module):
    def __init__(self, spec):
        super().__init__()
        mean, scale = torch.tensor(spec['mean']), torch.tensor(spec['scale'])
        if mean.ndim != 1 or mean.shape != scale.shape or (scale <= 0).any() or not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError('Invalid TRAIN-only feature normalization.')
        self.register_buffer('mean', mean.float())
        self.register_buffer('scale', scale.float())
        self.actor = nn.Sequential(nn.Linear(len(mean), 16), nn.GELU(), nn.Linear(16, 1))
        self.value = nn.Sequential(nn.Linear(len(mean), 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, states):
        normalized = ((states-self.mean)/self.scale).clamp(-10, 10)
        # A bounded policy class prevents arbitrarily large, irreversible logits.
        return 4*torch.tanh(self.actor(normalized).squeeze(-1)/4), self.value(normalized).squeeze(-1)


def competitive_score(row, scales, cost):
    if any(not math.isfinite(float(v)) or v <= 0 for v in scales.values()):
        raise ValueError('Competitive scales must be finite and positive.')
    return (row['gap']/scales['gap'] + row['weighted_gain']/scales['weighted_gain']
            - row['weighted_harm']/scales['weighted_harm'] - cost)


def select_feedback(probability, used, cap, *, training=False, draw=0.):
    if not math.isfinite(probability) or not 0 <= probability <= 1 or not 0 <= used <= cap:
        raise ValueError('Invalid query decision.')
    if training and not 0 <= draw < 1:
        raise ValueError('Invalid policy draw.')
    return used < cap and (draw < probability if training else probability >= .5)


def original_paper_reward(frozen_prediction, context_prediction, truth):
    """Original paper Eq. reward, scored only after predictions are fixed.

    TRAIN may observe full labels. TEST must never call this to decide a query.
    This table is not a continuous loss-gap zero-sum objective.
    """
    frozen_correct = frozen_prediction == truth
    context_correct = context_prediction == truth
    if not frozen_correct:
        return 2. if context_correct else 3.
    return 0. if context_correct else -2.
