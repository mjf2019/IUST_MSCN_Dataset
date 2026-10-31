"""Original paper reward plus signed same-domain TRAIN generalization gain."""
import math

import torch
import torch.nn.functional as F

from .query_utility_reward import FixedProbeReward


def additive_reward(paper_reward, gain, beta):
    if not all(math.isfinite(float(v)) for v in (paper_reward, gain, beta)) or beta < 0:
        raise ValueError('Paper reward, gain and nonnegative beta must be finite.')
    weighted_gain = float(beta*gain)
    reward = float(paper_reward+weighted_gain)
    if not math.isfinite(reward): raise ValueError('Nonfinite additive query reward.')
    return reward, dict(paper_reward_component=float(paper_reward), probe_gain=float(gain),
        weighted_probe_gain=weighted_gain, beta=float(beta))


class SameDomainGain(FixedProbeReward):
    """Reuse only separation checks; no retention penalty or request cost."""
    def __init__(self, probe, excluded_reference, beta):
        additive_reward(0., 0., beta)
        super().__init__([probe], excluded_reference, forgetting_weight=0., cost=0.)
        self.beta = float(beta)

    @torch.no_grad()
    def measure(self, cnn, context):
        if cnn.training or context.training:
            raise ValueError('Reward observation requires eval mode and fixed running statistics.')
        x, y = self.generalization[0][:2]
        truth = torch.as_tensor(y, dtype=torch.long, device=x.device)
        # Exactly the approved mean CE on the independent same-domain probe.
        value = float(F.cross_entropy(cnn(context(x)), truth, reduction='mean'))
        if not math.isfinite(value): raise ValueError('Nonfinite same-domain TRAIN probe loss.')
        self.calls += 1
        return value

    def finish(self, cnn, context, before, paper_reward):
        after = self.measure(cnn, context)
        reward, parts = additive_reward(paper_reward, before-after, self.beta)
        parts.update(generalization_loss_before=float(before), generalization_loss_after=float(after))
        return reward, parts
