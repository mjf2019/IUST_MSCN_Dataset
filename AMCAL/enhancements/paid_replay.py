"""Evaluation-only paid-feedback replay controls for the selected PairOnly model."""
import torch
from torch import nn
import torch.nn.functional as F
import amcal_burst_structured_bank_protocol as legacy
from .counterfactual import context_step


def paid_context_step(a, cnn, context, optimizer, inputs, label, frozen_logits, replay, paid_rows):
    rows = [(inputs, label), *paid_rows]
    if a.paid_replay_mode == 'sequential':
        for i, (x, y) in enumerate(rows):
            with torch.no_grad():
                frozen = frozen_logits if i == 0 else cnn(x)
            context_step(a, cnn, context, optimizer, x, y, frozen, replay)
        return
    if a.paid_replay_mode != 'batch': raise ValueError('Unknown paid replay mode.')
    # With no old feedback, preserve the original single-row update exactly.
    if len(rows) == 1:
        context_step(a, cnn, context, optimizer, inputs, label, frozen_logits, replay)
        return
    x = torch.cat([r[0] for r in rows]); y = torch.cat([r[1] for r in rows])
    with torch.no_grad(): frozen_ce = F.cross_entropy(cnn(x), y)
    rx, ry = (x, y) if replay is None else (torch.cat([x, replay[0]]), torch.cat([y, replay[1]]))
    optimizer.zero_grad(set_to_none=True)
    transformed = context(rx)
    ce = F.cross_entropy(cnn(transformed), ry, reduction='none')
    loss = legacy.context_objective(ce[:len(rows)].mean(), frozen_ce, a.context_alpha)
    if replay is not None: loss = loss + a.config.memory_ce_weight * ce[len(rows):].mean()
    loss = loss + a.config.residual_weight * (transformed-rx).square().mean()
    loss.backward()
    nn.utils.clip_grad_norm_([p for p in context.parameters() if p.requires_grad], 1)
    optimizer.step()
