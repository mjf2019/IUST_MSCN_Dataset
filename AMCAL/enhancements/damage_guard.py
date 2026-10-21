"""Online Context-step acceptance on existing protected reference labels."""
from copy import deepcopy
import numpy as np
import torch
import amcal_burst_structured_bank_protocol as legacy
from .counterfactual import context_gradients
from .memory import macro

SCALES = (1., .5, .25)


def guarded_context_step(a, cnn, context, optimizer, inputs, label, frozen_logits,
                         replay, batch, before):
    """Commit at most one Adam step; rejected trials never touch live state.

    Compute the clipped gradient once on an isolated Context/Adam copy. Every
    trial starts from the SAME parameters, moments and step counter. The trial
    LR is temporary; accepted moments advance once and the base LR is retained.
    Only the source/historical part of the already sampled reference is guarded.
    """
    limit = float(a.config.damage_limit)
    info = dict(accepted=False, scale=0., attempts=0, trial_harms=[], accepted_harm=0.,
                protected_rows=0, limit=limit, reason='no_protected_reference')
    if batch is None or before is None: return info, before
    protected = np.asarray(batch.segments) != 2
    info['protected_rows'] = int(protected.sum())
    if not protected.any(): return info, before
    before = np.asarray(before)
    if not np.isfinite(before).all(): raise ValueError('Nonfinite pre-update reference losses.')
    if context.training or cnn.training: raise ValueError('Damage guard requires eval-mode Context/CNN.')
    labels = batch.y.detach().cpu().numpy()[protected]
    devices = [inputs.device.index if inputs.device.index is not None else torch.cuda.current_device()] if inputs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        trial, trial_optimizer = deepcopy((context, optimizer))
        base_weights = deepcopy(trial.state_dict())
        base_optimizer = deepcopy(trial_optimizer.state_dict())
        base_rates = [group['lr'] for group in trial_optimizer.param_groups]
        context_gradients(a, cnn, trial, trial_optimizer, inputs, label, frozen_logits, replay)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trial.parameters()):
            info['reason'] = 'nonfinite_gradient'
            return info, before.copy()
        for scale in SCALES:
            trial.load_state_dict(base_weights)
            trial_optimizer.load_state_dict(deepcopy(base_optimizer))
            for group, rate in zip(trial_optimizer.param_groups, base_rates): group['lr'] = rate*scale
            trial_optimizer.step()
            after = legacy.core.probe_losses(cnn, trial, (batch.x, batch.y))
            harm = None
            if np.isfinite(after).all():
                delta = (before[protected]-after[protected])/(before[protected]+after[protected]+1e-8)
                harm = macro(np.maximum(-delta, 0), labels)
            info['attempts'] += 1; info['trial_harms'].append(harm)
            if harm is not None and harm <= limit:
                # Restore the configured LR before exporting the one accepted step.
                for group, rate in zip(trial_optimizer.param_groups, base_rates): group['lr'] = rate
                context.load_state_dict(trial.state_dict())
                optimizer.load_state_dict(deepcopy(trial_optimizer.state_dict()))
                for live, copied in zip(context.parameters(), trial.parameters()):
                    live.grad = None if copied.grad is None else copied.grad.detach().clone()
                info.update(accepted=True, scale=scale, accepted_harm=harm, reason='accepted')
                return info, after
    info['reason'] = 'harm_limit'
    return info, before.copy()
