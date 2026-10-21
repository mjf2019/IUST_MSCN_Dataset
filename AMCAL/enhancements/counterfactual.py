"""TRAIN-only alternative Context updates on isolated model/Adam copies."""
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import amcal_burst_structured_bank_protocol as legacy
from .memory import transfer
from .models import BankExecutor


def context_step(a, cnn, context, optimizer, inputs, label, frozen_logits, replay):
    rx, ry = (inputs, label) if replay is None else (torch.cat([inputs, replay[0]]), torch.cat([label, replay[1]]))
    optimizer.zero_grad(set_to_none=True)
    transformed = context(rx)
    ce = F.cross_entropy(cnn(transformed), ry, reduction='none')
    loss = legacy.context_objective(ce[0], F.cross_entropy(frozen_logits, label), a.context_alpha)
    if replay is not None: loss = loss + a.config.memory_ce_weight * ce[1:].mean()
    loss = loss + a.config.residual_weight * (transformed-rx).square().mean()
    loss.backward()
    nn.utils.clip_grad_norm_([p for p in context.parameters() if p.requires_grad], 1)
    optimizer.step()


def alternatives(a, cnn, bank, optimizers, inputs, label, outputs, batch, replay,
                 history, used, cap, position, count, successor_outputs=None,
                 successor_input=None, successor_embedding=None, successor_gate=0):
    """Use the same pre-feedback bank, reference and replay for all destinations.

    No real Context, optimizer, history, memory or RNG is advanced. Inputs include
    only the current acquired label, protected labels and optional successor
    features; no successor label is accepted by this interface.
    """
    if not a.config.protected_reference or a.config.routed_utility or a.config.architecture != 'legacy':
        raise ValueError('Counterfactual targets require legacy Contexts and unrouted protected references.')
    devices = [inputs.device.index if inputs.device.index is not None else torch.cuda.current_device()] if inputs.is_cuda else []
    utilities, successors = [], []
    with torch.random.fork_rng(devices=devices):
        for slot in range(3):
            # Copy together: Adam parameters must point to the copied Context.
            context, optimizer = deepcopy((bank[slot], optimizers[slot]))
            before = legacy.core.probe_losses(cnn, context, (batch.x,batch.y)) if batch is not None else None
            context_step(a, cnn, context, optimizer, inputs, label, outputs[0][:,0], replay)
            after = legacy.core.probe_losses(cnn, context, (batch.x,batch.y)) if batch is not None else None
            utility, _, _ = transfer(before, after, batch, a.gain_weight,
                a.retention_weight, a.selection_cost, a.config)
            utilities.append(utility)
            if successor_outputs is not None:
                temporary_bank = list(bank); temporary_bank[slot] = context
                executor = BankExecutor(cnn, temporary_bank, 'sequential')
                changed = executor.refresh_output(inputs, outputs, slot)
                change = (changed[0][:,slot+1].softmax(1)-outputs[0][:,slot+1].softmax(1)).abs().mean()
                temporary_history = deepcopy(history)
                temporary_history.observe_paid(outputs[0].softmax(2), int(label[0]))
                temporary_history.updated(slot+1, position+1, change)
                next_outputs = executor.refresh_output(successor_input, successor_outputs, slot)
                successors.append(legacy.build_state(next_outputs, successor_embedding,
                    temporary_history, used+1, cap, position+1, count, successor_gate).detach())
    return utilities, torch.cat(successors) if successors else None
