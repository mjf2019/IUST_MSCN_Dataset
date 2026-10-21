"""Bounded residual Contexts and inference-only bank execution.

Shared trunks are pretrained offline and frozen online. Every slot retains its
own adapter parameters, gradients and Adam state. No final input LayerNorm is
used: the residual identity path preserves the classifier's input coordinates.
"""
from copy import deepcopy
import torch
from torch import nn


class FeatureTrunk(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        width = config.token_width
        self.projection = nn.Linear(1, width)
        self.identity = nn.Parameter(torch.empty(1, features, width))
        nn.init.normal_(self.identity, std=.02)
        layer = nn.TransformerEncoderLayer(width, 4, config.feedforward_width,
                                           dropout=.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.transformer_layers,
                                                 enable_nested_tensor=False)

    def forward(self, x):
        return self.transformer(self.projection(x.squeeze(1).unsqueeze(-1)) + self.identity)


class ResidualAdapter(nn.Module):
    def __init__(self, features, config):
        super().__init__()
        width = config.token_width
        self.depthwise = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.output = nn.Conv1d(width, 1, 1)
        self.gate = nn.Parameter(torch.full((1, 1, features), -2.))
        self.bound = config.residual_bound
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, features):
        delta = self.output(torch.nn.functional.gelu(self.depthwise(features.transpose(1, 2))))
        return x + self.bound * self.gate.sigmoid() * delta.tanh()


class ResidualContext(nn.Module):
    def __init__(self, features, config, trunk=None):
        super().__init__()
        self.trunk = FeatureTrunk(features, config) if trunk is None else trunk
        self.adapter = ResidualAdapter(features, config)

    def forward(self, x): return self.adapter(x, self.trunk(x))


def residual_bank(features, config, initial=None, device='cpu'):
    shared = FeatureTrunk(features, config).to(device) if config.architecture == 'shared' else None
    bank = [ResidualContext(features, config, shared).to(device) for _ in range(3)]
    if initial is None: initial = deepcopy(bank[0].state_dict())
    for context in bank:
        context.load_state_dict(initial)
        context.eval()
    if shared is not None:
        shared.requires_grad_(False)
    return bank


class BankExecutor:
    """Current weights only; refresh a stacked slot immediately after its update."""
    def __init__(self, cnn, bank, mode='sequential'):
        self.cnn, self.bank, self.mode = cnn, bank, mode
        self.shared = len(bank) > 1 and hasattr(bank[0], 'trunk') and all(
            context.trunk is bank[0].trunk for context in bank)
        self.context_forwards = 0
        self.trunk_forwards = 0
        if mode == 'vmap' and not self.shared:
            from torch.func import stack_module_state
            self.params, self.buffers = stack_module_state(bank)
            self.template = deepcopy(bank[0]).to('meta')

    def refresh(self, slot):
        if self.mode == 'vmap' and not self.shared:
            with torch.no_grad():
                for name, value in self.bank[slot].named_parameters(): self.params[name][slot].copy_(value)
                for name, value in self.bank[slot].named_buffers(): self.buffers[name][slot].copy_(value)

    def adapted(self, x, slots):
        if not slots: return []
        self.context_forwards += len(slots) * len(x)
        if self.shared:
            self.trunk_forwards += len(x)
            features = self.bank[0].trunk(x)
            return [self.bank[slot].adapter(x, features) for slot in slots]
        self.trunk_forwards += len(slots) * len(x)
        if self.mode == 'vmap' and len(slots) == len(self.bank):
            from torch.func import functional_call
            out = torch.vmap(lambda p, b: functional_call(self.template, (p, b), (x,)),
                             in_dims=(0, 0))(self.params, self.buffers)
            return list(out.unbind(0))
        return [self.bank[slot](x) for slot in slots]

    def outputs(self, x, frozen_logits, slots=(0, 1, 2)):
        with torch.no_grad():
            adapted = self.adapted(x, slots)
            zs = self.cnn(torch.cat(adapted)).reshape(len(slots), len(x), -1).transpose(0, 1) if adapted else None
            # NaN denotes an unexecuted route; it is never a selector input or a target.
            logits = frozen_logits.new_full((len(x), 4, frozen_logits.shape[-1]), float('nan'))
            logits[:, 0] = frozen_logits
            distortion = x.new_full((len(x), 4), float('nan')); distortion[:, 0] = 0
            denominator = x.flatten(1).norm(dim=1).clamp_min(1e-6)
            for column, (slot, value) in enumerate(zip(slots, adapted)):
                logits[:, slot + 1] = zs[:, column]
                d = (value - x).flatten(1).norm(dim=1) / denominator
                distortion[:, slot + 1] = d / (1 + d)
        return logits, distortion

    def refresh_output(self, x, outputs, slot):
        self.refresh(slot)
        frozen = outputs[0][:, 0]
        changed = self.outputs(x, frozen, (slot,))
        z, d = (value.clone() for value in outputs)
        z[:, slot + 1] = changed[0][:, slot + 1]
        d[:, slot + 1] = changed[1][:, slot + 1]
        return z, d


class PreRouter(nn.Module):
    def __init__(self, embedding_width, classes):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(embedding_width + classes, 64), nn.ReLU(), nn.Linear(64, 4))
        self.compute_value = nn.Sequential(nn.Linear(embedding_width + classes, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.compute_value[-1].weight); nn.init.zeros_(self.compute_value[-1].bias)
        self.register_buffer('source_mean', torch.zeros(embedding_width))
        self.register_buffer('source_scale', torch.ones(embedding_width))
        self.register_buffer('novelty_limit', torch.tensor(float('inf')))

    def forward(self, embedding, frozen):
        return self.network(torch.cat([embedding, frozen.softmax(-1)], -1))

    def fallback_value(self, embedding, frozen):
        return self.compute_value(torch.cat([embedding, frozen.softmax(-1)], -1)).squeeze(-1)

    @torch.no_grad()
    def calibrate(self, train_embedding, validation_embedding):
        self.source_mean.copy_(train_embedding.mean(0))
        self.source_scale.copy_(train_embedding.std(0, unbiased=False).clamp_min(.1))
        scores = self.novelty(validation_embedding)
        self.novelty_limit.copy_(torch.quantile(scores, .99).clamp_min(1.))

    def novelty(self, embedding):
        return ((embedding - self.source_mean) / self.source_scale).square().mean(-1)
