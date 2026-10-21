"""Serializable settings for isolated, reproducible enhancement ablations."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math


@dataclass(frozen=True)
class EnhancementConfig:
    profile: str = 'compact'
    architecture: str = 'compact'
    rank_weight: float = .2
    paired_weight: float = .5
    route_prior_weight: float = .001
    route_lr_online: float = .0003
    version_half_life: float = 8.
    source_capacity: int = 64
    historical_capacity: int = 64
    recent_capacity: int = 128
    reference_size: int = 32
    memory_replay_size: int = 8
    memory_ce_weight: float = .1
    residual_weight: float = .001
    damage_limit: float = .02
    damage_weight: float = 1.
    protected_reference: bool = True
    routed_utility: bool = False
    token_width: int = 32
    transformer_layers: int = 2
    feedforward_width: int = 128
    residual_bound: float = 2.
    bank_executor: str = 'sequential'
    cascade: bool = False
    cascade_confidence: float = .8
    cascade_margin: float = .15
    cascade_audit_fraction: float = .05
    cascade_max_gap: int = 32
    compute_cost: float = .02
    prerouter_weight: float = .2

    @classmethod
    def preset(cls, name):
        if name not in ('accuracy', 'compact', 'shared', 'cascade'):
            raise ValueError('Unknown enhancement profile: ' + name)
        architecture = 'legacy' if name == 'accuracy' else 'shared' if name in ('shared', 'cascade') else 'compact'
        return cls(profile=name, architecture=architecture, cascade=name == 'cascade')

    def validated(self):
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f'{name} must be finite and nonnegative.')
        for name in ('route_lr_online', 'version_half_life', 'token_width', 'transformer_layers',
                     'feedforward_width', 'residual_bound', 'reference_size', 'cascade_max_gap'):
            if getattr(self, name) <= 0:
                raise ValueError(name + ' must be positive.')
        if self.token_width % 4:
            raise ValueError('token_width must be divisible by four attention heads.')
        if self.architecture not in ('legacy', 'compact', 'shared'):
            raise ValueError('Invalid Context architecture.')
        if self.bank_executor not in ('sequential', 'vmap'):
            raise ValueError('bank_executor must be sequential or vmap.')
        if self.source_capacity + self.historical_capacity + self.recent_capacity > 256:
            raise ValueError('Total memory capacity may not exceed the fixed 256-row comparison budget.')
        if self.protected_reference and self.source_capacity < 1:
            raise ValueError('Protected references require source TRAIN anchors.')
        if self.routed_utility and not self.protected_reference:
            raise ValueError('Routed utility requires the protected reference protocol.')
        if any(not 0 <= getattr(self, name) <= 1 for name in
               ('cascade_confidence', 'cascade_margin', 'cascade_audit_fraction')):
            raise ValueError('Cascade probability thresholds must be in [0, 1].')
        return self

    def without(self, names):
        values = {}
        for name in names:
            if name == 'ranking': values['rank_weight'] = 0.
            elif name == 'paired': values['paired_weight'] = 0.
            elif name == 'prior': values['route_prior_weight'] = 0.
            elif name == 'memory':
                values.update(protected_reference=False, memory_ce_weight=0.)
            elif name == 'replay': values.update(memory_replay_size=0, memory_ce_weight=0.)
            elif name == 'damage': values['damage_weight'] = 0.
            else: raise ValueError('Unknown ablation: ' + name)
        return replace(self, **values).validated()

    def as_dict(self): return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]
