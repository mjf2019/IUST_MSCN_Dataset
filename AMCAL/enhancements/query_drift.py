"""Label-blind feature drift, parameterized using adaptive TRAIN features only."""
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class DriftSpec:
    seed: int
    severity: float
    scale: float = .3
    shift: float = 1.
    noise: float = .03

    def validated(self):
        if self.seed < 0 or not all(math.isfinite(v) and v >= 0 for v in
                (self.severity, self.scale, self.shift, self.noise)):
            raise ValueError('Invalid synthetic drift parameters.')
        if self.severity*self.scale >= .9:
            raise ValueError('Synthetic affine gains must remain strictly positive.')
        return self


def fit_train_scale(domains):
    """No labels, VAL rows or TEST rows enter the feature scale."""
    if not domains or any(len(d[0]) < 2 for d in domains):
        raise ValueError('Nonempty adaptive TRAIN domains are required.')
    shapes = {tuple(d[0].shape[1:]) for d in domains}
    if len(shapes) != 1: raise ValueError('TRAIN feature shapes differ.')
    x = torch.cat([d[0].detach().cpu().double() for d in domains])
    scale = x.std(dim=0, unbiased=False)
    if not torch.isfinite(scale).all(): raise ValueError('Nonfinite TRAIN feature scale.')
    return scale.float()


def transform_domain(data, feature_scale, spec):
    """Identical affine map for stream/probe; row-ID seeded independent noise.

    The caller must preserve official partition IDs. Labels are copied verbatim,
    never consulted to choose transformations, severity or affected features.
    This is a synthetic feature stress test, not a packet-valid adversarial attack.
    """
    spec.validated()
    x, y, ids = data
    if tuple(feature_scale.shape) != tuple(x.shape[1:]) or len(x) != len(ids):
        raise ValueError('Drift feature scale/row IDs differ from the domain.')
    if not torch.isfinite(x).all() or not torch.isfinite(feature_scale).all() or (feature_scale < 0).any():
        raise ValueError('Drift inputs must be finite with a nonnegative TRAIN scale.')
    rng = np.random.default_rng(spec.seed)
    gain = 1+spec.severity*rng.uniform(-spec.scale, spec.scale, size=tuple(x.shape[1:]))
    bias = spec.severity*rng.uniform(-spec.shift, spec.shift, size=tuple(x.shape[1:]))
    gain = torch.as_tensor(gain, dtype=x.dtype, device=x.device)
    bias = torch.as_tensor(bias, dtype=x.dtype, device=x.device)
    scale = feature_scale.to(x)
    shifted = x*gain+scale*bias
    if spec.noise:
        samples = [np.random.default_rng(np.random.SeedSequence([spec.seed, int(i) & 0xffffffff,
            (int(i) >> 32) & 0xffffffff])).normal(0., spec.noise, tuple(x.shape[1:])) for i in ids]
        noise = torch.as_tensor(np.stack(samples), dtype=x.dtype, device=x.device)
        shifted = shifted+spec.severity*scale*noise
    if not torch.isfinite(shifted).all(): raise ValueError('Nonfinite synthetic drift output.')
    return shifted.detach(), np.asarray(y).copy(), np.asarray(ids).copy()


def episode_spec(seed, phase, level, epoch=1):
    if phase not in ('TRAIN', 'VAL'): raise ValueError('Synthetic TEST transformations are prohibited.')
    if level not in (0, 1, 3) or epoch < 1 or seed < 0:
        raise ValueError('Invalid synthetic episode identity.')
    if phase == 'TRAIN':
        return DriftSpec(seed+112000+epoch*503+level*17, (.75, 1., 1.25)[(epoch-1)%3]).validated()
    return DriftSpec(seed+952000+level*29, (1., 1.25, 1.5)[(0, 1, 3).index(level)]).validated()


def choose_validation_candidate(candidates, frontloaded, no_query, accuracy_tolerance=.5,
                                forgetting_tolerance=.5, min_adaptation_gain=.5):
    """Require actual validation adaptation value; no forced query floor."""
    if not candidates: raise ValueError('No utility candidates were evaluated.')
    if not all(math.isfinite(v) and v >= 0 for v in
            (accuracy_tolerance, forgetting_tolerance, min_adaptation_gain)):
        raise ValueError('Validation tolerances must be finite and nonnegative.')
    scenario_valid = frontloaded['AC'] >= no_query['AC']+min_adaptation_gain
    eligible = [r for r in candidates if scenario_valid and
        r['AC'] >= max(frontloaded['AC']-accuracy_tolerance, no_query['AC']+min_adaptation_gain)
        and r['positive_forgetting'] <= frontloaded['positive_forgetting']+forgetting_tolerance
        and r['labels_percent'] > 0]
    if eligible:
        selected = min(eligible, key=lambda r:(r['labels_percent'],
            -(r['AC']-r['positive_forgetting']), r['candidate']))
    else:
        selected = max(candidates, key=lambda r:(r['AC']-r['positive_forgetting'], -r['labels_percent']))
    saving = frontloaded['labels_percent']-selected['labels_percent']
    return dict(selected=selected['candidate'], development_scenario_valid=scenario_valid,
        validation_goal_met=bool(eligible) and saving > 1e-8,
        accuracy_retention_constraints_met=bool(eligible), label_saving_pp=saving,
        adaptation_gain_vs_no_query_pp=selected['AC']-no_query['AC'],
        eligible=[r['candidate'] for r in eligible], reference=frontloaded, no_query_reference=no_query,
        min_adaptation_gain_pp=min_adaptation_gain, accuracy_tolerance_pp=accuracy_tolerance,
        forgetting_tolerance_pp=forgetting_tolerance,
        criterion='lowest VAL label use with positive adaptation gain over NoQuery and accuracy/retention '
            'near FrontLoaded; otherwise explicit failure and fallback; no forced feedback')
