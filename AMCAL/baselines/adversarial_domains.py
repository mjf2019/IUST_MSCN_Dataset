"""Mapped feature-space attack domains for the shared AMCAL evaluation engine.

FGSM/PGD use L-infinity radii. CW/DeepFool use L2 radii. All are measured in
source TRAIN StandardScaler units, without image-style [0,1] clipping.
CW is the projected, radius-constrained feature-space CW-L2 objective; DeepFool
is the multiclass L2 linearization with its final displacement capped at radius.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

VERSION = 'mapped_feature_attacks_v1'
LEVELS = {'FGSM': (.01, .05, .1, .2, .5),
          'PGD': (.01, .05, .1, .2, .5),
          'CW': (.1, .5, 1., 2., 5.),
          'DeepFool': (.1, .5, 1., 2., 5.)}


def strength(attack, level):
    if attack not in LEVELS or level not in range(6):
        raise ValueError('Unknown attack or ordinal level (expected 0..5).')
    return 0. if level == 0 else LEVELS[attack][level - 1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def domain_name(level):
    # File layout compatibility only; no Burst data/algorithm is used here.
    return f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'


def validate_ids(ids, source_rows):
    raw = np.asarray(ids)
    if (raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all()
            or not np.equal(raw, np.floor(raw)).all()):
        raise ValueError('Source IDs must be a nonempty integer vector.')
    result = raw.astype(np.int64)
    if len(np.unique(result)) != len(result) or (result < 0).any() or (result >= source_rows).any():
        raise ValueError('Source IDs must be unique and within the source file.')
    return result


def export_domain(destination, level, source, ids, scaled, artifact):
    ids = validate_ids(ids, len(source))
    scaled = np.asarray(scaled, dtype=np.float64)
    if scaled.shape != (len(ids), len(artifact['feature_names'])) or not np.isfinite(scaled).all():
        raise ValueError('Invalid attack array shape or nonfinite features.')
    raw = artifact['scaler'].inverse_transform(scaled)
    if not np.isfinite(raw).all():
        raise ValueError('Inverse-scaled attack contains nonfinite features.')
    frame = pd.DataFrame(raw, columns=artifact['feature_names'])
    frame[artifact['label_name']] = source.iloc[ids, -1].to_numpy()
    path = Path(destination) / domain_name(level)
    frame.to_csv(path, index=False)
    mapping = path.with_name(path.stem + '_rows.csv')
    pd.DataFrame({'source_row': ids}).to_csv(mapping, index=False)
    return dict(csv_sha256=digest(path), row_map_sha256=digest(mapping), rows=len(ids))


def project_l2(delta, radius):
    norm = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
    scale = (radius / norm).clamp(max=1.)
    return delta * scale.reshape(-1, *([1] * (delta.ndim - 1)))


def _margin(torch, logits, labels):
    real = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(torch.nn.functional.one_hot(labels, logits.shape[1]).bool(), -torch.inf).max(1).values
    return real - other


def perturb(torch, model, inputs, labels, attack, radius, *, pgd_steps=20,
            cw_steps=100, cw_search=3, deepfool_steps=25):
    """No model parameter updates; only the input perturbation is optimized."""
    if radius <= 0:
        return inputs.detach().clone()
    x = inputs.detach()
    if attack == 'FGSM':
        point = x.clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(point), labels)
        gradient, = torch.autograd.grad(loss, point)
        return (x + radius * gradient.sign()).detach()
    if attack == 'PGD':
        point = x + torch.empty_like(x).uniform_(-radius, radius)
        with torch.no_grad():
            best_loss = torch.nn.functional.cross_entropy(model(x), labels, reduction='none')
        best = x.clone()
        for step in range(pgd_steps + 1):
            point = point.detach().requires_grad_(True)
            losses = torch.nn.functional.cross_entropy(model(point), labels, reduction='none')
            with torch.no_grad():
                improved = losses > best_loss
                best[improved], best_loss[improved] = point[improved], losses[improved]
            if step == pgd_steps:
                break
            gradient, = torch.autograd.grad(losses.sum(), point)
            point = x + (point.detach() + (2 * radius / pgd_steps) * gradient.sign() - x).clamp(-radius, radius)
        return best.detach()
    if attack == 'CW':
        with torch.no_grad():
            original_margin = _margin(torch, model(x), labels)
        best, fallback = x.clone(), x.clone()
        best_norm = torch.where(original_margin < 0, torch.zeros_like(original_margin), torch.full_like(original_margin, torch.inf))
        fallback_margin = original_margin.clone()
        lower, upper = torch.zeros_like(original_margin), torch.full_like(original_margin, torch.inf)
        constant = torch.full_like(original_margin, .01)
        for _ in range(cw_search):
            delta = torch.zeros_like(x, requires_grad=True)
            optimizer = torch.optim.Adam([delta], lr=.01)
            successful = torch.zeros_like(original_margin, dtype=torch.bool)
            for step in range(cw_steps + 1):
                candidate = x + delta
                margin = _margin(torch, model(candidate), labels)
                l2 = delta.flatten(1).square().sum(1)
                with torch.no_grad():
                    valid = margin < 0
                    successful |= valid
                    improved = valid & (l2 < best_norm)
                    best[improved], best_norm[improved] = candidate[improved], l2[improved]
                    closer = margin < fallback_margin
                    fallback[closer], fallback_margin[closer] = candidate[closer], margin[closer]
                if step == cw_steps:
                    break
                objective = (l2 + constant * (margin + 1e-6).clamp_min(0)).sum()
                gradient, = torch.autograd.grad(objective, delta)
                optimizer.zero_grad()
                delta.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    delta.copy_(project_l2(delta, radius))
            upper = torch.where(successful, torch.minimum(upper, constant), upper)
            lower = torch.where(successful, lower, torch.maximum(lower, constant))
            constant = torch.where(torch.isfinite(upper), (lower + upper) / 2, constant * 10)
        failed = ~torch.isfinite(best_norm)
        best[failed] = fallback[failed]
        return (x + project_l2(best - x, radius)).detach()
    if attack == 'DeepFool':
        with torch.no_grad():
            original_class = model(x).argmax(1)
            initially_correct = original_class == labels
        result = x.clone()
        for _ in range(deepfool_steps):
            with torch.no_grad():
                active = initially_correct & (model(result).argmax(1) == original_class)
            if not active.any():
                break
            ids = active.nonzero().flatten()
            point = result[ids].detach().requires_grad_(True)
            output = model(point)
            origin = original_class[ids]
            base = output.gather(1, origin[:, None]).squeeze(1)
            base_grad, = torch.autograd.grad(base.sum(), point, retain_graph=True)
            distance = torch.full((len(ids),), torch.inf, device=x.device)
            update = torch.zeros_like(point)
            for cls in range(output.shape[1]):
                gradient, = torch.autograd.grad(output[:, cls].sum(), point, retain_graph=True)
                normal = gradient - base_grad
                norm = normal.flatten(1).norm(dim=1)
                gap = output[:, cls] - base
                candidate_distance = gap.abs() / norm.clamp_min(1e-12)
                candidate_distance = candidate_distance.masked_fill((origin == cls) | (norm < 1e-12), torch.inf)
                closer = candidate_distance < distance
                displacement = ((gap.abs() + 1e-6) / norm.square().clamp_min(1e-12)).reshape(-1, 1, 1) * normal
                update[closer] = displacement[closer]
                distance[closer] = candidate_distance[closer]
            result[ids] = (point.detach() + 1.02 * update.detach())
        return (x + project_l2(result - x, radius)).detach()
    raise ValueError('Unsupported attack.')


def generate(torch, model, scaled, labels, attack, radius, device, seed, **settings):
    if attack not in LEVELS or not np.isfinite(radius) or radius < 0:
        raise ValueError('Invalid attack/radius.')
    if any(int(value) < 1 for value in settings.values()):
        raise ValueError('Attack steps must be positive.')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = []
    # Do not alter the RNG used by subsequent CNN/Context/selector training.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        for start in range(0, len(labels), 64):
            inputs = torch.tensor(scaled[start:start + 64], dtype=torch.float32, device=device).unsqueeze(1)
            truth = torch.tensor(labels[start:start + 64], dtype=torch.long, device=device)
            with torch.enable_grad():
                output = perturb(torch, model, inputs, truth, attack, radius, **settings)
            delta = (output - inputs).flatten(1)
            norm = delta.abs().max(1).values if attack in ('FGSM', 'PGD') else delta.norm(dim=1)
            if not torch.isfinite(output).all() or bool((norm > radius + 1e-5).any()):
                raise RuntimeError('Attack violated its finite perturbation budget.')
            result.append(output.squeeze(1).cpu().numpy())
            print(f'  {attack} radius {radius:g}: {min(start+64,len(labels))}/{len(labels)} rows', flush=True) if start % 1024 == 0 else None
    return np.concatenate(result)


def evaluation_row_ids(available, partition, full_domain=False):
    """Full supplied domain, or the original heldout source-ID intersection."""
    available=np.asarray(available,dtype=np.int64)
    if full_domain:return available.copy()
    lookup=set(available.tolist())
    return np.asarray([i for i in partition if int(i) in lookup],dtype=np.int64)
