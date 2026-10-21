"""Fixed TRAIN-only scaling of relative update utility; raw critic units remain unchanged."""
import numpy as np


def fit_scale(updates, probabilities, floor=1e-4):
    if not np.isfinite(floor) or floor <= 0: raise ValueError('Calibration floor must be finite and positive.')
    updates = np.asarray(updates, dtype=np.float64)
    rewards = 2*np.asarray(probabilities, dtype=np.float64)-1
    if updates.ndim != 2 or updates.shape[1] != 3 or rewards.shape != updates.shape or not len(updates):
        raise ValueError('Expected matching nonempty N-by-3 TRAIN arrays.')
    if not np.isfinite(updates).all() or not np.isfinite(rewards).all():
        raise ValueError('Nonfinite TRAIN calibration scores.')
    u = np.sqrt(np.mean((updates-updates.mean(1, keepdims=True))**2, axis=1))
    r = np.sqrt(np.mean((rewards-rewards.mean(1, keepdims=True))**2, axis=1))
    us, rs = float(np.quantile(u, .75)), float(np.quantile(r, .75))
    return dict(version=1, scope='TRAIN', method='q75_centered_rms_v1', rows=len(updates),
        utility_spread=us, correctness_spread=rs, floor=floor,
        scale=max(rs, floor)/max(us, floor))


def relative_utility(updates, scale, weight=1., clip=1.):
    centered = (updates-updates.mean(1, keepdim=True)) * (scale*weight)
    # Only shrink outliers; never stretch a tiny difference on a particular row.
    shrink = (centered.abs().amax(1, keepdim=True)/clip).clamp_min(1.)
    return updates.mean(1, keepdim=True) + centered/shrink


def validate_calibration(value, checkpoint_sha256, training_levels):
    if (value.get('version') != 1 or value.get('scope') != 'TRAIN'
            or value.get('method') != 'q75_centered_rms_v1'
            or value.get('checkpoint_sha256') != checkpoint_sha256
            or not value.get('levels') or not set(value['levels']).issubset(training_levels)
            or value.get('rows', 0) < 1 or not np.isfinite(value.get('scale', np.nan))
            or value['scale'] <= 0):
        raise ValueError('Invalid or incompatible TRAIN utility calibration.')
    return value
