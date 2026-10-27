"""Causal deployment of a TRAIN-calibrated, fixed-budget query ranking."""
import math
import numpy as np


def request_by_rank(score, calibration, used, cap, position, count, draw):
    if count < 1 or not 0 <= position < count or not 0 <= used <= cap <= count:
        raise ValueError('Invalid stream budget state.')
    if not 0 <= draw < 1: raise ValueError('Tie draw must be in [0,1).')
    remaining, rows_left = cap-used, count-position
    if remaining == 0: return False, dict(rank=None, fraction=0., forced_tail=False)
    if not math.isfinite(score): raise ValueError('Nonfinite query ranking score.')
    values = np.asarray(calibration, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A finite TRAIN ranking reference is required.')
    # Randomized ties avoid an all-request/all-skip result for a constant Actor.
    rank = float((np.sum(values < score) + draw*np.sum(values == score))/len(values))
    fraction = min(1., remaining/rows_left)
    force = remaining >= rows_left
    return bool(force or rank >= 1-fraction), dict(rank=rank, fraction=fraction, forced_tail=force)
