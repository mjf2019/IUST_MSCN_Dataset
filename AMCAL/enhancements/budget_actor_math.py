"""Causal deployment of a TRAIN-calibrated, fixed-budget query ranking."""
import math
import numpy as np


def query_window_count(count, cap, fraction=1.):
    if count < 1 or not 0 <= cap <= count or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Invalid query window.')
    window=max(1,int(count*fraction))
    if cap > window: raise ValueError('Label cap exceeds the query window.')
    return window


def fit_signal_scales(rows):
    """Fixed positive TRAIN scales; never fit or refit on VAL/TEST."""
    keys=('gap','weighted_gain','weighted_harm')
    values=np.asarray([[float(row[key]) for key in keys] for row in rows],dtype=np.float64)
    if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Finite measured TRAIN components are required.')
    scales=values.std(axis=0)
    return {key:float(scale) if scale>1e-8 else 1. for key,scale in zip(keys,scales)}


def scaled_competitive_score(row, scales):
    if any(not math.isfinite(v) or v<=0 for v in scales.values()):
        raise ValueError('Signal scales must be finite and positive.')
    score=(float(row['gap'])/scales['gap']+float(row['weighted_gain'])/scales['weighted_gain']
        -float(row['weighted_harm'])/scales['weighted_harm']-float(row['request_cost']))
    if not math.isfinite(score): raise ValueError('Nonfinite scaled TRAIN target.')
    return score


def request_by_rank(score, calibration, used, cap, position, count, draw, window_fraction=1.):
    if count < 1 or not 0 <= position < count or not 0 <= used <= cap <= count:
        raise ValueError('Invalid stream budget state.')
    if not 0 <= draw < 1: raise ValueError('Tie draw must be in [0,1).')
    window=query_window_count(count,cap,window_fraction)
    remaining, rows_left = cap-used, window-position
    if remaining == 0: return False, dict(rank=None, fraction=0., forced_tail=False)
    if rows_left<=0: raise ValueError('Unspent label budget after the query deadline.')
    if not math.isfinite(score): raise ValueError('Nonfinite query ranking score.')
    values = np.asarray(calibration, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('A finite TRAIN ranking reference is required.')
    # Randomized ties avoid an all-request/all-skip result for a constant Actor.
    rank = float((np.sum(values < score) + draw*np.sum(values == score))/len(values))
    fraction = min(1., remaining/rows_left)
    force = remaining >= rows_left
    return bool(force or rank >= 1-fraction), dict(rank=rank, fraction=fraction, forced_tail=force)
