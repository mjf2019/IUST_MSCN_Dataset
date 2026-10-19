"""Causal features and reward costs for a learned seven-action policy.

This module never chooses, schedules, forces or masks a query. Accuracy is a
recency-weighted estimate from paid pre-feedback predictions, with clipped
inverse-propensity weights; it is not an observed population accuracy.
"""
from collections import deque
import math

import numpy as np

VERSION = 1
GLOBAL_WIDTH = 8
SLOT_WIDTH = 7
ROUTES = 4
STATE_WIDTH = GLOBAL_WIDTH + ROUTES * SLOT_WIDTH


class BudgetState:
    def __init__(self, priors, window=64, need_weight=1., price_step=.005,
                 price_max=.2, propensity_clip=20.):
        priors = np.asarray(priors, dtype=float)
        values = (need_weight, price_step, price_max, propensity_clip)
        if priors.shape != (ROUTES,) or not np.isfinite(priors).all() or np.any((priors < 0) | (priors > 1)):
            raise ValueError('Four finite route accuracy priors in [0,1] required.')
        if window < 1 or any(not math.isfinite(v) or v < 0 for v in values) or propensity_clip < 1:
            raise ValueError('Invalid RL budget settings.')
        self.priors = priors.copy()
        self.reference = float(priors[0])
        self.need_weight, self.price_step, self.price_max = need_weight, price_step, price_max
        self.propensity_clip = propensity_clip
        self.observations = deque(maxlen=window)
        self.price = 0.
        self.gain = np.zeros(ROUTES)
        self.harm = np.zeros(ROUTES)
        self.update_count = np.zeros(ROUTES)
        self.last_update = np.zeros(ROUTES)

    def estimates(self, position, count):
        accuracy = self.priors.copy()
        system, support, freshness = self.reference, 0., 1.
        if self.observations:
            weights = np.array([r[2] * np.exp2(-(position-r[3])/max(32., count*.2))
                                for r in self.observations])
            # Normalize weights before prior shrinkage: a rare query is not
            # mistaken for many independent accuracy observations.
            support = float(weights.sum()**2 / np.square(weights).sum())
            weights = weights / weights.sum() * support
            accuracy = (weights @ np.stack([r[0] for r in self.observations]) + 2*self.priors) / (support+2)
            system = float((weights @ np.array([r[1] for r in self.observations]) + 2*self.reference) / (support+2))
            freshness = (position-self.observations[-1][3])/max(count,1)
        confidence = support/(support+8)
        deficit = np.maximum(0., self.reference-accuracy)/max(self.reference,.1)*confidence
        system_deficit = max(0., self.reference-system)/max(self.reference,.1)*confidence
        return accuracy, system, support, freshness, deficit, system_deficit

    def features(self, used, cap, position, count):
        accuracy, system, support, freshness, deficit, _ = self.estimates(position,count)
        global_x = [self.reference, system, support/(support+8), freshness,
                    self.price/max(self.price_max,1e-8),
                    used/max(cap,1)-position/max(count,1),
                    min(cap-used,count-position)/max(count-position,1),cap/max(count,1)]
        slots = np.column_stack([accuracy, deficit, np.full(ROUTES,support/(support+8)),
                                 self.gain,self.harm,self.update_count/(self.update_count+4),
                                 (position-self.last_update)/max(count,1)])
        return np.concatenate([global_x,slots.flatten()]).astype(np.float32)

    def observe_paid(self, correct, prediction_route, propensity, position):
        correct = np.asarray(correct,dtype=float)
        if correct.shape != (ROUTES,) or not np.isfinite(correct).all() or not np.isin(correct,[0,1]).all():
            raise ValueError('Paid correctness must contain four binary outcomes.')
        if not 0 < propensity <= 1 or not 0 <= prediction_route < ROUTES:
            raise ValueError('Positive query propensity and legal prediction route required.')
        weight = min(self.propensity_clip,1/propensity)
        self.observations.append((correct.copy(),float(correct[prediction_route]),weight,position))
        return weight

    def query_adjustment(self, slot, gain, harm, used, cap, position, count,
                         gain_weight=1., retention_weight=1., selection_cost=.0001):
        if slot not in range(3) or any(not math.isfinite(v) or v < 0 for v in (gain,harm,gain_weight,retention_weight,selection_cost)):
            raise ValueError('Invalid destination transfer reward.')
        *_, deficit, system_deficit = self.estimates(position,count)
        need = max(float(deficit[slot+1]),system_deficit)
        # A deficit cannot create reward without measured positive transfer.
        benefit = gain_weight*(1+self.need_weight*need)*gain
        damage = retention_weight*harm
        return dict(utility=benefit-damage-selection_cost-self.price,
                    benefit=benefit,damage=damage,price=self.price,need=need)

    def observe_transfer(self, slot, gain, harm, position):
        r=slot+1
        self.gain[r]=.8*self.gain[r]+.2*gain
        self.harm[r]=.8*self.harm[r]+.2*harm
        self.update_count[r]+=1
        self.last_update[r]=position

    def advance(self, queried, used_before, cap, position, count):
        # A soft resource price, not a minimum-spending constraint. The base
        # selection cost remains even when this multiplier reaches zero.
        rate=min(max(cap-used_before,0),count-position)/max(count-position,1)
        self.price=float(np.clip(self.price+self.price_step*(int(queried)-rate)/
                                 math.sqrt(1+position/32),0,self.price_max))

    def statistics(self, used, cap, position, count):
        accuracy,system,support,freshness,_,_=self.estimates(position,count)
        return dict(version=VERSION,reference_accuracy=self.reference,
                    estimated_route_accuracies=accuracy.tolist(),estimated_system_accuracy=system,
                    effective_support=support,last_feedback_age_fraction=freshness,
                    resource_price=self.price,labels=used,cap=cap,
                    protocol='learned seven-action policy; no external query schedule; '
                             'clipped inverse-propensity paid pre-feedback estimates; '
                             'source TRAIN priors; soft budget price, hard upper cap')
