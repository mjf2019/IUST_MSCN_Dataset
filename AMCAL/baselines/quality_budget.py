"""Causal feedback controller; does not modify router scores or rewards.

Only paid pre-feedback prediction-only route correctness and past-only transfer
probes are accepted. These shadow predictions never replace scored predictions.
The clean frozen-CNN validation accuracy is an offline reference, not a target
test accuracy. Random monitoring avoids relying exclusively on adaptive queries.
"""
from collections import Counter, deque
from bisect import bisect_right
import math
import numpy as np

VERSION = 1


class QualityBudget:
    def __init__(self, rows, cap, reference_accuracy, seed=42, bootstrap_labels=12,
                 audit_fraction=.1, window=32, return_window=12,
                 min_return_updates=8, tolerance=.02, gap_scale=.15,
                 gain_scale=.001, burst=4, gain_weight=1., retention_weight=1., cost=.0001):
        if rows<1 or not 0<=cap<=rows or not math.isfinite(reference_accuracy) or not 0<=reference_accuracy<=1:
            raise ValueError('Invalid quality-controller rows/cap/reference accuracy.')
        if min(bootstrap_labels,burst)<0 or min(window,return_window,min_return_updates)<1 or min_return_updates>return_window:
            raise ValueError('Invalid quality-controller windows/bootstrap/burst.')
        if any(not math.isfinite(v) for v in (audit_fraction,tolerance,gap_scale,gain_scale)) or not 0<=audit_fraction<=1 or not 0<=tolerance<=1 or gap_scale<=0 or gain_scale<=0:
            raise ValueError('Invalid quality-controller audit/gap/gain settings.')
        if any(not math.isfinite(v) or v<0 for v in (gain_weight,retention_weight,cost)):
            raise ValueError('Invalid quality-controller utility weights.')
        self.rows,self.cap,self.reference=rows,cap,float(reference_accuracy)
        self.tolerance,self.gap_scale,self.gain_scale=tolerance,gap_scale,gain_scale
        self.min_return_updates,self.burst=min_return_updates,burst
        self.gain_weight,self.retention_weight,self.cost=gain_weight,retention_weight,cost
        self.rng=np.random.default_rng(seed+601003)
        count=min(bootstrap_labels,cap)
        span=max(count,min(rows,int(math.ceil(rows*.15))))
        self.bootstrap=set(((np.arange(count)+.5)*span/max(count,1)).astype(int).tolist())
        self.trial_labels=max(count,2*min_return_updates)
        candidates=np.asarray([p for p in range(rows) if p not in self.bootstrap],dtype=int)
        audits=min(cap-count,int(math.ceil(cap*audit_fraction)))
        self.audits=set(np.random.default_rng(seed+601019).choice(candidates,audits,replace=False).tolist())
        self.monitor=self.bootstrap|self.audits
        self.monitor_positions=sorted(self.monitor)
        self.monitor_correct=deque(maxlen=window)
        self.returns=deque(maxlen=return_window)
        self.counts=Counter();self.labels=0;self.pending=None;self.next_position=0
        self.decisions={}

    def signals(self):
        # Weak offline prior, replaced by the latest label-blind paid monitors.
        accuracy=(sum(self.monitor_correct)+2*self.reference)/(len(self.monitor_correct)+2)
        deficit=max(0.,self.reference-accuracy)
        need=float(np.clip((deficit-self.tolerance)/self.gap_scale,0.,1.))
        mean_return=float(np.mean(self.returns)) if self.returns else None
        # Unknown yield gets a finite trial; subsequent harm/plateau suppresses
        # adaptive requests but never cancels pre-reserved random monitoring.
        if len(self.returns)<self.min_return_updates and self.labels<self.trial_labels:
            efficiency=1.
        elif mean_return is None:
            efficiency=0.
        else:
            confidence=min(1.,len(self.returns)/self.min_return_updates)
            efficiency=confidence*float(np.clip((mean_return-self.cost)/self.gain_scale,0.,1.))
        return dict(estimated_accuracy=float(accuracy),deficit=deficit,need=need,
                    mean_net_transfer=mean_return,efficiency=efficiency,
                    adaptive_probability=need*efficiency,
                    monitor_observations=len(self.monitor_correct),return_observations=len(self.returns))

    def decide(self, position, used):
        if position in self.decisions:return self.decisions[position]
        if position!=self.next_position or self.pending is not None or used!=self.labels:
            raise RuntimeError('Quality decisions must be sequential and paid feedback completed.')
        if not 0<=position<self.rows:raise ValueError('Position outside stream.')
        signals=self.signals()
        released=min(self.cap,int(math.floor(self.cap*(position+1)/self.rows))+self.burst)
        reserved=len(self.monitor_positions)-bisect_right(self.monitor_positions,position)
        reason='skip';query=False
        if used<self.cap:
            if position in self.monitor:
                query=True;reason='bootstrap' if position in self.bootstrap else 'audit'
            elif used<released and used<self.cap-reserved and self.rng.random()<signals['adaptive_probability']:
                query=True;reason='adaptive'
        decision=dict(query=query,reason=reason,released=released,reserved_future=reserved,**signals)
        self.decisions[position]=decision;self.next_position+=1
        if query:self.pending=position
        return decision

    def observe(self, position, pre_feedback_correct, gain=None, harm=None):
        if self.pending!=position:raise RuntimeError('Quality feedback requires the current paid request.')
        if pre_feedback_correct not in (True,False,0,1):raise ValueError('Correctness must be binary.')
        if (gain is None)!=(harm is None):raise ValueError('Transfer gain and harm must be paired.')
        if gain is not None and any(not math.isfinite(v) or v<0 for v in (gain,harm)):
            raise ValueError('Transfer measurements must be finite and nonnegative.')
        decision=self.decisions[position]
        if position in self.monitor:self.monitor_correct.append(float(pre_feedback_correct))
        if gain is not None:self.returns.append(self.gain_weight*gain-self.retention_weight*harm)
        self.labels+=1;self.counts[decision['reason']]+=1;self.pending=None

    def statistics(self):
        return dict(version=VERSION,reference_accuracy=self.reference,labels=self.labels,cap=self.cap,
                    bootstrap_labels=self.counts['bootstrap'],audit_labels=self.counts['audit'],
                    adaptive_labels=self.counts['adaptive'],planned_monitors=len(self.monitor),
                    yield_trial_label_limit=self.trial_labels,
                    **self.signals(),
                    protocol='external quality/yield gate; label-blind paid monitors; unchanged routing/update scores; '
                             'strict global cap, gradual release and reserved future monitors')
