"""Opt-in TEST action mask using the existing TRAIN q75; rewards and weights unchanged."""
import math
import torch
from .independent_target_holdout import run_holdout_case

GUIDE_METHODS=['LearnedBetaDDQN','Q75GuidedDDQN','Q75Only']


class Q75Decision:
    def __init__(self,learner,classes,threshold,mode):
        if mode not in ('Q75GuidedDDQN','Q75Only') or not isinstance(classes,int) or classes<2:
            raise ValueError('Invalid q75 guide mode/classes.')
        if not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('Invalid frozen TRAIN q75.')
        self.learner=learner; self.classes=classes; self.threshold=float(threshold); self.mode=mode; self.decisions=[]

    def __getattr__(self,name): return getattr(self.learner,name)

    def choose(self,state,can_query,epsilon,rng):
        if epsilon!=0.: raise ValueError('q75 guidance is evaluation-only; no exploration/training.')
        value=torch.as_tensor(state).detach().cpu().reshape(-1)
        if value.numel()!=2*self.classes+18: raise ValueError('Unexpected compact DDQN state shape.')
        entropy=float(value[2*self.classes+5])
        if not math.isfinite(entropy) or not 0<=entropy<=1.000001: raise ValueError('Invalid normalized entropy state.')
        # q75-only must not consult Q values or DDQN actions.
        action,q=(None,None) if self.mode=='Q75Only' else self.learner.choose(state,can_query,0.,rng)
        eligible=entropy>=self.threshold
        chosen=int(bool(can_query) and eligible and (action==1 or self.mode=='Q75Only'))
        self.decisions.append(dict(guide_entropy_state=entropy,guide_threshold=self.threshold,
            guide_uncertainty_eligible=eligible,guide_budget_eligible=bool(can_query),
            guide_ddqn_action=None if action is None else int(action),guide_action=chosen,guide_mode=self.mode))
        return chosen,q


def run_guided_case(experiment,learner,stream,audit,source_reference,seed,method,window_size):
    if method not in GUIDE_METHODS: raise ValueError('Unsupported q75 comparison.')
    if method=='LearnedBetaDDQN':
        result,trace=run_holdout_case(experiment,learner,stream,audit,source_reference,seed,method,window_size)
        result['mean_feedback_delay_rows']=0.
        for row in trace:
            row.update(feedback_step=row['step'] if row['queried'] else None,
                feedback_delay_rows=0 if row['queried'] else None)
        return result,trace
    guided=Q75Decision(learner,len(experiment.a.classes),experiment.threshold,method)
    result,trace=run_holdout_case(experiment,guided,stream,audit,source_reference,seed,'LearnedBetaDDQN',window_size)
    if len(trace)!=len(guided.decisions): raise RuntimeError('q75 decision/trace length mismatch.')
    for row,decision in zip(trace,guided.decisions):
        if int(row['queried'])!=decision['guide_action']: raise RuntimeError('q75 recorded action mismatch.')
        row.update(decision)
        row.update(feedback_step=row['step'] if row['queried'] else None,
            feedback_delay_rows=0 if row['queried'] else None)
    result['query_rule']=method
    result['q75_threshold']=experiment.threshold
    result['q75_source']='saved source Actor TRAIN 75th percentile; fixed across all TEST levels'
    result['guide_scope']='hard query-eligibility mask; no Q/reward update; no budget-forcing fallback'
    result['guide_entropy']='Context entropy coordinate of the existing DDQN state; TRAIN checkpoint threshold'
    result['Q_learning']='none during TEST; q75-only does not consult Q; guided mode consults frozen Q only'
    result['mean_feedback_delay_rows']=0.
    result['guide_veto_count']=sum(d['guide_budget_eligible'] and d['guide_ddqn_action']==1
        and not d['guide_uncertainty_eligible'] for d in guided.decisions)
    result['guide_high_entropy_skipped_by_ddqn']=sum(d['guide_budget_eligible'] and d['guide_ddqn_action']==0
        and d['guide_uncertainty_eligible'] for d in guided.decisions)
    return result,trace

