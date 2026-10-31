"""Group-disjoint target holdout and observation of a real adaptation Context."""
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import amcal_two_action_paper_gain as base
from .causal_window_query import run_window_episode


def split_targets(domains,seed):
    """One joint group assignment across intensities; first fixed stratified fold."""
    levels=sorted(domains)
    counts={level:len(domains[level][1]) for level in levels}
    if not levels or min(counts.values())<5: raise ValueError('Insufficient target rows for a 20% holdout.')
    parents=[]; id_owner={}; key_owner={}
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j: parents[max(i,j)]=min(i,j)
    labels=[]; entries=[]
    for level in levels:
        data=domains[level]; keys=base.pilot.feature_keys(data[0])
        for local,(identity,key,label) in enumerate(zip(data[2],keys,data[1])):
            i=len(parents); parents.append(i); identity=int(identity)
            if identity in id_owner: union(i,id_owner[identity])
            else: id_owner[identity]=i
            if key in key_owner: union(i,key_owner[key])
            else: key_owner[key]=i
            labels.append(int(label)); entries.append((level,local))
    groups=np.asarray([find(i) for i in range(len(parents))])
    if len(set(groups))<5: raise ValueError('Fewer than five independent target groups; no row-split fallback.')
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed+8801)
    _,held=next(splitter.split(np.zeros(len(labels)),np.asarray(labels),groups))
    held_groups=set(groups[held])
    masks={level:np.zeros(counts[level],dtype=bool) for level in levels}
    for i,(level,local) in enumerate(entries): masks[level][local]=groups[i] in held_groups
    streams={}; audits={}; report={}
    for level in levels:
        mask=masks[level]; streams[level]=base.pilot.subset(domains[level],~mask)
        audits[level]=base.pilot.subset(domains[level],mask)
        if not mask.any() or mask.all(): raise ValueError('Empty target stream/holdout after group assignment.')
        report[str(level)]=dict(total_rows=counts[level],adaptation_rows=int((~mask).sum()),
            holdout_rows=int(mask.sum()),holdout_percent=100*float(mask.mean()),
            adaptation_sha256=base.domain_hash(streams[level]),holdout_sha256=base.domain_hash(audits[level]),
            adaptation_ids=list(map(int,streams[level][2])),holdout_ids=list(map(int,audits[level][2])),
            adaptation_class_counts={str(k):int(v) for k,v in Counter(map(int,streams[level][1])).items()},
            holdout_class_counts={str(k):int(v) for k,v in Counter(map(int,audits[level][1])).items()})
    stream_ids={int(i) for d in streams.values() for i in d[2]}
    hold_ids={int(i) for d in audits.values() for i in d[2]}
    stream_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in streams.values()))
    hold_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in audits.values()))
    if stream_ids & hold_ids or stream_keys & hold_keys:
        raise RuntimeError('Cross-level target holdout leakage.')
    return streams,audits,dict(policy='joint flow-ID/processed-feature groups; first fixed StratifiedGroupKFold fold of five',
        seed=seed+8801,requested_holdout_percent=20.,groups=len(set(groups)),
        actual_fraction='approximately 20 percent; groups never broken to force an exact row count',
        levels=report)


def purge_replay(anchors,audits):
    held_ids={int(i) for d in audits.values() for i in d[2]}
    held_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in audits.values()))
    mask=np.asarray([int(i) not in held_ids and key not in held_keys
        for i,key in zip(anchors[2],base.pilot.feature_keys(anchors[0]))])
    kept=base.pilot.subset(anchors,mask)
    if not len(kept[1]) or set(np.unique(kept[1]))!=set(np.unique(anchors[1])):
        raise ValueError('Removing target holdout from source replay loses a class; no unsafe fallback.')
    return kept,dict(before_rows=len(anchors[1]),after_rows=len(kept[1]),
        removed_rows=int((~mask).sum()),kept_sha256=base.domain_hash(kept),
        guarantee='holdout IDs and exact processed features cannot enter current source replay',
        historical_models='pre-existing CNN/Context weights are not retrospectively refitted')


class ContextObserver:
    def __init__(self,experiment):
        self.experiment=experiment; self.live_context=None

    def __getattr__(self,name): return getattr(self.experiment,name)

    def context(self):
        self.live_context=self.experiment.context()
        return self.live_context


def run_holdout_case(experiment,learner,stream,audit,source_reference,seed,method,window_size):
    if method not in ('LearnedBetaDDQN','UncertaintyWindow'): raise ValueError('Unsupported holdout comparison.')
    stream_ids=set(map(int,stream[2])); audit_ids=set(map(int,audit[2]))
    audit_keys=set(base.pilot.feature_keys(audit[0]))
    if stream_ids & audit_ids or audit_keys & set(base.pilot.feature_keys(stream[0])):
        raise ValueError('Audit overlaps adaptive stream.')
    if audit_ids & set(map(int,experiment.anchors[2])) or audit_keys & set(base.pilot.feature_keys(experiment.anchors[0])):
        raise ValueError('Audit overlaps source replay.')
    observer=ContextObserver(experiment)
    before=base.endpoint(experiment,experiment.template,audit)
    if method=='LearnedBetaDDQN':
        result,trace=base.run_episode(observer,learner,stream,source_reference,seed,training=False,mode='DDQN')
    else:
        result,trace=run_window_episode(observer,learner,stream,source_reference,seed,'UncertaintyWindow',window_size)
    if observer.live_context is None: raise RuntimeError('Adaptation Context was not observed.')
    after=base.endpoint(experiment,observer.live_context,audit)
    # Preserve the old seen-row endpoint only under an explicit diagnostic name.
    result['adaptation_stream_seen_before']=result.pop('target_before')
    result['adaptation_stream_seen_after']=result.pop('target_after')
    result['target_before']=before; result['target_after']=after
    result['target_holdout_rows']=len(audit[1])
    result['target_final_definition']='independent same-level holdout; no audit row used in updates'
    result['target_holdout_gain']={k:after[k]-before[k] for k in before}
    result['labels_percent_of_all_target_rows']=100*result['labels']/(len(stream[1])+len(audit[1]))
    result['target_total_rows']=len(stream[1])+len(audit[1])
    return result,trace

