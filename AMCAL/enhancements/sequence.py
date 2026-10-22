"""Persistent TRAIN rollouts and source-disjoint VAL audit selection."""
import numpy as np
from .agent import DeviceReliability
from .stream import episode, observer


def next_sequence_features(a, domains, index, seed):
    """Only unlabelled adaptation features may cross a sequence boundary."""
    if not getattr(a, 'connected_sequence_td', False) or index+1 >= len(a.levels): return None
    x, _, _ = domains[a.levels[index+1]]
    count = len(x)
    if not count: raise ValueError('Empty successor adaptation domain.')
    first = int(np.random.default_rng(seed).permutation(count)[0])
    return x[first:first+1], count


def split_validation(domains, clean, seed, fraction):
    """A source identity is audit-only at every intensity, including clean."""
    common = sorted(set.intersection(*(set(map(int, d[2])) for d in domains.values()),
                                     set(map(int, clean[2]))))
    if len(common) < 10: raise ValueError('Insufficient common VAL IDs for an independent audit.')
    order = np.random.default_rng(seed+8121).permutation(common)
    audit_ids = set(map(int, order[:max(1, int(len(order)*fraction))]))
    streams, audits = {}, {}
    for level, (x, y, ids) in domains.items():
        mask = np.array([int(i) in audit_ids for i in ids])
        streams[level] = x[~mask], np.asarray(y)[~mask], np.asarray(ids)[~mask]
        audits[level] = x[mask], np.asarray(y)[mask]
        if not len(streams[level][1]) or not mask.any(): raise ValueError('Empty VAL adaptation/audit split.')
        if audit_ids & set(map(int, streams[level][2])): raise RuntimeError('VAL audit identity entered adaptation.')
    cx, cy, ci = clean
    mask = np.array([int(i) in audit_ids for i in ci])
    return streams, audits, (cx[mask], np.asarray(cy)[mask]), sorted(audit_ids)


def fit_sequence(a, cnn, bank, initial, agent, domains, anchors, rollout_seed):
    continuation = {}; stages = []
    for index, level in enumerate(a.levels):
        result, _ = episode(a, cnn, bank, initial, agent, domains[level], anchors,
            mode='learned', training=True, paid_supervision=True,
            rollout_seed=rollout_seed, continuation=continuation,
            next_domain_features=next_sequence_features(a, domains, index, rollout_seed))
        stages.append(dict(level=level, labels=result['labels'], cap=result['label_cap'],
            online_accuracy=result['accuracy'], route_updates=result['route_updates'],
            td_updates=result['td_updates']))
        for key in ('paired_target_diagnostics', 'sequence_td'):
            if key in result: stages[-1][key] = result[key]
        if getattr(a, 'counterfactual_utility', False):
            stages[-1].update({key: result[key] for key in ('counterfactual_rows',
                'counterfactual_context_steps', 'counterfactual_updates', 'counterfactual_loss',
                'counterfactual_selected_max_error')})
    return dict(labels=sum(r['labels'] for r in stages), stages=stages)


def validate_sequence(a, cnn, bank, initial, agent, streams, audits, clean, anchors):
    """Only temporary VAL learners adapt; audit labels are never learner inputs."""
    continuation = {}; matrix = []; clean_scores = []; stages = []
    def observe_stage(history, used, cap, position):
        scores = []
        for x, y in audits.values():
            pred, _, _ = observer(cnn, bank, agent, x, y, history, used, cap,
                                   position, a.seed, a.config)
            scores.append(float(np.mean(pred == y)))
        pred, _, _ = observer(cnn, bank, agent, *clean, history, used, cap,
                               position, a.seed, a.config)
        matrix.append(scores); clean_scores.append(float(np.mean(pred == clean[1])))
    observe_stage(DeviceReliability(len(a.classes), streams[a.levels[0]][0].device, len(bank)+1),
                  0, int(a.budget_fraction*len(streams[a.levels[0]][1])), 0)
    for index, level in enumerate(a.levels):
        m, _ = episode(a, cnn, bank, initial, agent, streams[level], anchors,
                       mode='learned', continuation=continuation,
                       next_domain_features=next_sequence_features(a, streams, index, a.seed))
        stages.append(dict(level=level, labels=m['labels'], cap=m['label_cap'], accuracy=m['accuracy']))
        observe_stage(continuation['history'], m['labels'], m['label_cap'], m['samples'])
    scores = np.asarray(matrix); k = len(stages)
    forgetting = float(np.mean([scores[j+1:, j].max()-scores[-1, j] for j in range(k-1)])) if k>1 else 0.
    bwt = float(np.mean([scores[-1, j]-scores[j+1, j] for j in range(k-1)])) if k>1 else 0.
    return dict(accuracy=float(scores[-1].mean()), drop_pp=100*(clean_scores[0]-clean_scores[-1]),
        online_accuracy=float(np.mean([r['accuracy'] for r in stages])), forgetting_pp=100*forgetting,
        bwt_pp=100*bwt, cross_domain_retention_defined=k>1, labels=sum(r['labels'] for r in stages), stages=stages,
        audit_accuracy_by_stage=scores.tolist(), clean_accuracy_by_stage=clean_scores)
