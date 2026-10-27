"""Offline ranking diagnostics. Oracles use observer labels and cannot be deployed."""
import numpy as np


def average_ranks(values):
    values=np.asarray(values,dtype=np.float64)
    order=np.argsort(values,kind='stable');ranks=np.empty(len(values),dtype=np.float64)
    start=0
    while start<len(values):
        stop=start+1
        while stop<len(values) and values[order[stop]]==values[order[start]]:stop+=1
        ranks[order[start:stop]]=(start+stop-1)/2.
        start=stop
    return ranks


def correlation(first,second,rank=False):
    a,b=np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
    if a.ndim!=1 or b.shape!=a.shape or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Correlations require matching finite vectors with at least two rows.')
    if rank:a,b=average_ranks(a),average_ranks(b)
    a,b=a-a.mean(),b-b.mean();denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a,b)/denominator,-1.,1.)) if denominator>1e-15 else None


def ranking_report(rows,selected_count,seed=42):
    if len(rows)<2 or not 1<=selected_count<=len(rows):
        raise ValueError('At least two candidates and a valid positive top-k count are required.')
    def vector(key):
        values=np.asarray([float(row[key]) for row in rows],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite diagnostic values: '+key)
        return values
    actor,target,gain,harm,cost=[vector(key) for key in
        ('actor_logit_before','ranking_target','weighted_gain','weighted_harm','request_cost')]
    utility=gain-harm-cost
    measures=dict(target=target,gain=gain,harm=harm,net_utility=utility)
    report=dict(candidates=len(rows),top_k=int(selected_count),
        scope='fixed common uniform-feedback trajectory; top-k ranking only, not a new online rollout',
        correlations={},selectors={},components={})
    for name,first,second in [('actor_target',actor,target),('target_utility',target,utility),
            ('actor_utility',actor,utility),('target_gain',target,gain),('actor_gain',actor,gain)]:
        report['correlations'][name]=dict(pearson=correlation(first,second),spearman=correlation(first,second,True))
    for key,values in measures.items():
        report['components'][key]=dict(mean=float(values.mean()),std=float(values.std()),
            positive_fraction=float((values>0).mean()))
    ties=np.random.default_rng(seed).random(len(rows))
    for name,values in [('Actor',actor),('OracleTarget',target),('OracleUtility',utility)]:
        selected=np.lexsort((ties,-values))[:selected_count]
        report['selectors'][name]={key:float(values[selected].mean()) for key,values in measures.items()}
    # Mean over all candidates is the exact expectation of a uniform random top-k subset.
    report['selectors']['RandomExpectation']={key:float(values.mean()) for key,values in measures.items()}
    report['actor_target_regret']=report['selectors']['OracleTarget']['target']-report['selectors']['Actor']['target']
    report['actor_utility_regret']=report['selectors']['OracleUtility']['net_utility']-report['selectors']['Actor']['net_utility']
    return report


def print_ranking_report(split,level,report):
    def show(value):return 'undefined' if value is None else f'{value:+.3f}'
    c=report['correlations'];s=report['selectors']
    print(f"{split} level {level} | candidates {report['candidates']} | top-k {report['top_k']} | "
        f"Spearman Actor~target {show(c['actor_target']['spearman'])} | "
        f"target~utility {show(c['target_utility']['spearman'])} | "
        f"Actor~utility {show(c['actor_utility']['spearman'])}",flush=True)
    print('  Selector          target          gain          harm   net utility',flush=True)
    for name in ('Actor','OracleTarget','OracleUtility','RandomExpectation'):
        print(f"  {name:<17}"+' '.join(f'{s[name][key]:13.6f}' for key in ('target','gain','harm','net_utility')),flush=True)
    print(f"  Actor regret: target {report['actor_target_regret']:.6f} | utility {report['actor_utility_regret']:.6f}",flush=True)


def print_macro_summary(split,summary):
    print(f'\n{split} MACRO SUMMARY (mean across levels 0/1/3)',flush=True)
    print('Actor~target | target~utility | Actor~utility (Spearman)',flush=True)
    print(' | '.join('undefined' if summary['mean_correlations'][key]['spearman'] is None
        else f"{summary['mean_correlations'][key]['spearman']:+.3f}"
        for key in ('actor_target','target_utility','actor_utility')),flush=True)
    for name,values in summary['selectors'].items():
        print(f"{name:<17} | gain {values['gain']:+.6f} | harm {values['harm']:+.6f} | net utility {values['net_utility']:+.6f}",flush=True)

