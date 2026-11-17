"""Percent-scale sequential retention statistics; standard library only."""
import math

LEVELS = [5,7,10,12,15,17,20]
METRICS = ('AC','PR','RC','F1')


def summarize(matrix, online):
    """stage=0 is initialization; stages 1..7 follow the corresponding level."""
    indexed = {}
    for row in matrix:
        key = (row['stage'],row['domain'])
        if key in indexed:
            raise ValueError('Duplicate observer matrix cell.')
        if any(not math.isfinite(float(row[m])) or not 0<=float(row[m])<=100 for m in METRICS):
            raise ValueError('Observer metrics must be finite percentages.')
        indexed[key] = row
    expected = {(stage,domain) for stage in range(8) for domain in [0,*LEVELS]}
    if set(indexed)!=expected:
        raise ValueError('Expected the complete 8 by 8 observer matrix.')
    if [r['level'] for r in online]!=LEVELS:
        raise ValueError('Expected all seven sequential adaptation stages.')
    result = {}
    for metric in METRICS:
        forgetting,bwt = [],[]
        for j,level in enumerate(LEVELS[:-1],1):
            final = float(indexed[7,level][metric])
            peak = max(float(indexed[t,level][metric]) for t in range(j,8))
            forgetting.append(peak-final)
            bwt.append(final-float(indexed[j,level][metric]))
        initial_source = float(indexed[0,0][metric])
        final_source = float(indexed[7,0][metric])
        result[metric] = dict(
            domain_forgetting=sum(forgetting)/6,
            BWT=sum(bwt)/6,
            original_initial=initial_source,original_final=final_source,
            original_forgetting=initial_source-final_source,
            final_target=sum(float(indexed[7,l][metric]) for l in LEVELS)/7,
            mean_online=sum(float(r[metric]) for r in online)/7)
    total = sum(r['samples'] for r in online)
    labels = sum(r['labels'] for r in online)
    result['feedback'] = dict(samples=total,labels=labels,labels_percent=100*labels/total,
        mean_level_labels_percent=sum(r['labels_percent'] for r in online)/7)
    result['definitions'] = dict(
        domain_forgetting='mean peak-to-final loss on first six target domains; peak includes final stage',
        BWT='mean final minus immediate post-adaptation score on first six target domains',
        original_forgetting='original stage0 minus stage7; negative means improvement',
        final_target='mean stage7 score on all seven fixed target observers',
        units='percent for scores; percentage points for differences')
    return result


def support_scale(stage_paid, cumulative_paid):
    if not 0<=stage_paid<=cumulative_paid:
        raise ValueError('Invalid stage/cumulative feedback counts.')
    return max(stage_paid,1)/max(cumulative_paid,1)
