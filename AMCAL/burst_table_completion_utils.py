"""Shared percent-scale table summaries; standard library only."""
import math
from pathlib import Path
import amcal_final_multiseed as io

METRICS = ('AC', 'PR', 'RC', 'F1')
LEVELS = [5, 7, 10, 12, 15, 17, 20]
SUMMARY_KEYS = ['AC','PR','RC','F1','labels_percent'] + [
    prefix+metric for prefix in
    ('target_initial_','target_final_','target_gain_','target_forgetting_',
     'original_initial_','original_final_','original_forgetting_') for metric in METRICS]


def table_row(dataset, method, seed, level, online, target_before, target_after,
              source_before, source_after, samples, labels):
    if samples <= 0 or not 0 <= labels <= int(.2*samples):
        raise ValueError('Illegal sample/feedback count.')
    for scores in (online, target_before, target_after, source_before, source_after):
        if any(not math.isfinite(float(scores[k])) or not 0 <= float(scores[k]) <= 100
               for k in METRICS):
            raise ValueError('Expected finite percent-scale AC/PR/RC/F1.')
    row = dict(dataset=dataset, method=method, seed=seed, level=level,
               samples=samples, labels=labels, labels_percent=100*labels/samples,
               **{k:float(online[k]) for k in METRICS})
    for k in METRICS:
        row.update({f'target_initial_{k}':float(target_before[k]),
                    f'target_final_{k}':float(target_after[k]),
                    f'target_gain_{k}':float(target_after[k]-target_before[k]),
                    f'target_forgetting_{k}':float(target_before[k]-target_after[k]),
                    f'original_initial_{k}':float(source_before[k]),
                    f'original_final_{k}':float(source_after[k]),
                    f'original_forgetting_{k}':float(source_before[k]-source_after[k])})
    return row


def aggregate(root, rows, seeds, method):
    root = Path(root)
    expected = {(s,l) for s in seeds for l in LEVELS}
    actual = [(r['seed'],r['level']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Expected each requested seed and all seven levels exactly once.')
    per_seed = []
    for seed in seeds:
        selected = [r for r in rows if r['seed'] == seed]
        row = dict(dataset=rows[0]['dataset'],method=method,seed=seed,levels=len(selected))
        row.update({k:sum(r[k] for r in selected)/len(selected) for k in SUMMARY_KEYS})
        per_seed.append(row)
    overall = dict(dataset=rows[0]['dataset'],method=method,seeds=len(seeds),levels=7)
    for k in SUMMARY_KEYS:
        values = [r[k] for r in per_seed]
        overall[k+'_mean'] = sum(values)/len(values)
        overall[k+'_sd'] = io.moments(values)[1] if len(values)>1 else None
    by_level = []
    for level in LEVELS:
        selected = [r for r in rows if r['level'] == level]
        row = dict(dataset=rows[0]['dataset'],method=method,level=level,seeds=len(seeds))
        for k in SUMMARY_KEYS:
            values = [r[k] for r in selected]
            row[k+'_mean'] = sum(values)/len(values)
            row[k+'_sd'] = io.moments(values)[1] if len(values)>1 else None
        by_level.append(row)
    io.csv_write(root/'all_seed_levels.csv', rows)
    io.csv_write(root/'summary_per_seed.csv', per_seed)
    io.csv_write(root/'summary_by_level.csv', by_level)
    io.csv_write(root/'summary_overall.csv', [overall])
    io.write(root/'completed.json',dict(method=method,seeds=seeds,levels=LEVELS,
        summary_sha256=io.digest(root/'all_seed_levels.csv'),overall=overall))
    return overall
