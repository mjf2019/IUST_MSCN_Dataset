"""Full RAL: reuse the fixed seed42 model, complete evaluation seeds and endpoints.

No offline RAL/CNN training is performed. Online source+feedback refits remain
the existing full-RAL algorithm. Existing complete paired metrics are reused.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import amcal_final_multiseed as io
from burst_table_completion_utils import METRICS, LEVELS, table_row, aggregate

KEYS = dict(AC='accuracy', PR='macro_precision', RC='macro_recall', F1='macro_f1')


def percent(scores):
    return {k:100*float(scores[v]) for k,v in KEYS.items()}


def reusable(root, seed, level, n, provenance, target_hash):
    for path in sorted((root/'Results').glob(f'ral_per{level}_seed{seed}_*_metrics.json')):
        m = io.read(path)
        e = m.get('execution_config', {})
        trace = path.with_name(path.name.replace('_metrics.json', '_trace.csv'))
        if (not trace.exists() or e.get('seed') != seed or e.get('budget') != .2 or
                e.get('ral_budget_mode') != 'prefix' or e.get('ral_online_update_window') or
                e.get('test_scaler') or e.get('unseen_rows') or
                m.get('samples') != n or m.get('label_cap') != int(.2*n) or
                m.get('target_sha256') != target_hash or
                m.get('training_provenance', {}).get('partition_sha256') != provenance['partition_sha256'] or
                m.get('training_provenance', {}).get('source_preprocessing_sha256') != provenance['source_preprocessing_sha256'] or
                m.get('training_provenance', {}).get('train_levels') != [0,1,3]):
            continue
        try:
            pairs = [m['same_level']['unqueried']['metric_pairs'], m['reference']['metric_pairs']]
            if any(not p['initial']['samples'] or any(
                    p[phase].get(k) is None for phase in ('initial','final') for k in KEYS.values())
                   for p in pairs):
                continue
        except KeyError:
            continue
        return path, trace, m
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT','PS'], required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,48,54,62,72])
    p.add_argument('--model-run', type=Path)
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--progress-every', type=int, default=100)
    args = p.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds)<0:
        p.error('Use distinct nonnegative evaluation seeds.')
    import joblib
    import numpy as np
    import torch
    import amcal_burst_baselines as b
    from baselines.endpoint_metrics import install_baseline, classification

    runs = io.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'
    default = 'ral_paper_rf10_prefix_levels1_3_seed42' if args.dataset=='IAT' else 'baselines_levels1_3_rf10_seed42'
    model_run = (args.model_run or runs/default).resolve()
    checkpoint = model_run/'Models'/'ral.joblib'
    if not checkpoint.exists():
        raise FileNotFoundError('Restore the existing seed42 RAL checkpoint; no automatic retraining: '+str(checkpoint))
    checkpoint_hash = io.digest(checkpoint)
    saved = joblib.load(checkpoint)
    provenance = saved['provenance']
    if (saved.get('ral_protocol_version',0)<2 or provenance['train_levels'] != [0,1,3] or
            saved.get('ral_trees') != 10 or provenance['config']['seed'] != 42):
        raise ValueError('Expected existing seed42 full RAL RF10 trained on 0/1/3.')
    preprocessor = model_run/'Models'/'protocol_preprocessing.joblib'
    artifact = joblib.load(preprocessor)
    if io.digest(preprocessor) != provenance['source_preprocessing_sha256']:
        raise ValueError('RAL preprocessing changed.')
    burst = io.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{args.dataset}'
    for level, sha in provenance['training_domains_sha256'].items():
        if io.digest(burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != sha:
            raise ValueError('RAL TRAIN provenance changed.')
    b.core.device = torch.device('cpu')
    classes = list(range(len(artifact['label_encoder'].classes_)))
    install_baseline(b, classes)
    reference = b.shared.read_domain(burst, artifact, 0, 'val_ids')
    output = runs/'ral_full_paired_endpoints_v1'
    recipe = dict(version=1,dataset=args.dataset,source_checkpoint=str(checkpoint),
        source_sha256=checkpoint_hash,preprocessing_sha256=io.digest(preprocessor),
        training_levels=[0,1,3],budget_fraction=.2,budget_mode='prefix',update_window=0,
        source_reference='full clean VAL, identical before/after',
        target_reference='retrospective unqueried target rows, not a prospective holdout',
        evaluation='pre-feedback online; fixed seed42 source model reloaded per evaluation seed and level',
        code_sha256=io.digest(Path(__file__)),
        helper_sha256=io.digest(io.ROOT/'burst_table_completion_utils.py'),
        baseline_sha256=io.digest(io.ROOT/'amcal_burst_baselines.py'))
    pf = output/'protocol.json'
    if pf.exists() and io.read(pf) != recipe:
        raise ValueError('RAL endpoint recipe changed; existing results preserved.')
    io.write(pf, recipe)
    rows = []
    for seed in args.seeds:
        config = dict(provenance['config'])
        config.update(seed=seed,budget_fraction=.2,ral_update_window=0,ral_budget_mode='prefix',
                      progress_every=args.progress_every,jobs=args.jobs,device='cpu')
        a = SimpleNamespace(**config)
        b.core.set_seed(seed)
        for level in LEVELS:
            domain = b.shared.read_domain(burst, artifact, level, 'test_ids')
            n = len(domain[1])
            if n != (1073 if args.dataset=='IAT' else 886):
                raise ValueError('Unexpected full RAL TEST size.')
            location = output/f'seed{seed}'/f'level_{level}'
            mf, tf, cf = [location/name for name in ('metrics.json','online_trace.csv','completed.json')]
            file_hash = io.digest(burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
            case_recipe = dict(protocol_sha256=io.digest(pf),seed=seed,level=level,target_sha256=file_hash)
            if cf.exists():
                done = io.read(cf)
                if done['case'] != case_recipe or any(io.digest(location/name)!=sha for name,sha in done['sha256'].items()):
                    raise ValueError('Completed RAL endpoint case changed.')
                m = io.read(mf)
            else:
                reuse = reusable(model_run, seed, level, n, provenance, file_hash)
                files = [mf, tf]
                if reuse:
                    path, old_trace, m = reuse
                    with old_trace.open(encoding='utf-8-sig',newline='') as handle:
                        trace = list(csv.DictReader(handle))
                    m = dict(m,reused_from=str(path),reused_sha256=io.digest(path),
                             endpoint_predictions_available=False)
                    print(f'REUSE RAL DATASET={args.dataset} SEED={seed} LEVEL={level}: saved paired metrics',flush=True)
                else:
                    captured = []
                    summary = b.summary_pair
                    def capture(before, after, y, mask):
                        captured.append(tuple(np.asarray(v).copy() for v in (before,after,y,mask)))
                        return summary(before,after,y,mask)
                    b.summary_pair = capture
                    print(f'RUN RAL FULL DATASET={args.dataset} EVAL_SEED={seed} LEVEL={level} '
                          'BUDGET=20% | fixed source checkpoint; online full refits; capture endpoints',flush=True)
                    try:
                        m, trace = b.evaluate_level(a,'ral',saved,domain,reference[:2],observer=True)
                    finally:
                        b.summary_pair = summary
                    if len(captured) != 3:
                        raise RuntimeError('Expected full-target, unqueried-target and original observer pairs.')
                    ordered_ids = np.asarray(domain[2])[np.random.default_rng(seed).permutation(n)]
                    observers = []
                    for scope, ids, pair in [('same_target_unqueried',ordered_ids,captured[1]),
                                             ('original0_VAL',reference[2],captured[2])]:
                        before, after, truth, mask = pair
                        for identity, y, first, last in zip(np.asarray(ids)[mask],truth[mask],before[mask],after[mask]):
                            observers.append(dict(domain=scope,source_row=int(identity),truth=int(y),
                                                  initial_prediction=int(first),final_prediction=int(last)))
                    op = location/'observer_predictions.csv'
                    io.csv_write(op, observers)
                    files.append(op)
                    m.update(endpoint_predictions_available=True,
                             training_provenance=provenance,target_sha256=file_hash,
                             execution_config=dict(seed=seed,budget=.2,ral_budget_mode='prefix',ral_protocol_version=2))
                io.write(mf, m)
                io.csv_write(tf, trace)
                io.write(cf,dict(case=case_recipe,sha256={f.name:io.digest(f) for f in files}))
            with tf.open(encoding='utf-8-sig',newline='') as handle:
                trace = list(csv.DictReader(handle))
            if len(trace)!=n or m['samples']!=n or m['label_cap']!=int(.2*n):
                raise ValueError('RAL full-stream/budget mismatch.')
            online = percent(classification(np.asarray([int(r['truth']) for r in trace]),
                np.asarray([int(r['prediction']) for r in trace]), classes))
            target = m['same_level']['unqueried']['metric_pairs']
            source = m['reference']['metric_pairs']
            row = table_row(args.dataset,'RAL-full',seed,level,online,
                percent(target['initial']),percent(target['final']),
                percent(source['initial']),percent(source['final']),n,m['labels'])
            row.update(target_observer_rows=target['initial']['samples'],
                       original_observer_rows=source['initial']['samples'])
            rows.append(row)
            print(f'RAL RESULT DATASET={args.dataset} SEED={seed} LEVEL={level} '
                  f'AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
                  f'LABELS={row["labels_percent"]:.2f}% GAIN_AC={row["target_gain_AC"]:+.2f} '
                  f'FORGET_AC={row["original_forgetting_AC"]:+.2f}',flush=True)
    if io.digest(checkpoint)!=checkpoint_hash:
        raise RuntimeError('Initial RAL checkpoint changed.')
    overall = aggregate(output/'Results',rows,args.seeds,'RAL-full')
    print('ALL SEEDS COMPLETE '+json.dumps(overall,sort_keys=True),flush=True)
    print('OUTPUT: '+str(output),flush=True)


if __name__ == '__main__':
    main()
