"""Final AMCAL independent adaptive training seeds with a fixed pretrained CNN.

No model code is imported by this coordinator. Context development, compact
query warmup, DDQN and learned beta are independently fitted per training seed.
Preprocessing, the pretrained CNN and pre-existing Context/router warm start are
shared; this is not independent end-to-end pretraining from random weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 48, 54, 62, 72]
LEVELS = [5, 7, 10, 12, 15, 17, 20]
STAGES = ['context_cache', 'context', 'selector', 'warmstart', 'meta']
METRICS = ['AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting']
DEPENDENCIES = [
    'amcal_final_multiseed.py', 'amcal_final_multiseed_worker.py',
    'amcal_minimax_paironly.py', 'amcal_budget_actor.py',
    'amcal_competitive_selector.py', 'amcal_two_action_paper_gain.py',
    'amcal_meta_beta_ddqn.py', 'enhancements/standardized_actor.py',
    'enhancements/minimax_diagnostics.py', 'enhancements/minimax_query.py',
    'enhancements/two_action_ddqn.py', 'enhancements/meta_beta_ddqn.py',
    'enhancements/competitive_selector.py', 'enhancements/query_utility_reward.py',
    'enhancements/paper_gain_reward.py',
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def csv_write(path, rows):
    import csv
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def validate_seeds(values):
    if len(values) < 2 or len(set(values)) != len(values) or any(s < 0 for s in values):
        raise ValueError('Use at least two distinct nonnegative training seeds.')
    return list(values)


def seed_paths(output, seed):
    root = Path(output)/f'seed{seed}'
    return {name: root/name for name in [*STAGES, 'evaluation']}


def moments(values):
    if len(values) < 2 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError('At least two finite observations are required for sample SD.')
    return statistics.mean(values), statistics.stdev(values)


def check_plan(saved, current):
    if saved != current:
        raise ValueError('Immutable multiseed plan changed; use a separate output root.')


def make_plan(a):
    seeds = validate_seeds(a.seeds)
    runs = ROOT/f'ANT_Burst_{a.dataset}_TEST'/'protocol_runs'
    source = runs/('paironly_batch5_trainseed42' if a.dataset == 'IAT'
                   else 'light_ps_paironly_batch5_trainseed42')/'paironly_batch5'
    reference = runs/'meta_beta_ddqn_seed42_p300_e3_v1'
    reference_protocol = read(reference/'protocol.json')
    sp = reference_protocol['source_protocol']['protocol']
    if reference_protocol['dataset'] != a.dataset or reference_protocol['seed'] != 42:
        raise ValueError('Expected the selected final seed42 reference recipe.')
    hashes = {name: digest(source/'Models'/name) for name in
              ['protocol_preprocessing.joblib', 'initial_cnn_model.pth',
               'enhanced_context.pth', 'final_enhancement.pth']}
    for name in ['protocol_preprocessing.joblib', 'initial_cnn_model.pth', 'enhanced_context.pth']:
        if hashes[name] != sp['source_models'][name]:
            raise ValueError('Shared source differs from selected FINAL initialization: '+name)
    # The fixed evaluation uses only the original manifest/initialization, not its scores.
    execution = reference/'Results'/'testp0_budget20'/'LearnedBeta'/'execution.json'
    test_manifest = read(execution)['TEST_data']
    return dict(version=1, dataset=a.dataset, training_seeds=seeds, evaluation_seed=42,
        training_levels=[0, 1, 3], test_levels=LEVELS, test_rows=0, budget_fraction=.2,
        source=str(source), source_sha256=hashes, reference=str(reference),
        reference_protocol_sha256=digest(reference/'protocol.json'),
        fixed_TEST_data=test_manifest,
        architecture='legacy' if a.dataset == 'IAT' else 'light_ps_cnn_v1',
        context_epochs=6, context_patience=3, selector_rounds=2, selector_steps=32,
        ddqn_epochs=reference_protocol['epochs'], meta_epochs=reference_protocol['epochs'],
        train_rows_per_level=reference_protocol['pilot_rows'],
        warmstart_beta=reference_protocol['source_beta'],
        context_lr=.00025, replay_weight=.1,
        initialization='shared pretrained CNN, scaler and pretrained Context/router warm start; '
                       'independent Context development/selector/DDQN/beta fits per training seed',
        fixed='CNN weights and TRAIN/VAL/TEST partition IDs; TEST observer/stream/replay pool/order seed42',
        uncertainty_scope='variance of adaptive training conditional on common pretrained initialization; '
                          'not independent CNN or all-stage-from-scratch training',
        selection='per-seed VAL0/1/3 selection; no cross-seed or TEST model selection',
        implementation_sha256={name: digest(ROOT/name) for name in DEPENDENCIES})


def aggregate(output, plan):
    output = Path(output); all_rows = []; per_seed = []
    for seed in plan['training_seeds']:
        root = seed_paths(output, seed)['evaluation']
        completion = read(root/'completed.json')
        if (completion['training_seed'] != seed or completion['plan_sha256'] != digest(output/'plan.json')
                or completion['summary_sha256'] != digest(root/'summary_by_level.json')):
            raise ValueError('Incomplete or changed evaluation for training seed '+str(seed))
        rows = read(root/'summary_by_level.json')
        if [r['level'] for r in rows] != LEVELS or any(
                r['training_seed'] != seed or r['evaluation_seed'] != 42 for r in rows):
            raise ValueError('Evaluation levels/seeds differ.')
        all_rows.extend(rows)
        mean = dict(dataset=plan['dataset'], training_seed=seed, evaluation_seed=42,
                    levels=len(LEVELS), beta=rows[0]['beta'])
        for key in METRICS:
            mean[key] = statistics.mean(r[key] for r in rows)
        per_seed.append(mean)
    level_rows = []
    for level in LEVELS:
        selected = [r for r in all_rows if r['level'] == level]
        row = dict(dataset=plan['dataset'], level=level, seeds=len(selected))
        for key in METRICS:
            row[key+'_mean'], row[key+'_sd'] = moments([r[key] for r in selected])
        level_rows.append(row)
    overall = dict(dataset=plan['dataset'], seeds=len(per_seed), levels=len(LEVELS))
    for key in METRICS:
        overall[key+'_mean'], overall[key+'_sd'] = moments([r[key] for r in per_seed])
    csv_write(output/'Results'/'all_seed_levels.csv', all_rows)
    csv_write(output/'Results'/'summary_per_seed.csv', per_seed)
    csv_write(output/'Results'/'summary_by_level.csv', level_rows)
    csv_write(output/'Results'/'summary_overall.csv', [overall])
    write(output/'Results'/'completed.json', dict(plan_sha256=digest(output/'plan.json'),
        training_seeds=plan['training_seeds'], overall=overall,
        definition='overall SD across per-training-seed seven-level means, sample SD ddof1'))
    print('ALL TRAINING SEEDS COMPLETE | '+json.dumps(overall, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    p.add_argument('--action', choices=['All', 'Train', 'Evaluate', 'Collect'], default='All')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--output-root', type=Path)
    a = p.parse_args()
    try:
        plan = make_plan(a)
    except (ValueError, OSError) as error:
        p.error(str(error))
    output = (a.output_root or ROOT/f'ANT_Burst_{a.dataset}_TEST'/'protocol_runs'/
              'final_multiseed_fixedcnn_v1').resolve()
    source = Path(plan['source']).resolve()
    if output == source or source in output.parents or output in source.parents:
        p.error('Use a separate output root, preserving the source.')
    path = output/'plan.json'
    if path.exists():
        check_plan(read(path), plan)
    elif output.exists() and any(output.iterdir()):
        p.error('Unmatched output directory preserved; use a new --output-root.')
    elif a.action in ('Evaluate', 'Collect'):
        p.error('Run independent TRAIN first.')
    else:
        write(path, plan)
    print(f'DATASET={a.dataset} ARCHITECTURE={plan["architecture"]} '
          f'INDEPENDENT_TRAINING_SEEDS={plan["training_seeds"]} FIXED_EVALUATION_SEED=42 '
          'FIXED_CNN | TRAIN/VAL=0,1,3 | FULL TEST | BUDGET=20%', flush=True)
    for seed in plan['training_seeds']:
        stages = STAGES if a.action in ('All', 'Train') else []
        if a.action in ('All', 'Evaluate'):
            stages = [*stages, 'evaluate']
        for stage in stages:
            print(f'\n=== DATASET={a.dataset} TRAIN_SEED={seed} STAGE={stage} ===', flush=True)
            subprocess.run([sys.executable, str(ROOT/'amcal_final_multiseed_worker.py'),
                '--output-root', str(output), '--training-seed', str(seed),
                '--stage', stage, '--device', a.device], check=True, cwd=ROOT.parent)
    if a.action in ('All', 'Evaluate', 'Collect'):
        aggregate(output, plan)
    else:
        print('Independent training complete. Run Action Evaluate for full TEST.', flush=True)
    print('OUTPUT: '+str(output), flush=True)


if __name__ == '__main__':
    main()
