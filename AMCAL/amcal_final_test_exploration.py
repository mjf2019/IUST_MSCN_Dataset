"""Diagnostic epsilon-greedy TEST evaluation of existing final multiseed checkpoints.

No retraining or online Q/beta learning. The original immutable training plan,
checkpoint validation, canonical TEST stream and per-level resets are retained.
Epsilon is an explicitly chosen diagnostic value, not VAL-selected.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import amcal_final_multiseed as run
import amcal_final_multiseed_worker as worker

EPSILON = None
quarantine_partial = worker.quarantine_partial


def epsilon_tag(epsilon):
    return format(epsilon, '.8g').replace('.', 'p')


def exploratory_episode(base, experiment, learner, domain, reference, epsilon):
    original = learner.choose

    def choose(state, can_query, ignored_epsilon, rng):
        return original(state, can_query, epsilon, rng)

    learner.choose = choose
    try:
        result, trace = base.run_episode(
            experiment, learner, domain, reference, 42, training=False, mode='DDQN')
    finally:
        learner.choose = original
    for row in trace:
        row['epsilon'] = epsilon
        row['evaluation_policy'] = 'DDQN_EPSILON_GREEDY'
    result['evaluation_epsilon'] = epsilon
    return result, trace


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = paths['evaluation_exploration']
    model = paths['meta']/'Models'/'learned_beta.pth'
    done = run.read(paths['meta']/'training_complete.json')
    saved = torch.load(model, map_location='cpu', weights_only=False)
    if (done['protocol'] != saved['protocol'] or done['sha256'] != run.digest(model)
            or saved['protocol']['seed'] != seed):
        raise ValueError('Independent learned-beta checkpoint is incomplete or changed.')
    burst = run.ROOT/f'ANT_Burst_{ds}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{ds}'
    a = SimpleNamespace(dataset=ds, seed=seed, device=device,
        pilot_rows=plan['train_rows_per_level'], beta=float(saved['beta']), budget_fraction=.2,
        burst_dir=burst, teacher=paths['context'], actor_run=paths['selector'], output=root)
    exp = base.PaperGainExperiment(a)
    sp = saved['protocol']['source_protocol']['protocol']
    if (exp.teacher_hash != sp['healthy_teacher_sha256'] or exp.actor_sha != sp['actor_checkpoint_sha256']
            or exp.healthy['model_hashes'] != sp['source_models']):
        raise ValueError('Per-seed Context/selector lineage differs.')
    # A separate, data-only setup fixes TEST IDs/observer/replay/order at seed42.
    # Its trained scores/policy never select or initialize this seed's prediction.
    runs = Path(plan['reference']).parent
    fixed = SimpleNamespace(dataset=ds, seed=42, device=device,
        pilot_rows=plan['train_rows_per_level'], beta=0., budget_fraction=.2, burst_dir=burst,
        teacher=runs/'minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_natural',
        actor_run=runs/'competitive_selector_seed42_p300_r2_v1_paper_reward',
        output=root)
    canonical = base.PaperGainExperiment(fixed)
    canonical.a.pilot_rows = 0
    domains, reference = canonical.split('test_ids', base.TEST)
    if canonical.data_manifest('test_ids', base.TEST) != plan['fixed_TEST_data']:
        raise ValueError('Fixed full TEST stream/reference differs from selected final run.')
    exp.anchors = canonical.anchors
    del canonical
    a.pilot_rows = 0; a.seed = plan['evaluation_seed']
    base.pilot.base.legacy.core.set_seed(a.seed)
    source = torch.load(paths['warmstart']/'Models'/'two_action_ddqn.pth',
                        map_location='cpu', weights_only=False)
    learner = meta.learner_from_source(exp, source)
    learner.policy.load_state_dict(saved['policy_state'])
    with torch.no_grad():
        learner.log_beta.copy_(learner.log_beta.new_tensor(math.log(float(saved['beta']))))
    learner.policy.eval().requires_grad_(False); learner.log_beta.requires_grad_(False)
    before = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
              base.weight_hash(learner.policy.state_dict()), learner.beta_value(), learner.steps, learner.updates)
    execution = dict(version=1, dataset=ds, training_seed=seed, evaluation_seed=42,
        plan_sha256=run.digest(output/'plan.json'), learned_checkpoint_sha256=done['sha256'],
        evaluation_epsilon=EPSILON, evaluator_sha256=run.digest(Path(__file__)),
        beta=float(saved['beta']), TEST_data=plan['fixed_TEST_data'], test_rows=0,
        budget_fraction=.2, reset_per_level=True, policy='LearnedBetaDDQN_EpsilonGreedy',
        Q_beta='fixed TEST; Context alone adapts', source_replay='same canonical seed42 TRAIN pool',
        target_endpoints='adaptation-stream diagnostics include acquired rows; not independent target gain')
    ef = root/'execution.json'
    if ef.exists() and run.read(ef) != execution:
        raise ValueError('Existing evaluation differs; preserve output and use a new root.')
    run.write(ef, execution)
    rows = []
    for level in base.TEST:
        location = root/f'level_{level}'
        mf, tf, cf = (location/name for name in ('metrics.json', 'trace.csv', 'completed.json'))
        if cf.exists():
            record = run.read(cf)
            if record['execution_sha256'] != run.digest(ef) or any(
                    run.digest(p) != record['sha256'][p.name] for p in (mf, tf)):
                raise ValueError('Saved TEST case changed.')
            result = run.read(mf)
        else:
            quarantine_partial(location, cf)
            result, trace = exploratory_episode(base, exp, learner, domains[level], reference, EPSILON)
            location.mkdir(parents=True, exist_ok=True)
            base.pilot.write_json(mf, result); base.pilot.write_csv(tf, trace)
            run.write(cf, dict(execution_sha256=run.digest(ef),
                sha256={p.name: run.digest(p) for p in (mf, tf)}))
        if (result['samples'] != (968 if ds == 'IAT' else 801)
                or result['cap'] != int(.2*result['samples'])
                or result['labels'] > result['cap'] or result['td_updates'] != 0):
            raise RuntimeError('Full-TEST/frozen-Q/budget contract violated.')
        row = dict(dataset=ds, training_seed=seed, evaluation_seed=42, level=level,
            beta=float(saved['beta']), **result['online'], samples=result['samples'],
            labels=result['labels'], cap=result['cap'], labels_percent=result['labels_percent'],
            forgetting=result['original_domain_forgetting']['AC'])
        rows.append(row)
        print(f'TEST POLICY=DDQN_EPSILON_GREEDY EPSILON={EPSILON:g} DATASET={ds} TRAIN_SEED={seed} EVAL_SEED=42 LEVEL={level} '
              f'AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
              f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
              f'FORGET={row["forgetting"]:+.2f} pp BETA={row["beta"]:.6f}', flush=True)
    after = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
             base.weight_hash(learner.policy.state_dict()), learner.beta_value(), learner.steps, learner.updates)
    if before != after:
        raise RuntimeError('TEST modified frozen model/Q/beta state.')
    run.write(root/'summary_by_level.json', rows)
    run.csv_write(root/'summary_by_level.csv', rows)
    mean = {k: sum(r[k] for r in rows)/len(rows) for k in run.METRICS}
    run.write(root/'completed.json', dict(training_seed=seed,
        plan_sha256=run.digest(output/'plan.json'),
        summary_sha256=run.digest(root/'summary_by_level.json'), mean=mean))
    print(f'MEAN DATASET={ds} TRAIN_SEED={seed} '+json.dumps(mean, sort_keys=True), flush=True)


def main():
    global EPSILON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    parser.add_argument('--epsilon', type=float, default=.05)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.epsilon) or not 0 < args.epsilon <= 1:
        parser.error('Epsilon must be in (0,1].')
    EPSILON = args.epsilon
    output = (args.output_root or run.ROOT/f'ANT_Burst_{args.dataset}_TEST'/
              'protocol_runs'/'final_multiseed_fixedcnn_v1').resolve()
    plan = run.read(output/'plan.json')
    if plan['dataset'] != args.dataset:
        parser.error('Dataset differs from the saved training plan.')
    tag = epsilon_tag(EPSILON)
    original_evaluate = worker.evaluate

    def evaluate_separate(output, plan, seed, device, paths, base, meta, torch):
        paths = dict(paths)
        paths['evaluation_exploration'] = output/f'seed{seed}'/f'evaluation_eps{tag}_v1'
        return evaluate(output, plan, seed, device, paths, base, meta, torch)

    worker.evaluate = evaluate_separate
    original_argv = sys.argv
    try:
        for seed in plan['training_seeds']:
            print(f'=== DIAGNOSTIC TEST DATASET={args.dataset} TRAIN_SEED={seed} '
                  f'EPSILON={EPSILON:g} BUDGET=20% FULL TEST NO RETRAINING ===', flush=True)
            sys.argv = ['amcal_final_multiseed_worker.py',
                        '--output-root', str(output), '--training-seed', str(seed),
                        '--stage', 'evaluate', '--device', args.device]
            worker.main()
    finally:
        worker.evaluate = original_evaluate
        sys.argv = original_argv

    all_rows, per_seed = [], []
    for seed in plan['training_seeds']:
        root = output/f'seed{seed}'/f'evaluation_eps{tag}_v1'
        done = run.read(root/'completed.json')
        execution = run.read(root/'execution.json')
        if (done['plan_sha256'] != run.digest(output/'plan.json') or
                done['training_seed'] != seed or
                done['summary_sha256'] != run.digest(root/'summary_by_level.json') or
                execution['evaluation_epsilon'] != EPSILON or
                execution['evaluator_sha256'] != run.digest(Path(__file__))):
            raise ValueError('Incomplete or changed epsilon evaluation.')
        rows = run.read(root/'summary_by_level.json')
        if [r['level'] for r in rows] != run.LEVELS:
            raise ValueError('Expected all seven TEST levels.')
        all_rows.extend(rows)
        row = dict(dataset=args.dataset, training_seed=seed, evaluation_seed=42,
                   epsilon=EPSILON, levels=len(rows), beta=rows[0]['beta'])
        for key in run.METRICS:
            row[key] = sum(r[key] for r in rows)/len(rows)
        per_seed.append(row)
    by_level = []
    for level in run.LEVELS:
        selected = [r for r in all_rows if r['level'] == level]
        row = dict(dataset=args.dataset, level=level, epsilon=EPSILON, seeds=len(selected))
        for key in run.METRICS:
            row[key+'_mean'], row[key+'_sd'] = run.moments([r[key] for r in selected])
        by_level.append(row)
    overall = dict(dataset=args.dataset, seeds=len(per_seed), levels=len(run.LEVELS), epsilon=EPSILON)
    for key in run.METRICS:
        overall[key+'_mean'], overall[key+'_sd'] = run.moments([r[key] for r in per_seed])
    target = output/f'Results_eps{tag}_v1'
    run.csv_write(target/'all_seed_levels.csv', all_rows)
    run.csv_write(target/'summary_per_seed.csv', per_seed)
    run.csv_write(target/'summary_by_level.csv', by_level)
    run.csv_write(target/'summary_overall.csv', [overall])
    run.write(target/'completed.json', dict(plan_sha256=run.digest(output/'plan.json'),
        evaluator_sha256=run.digest(Path(__file__)), epsilon=EPSILON, overall=overall,
        scope='diagnostic TEST epsilon; not VAL-selected; original greedy results preserved'))
    print('ALL SEEDS COMPLETE | '+json.dumps(overall, sort_keys=True), flush=True)
    print('OUTPUT: '+str(target), flush=True)


if __name__ == '__main__':
    main()
