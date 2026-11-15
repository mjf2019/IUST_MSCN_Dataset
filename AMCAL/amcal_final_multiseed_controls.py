"""Final five-training-seed TEST query replacements, without retraining.

Random: immediate Bernoulli(.2), strict cap, no quota filling.
Uncertainty: causal entropy ranking in arrived windows of five, exact cap.
Both keep the same per-seed initial Context and canonical TEST/reference/replay.
These are deployment controls, not independently trained no-RL models.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import amcal_final_multiseed as run
import amcal_final_multiseed_worker as worker

METHODS = ('Random', 'Uncertainty')
METRICS = [*run.METRICS, 'source_initial_AC', 'source_final_AC',
           'target_initial_AC', 'target_final_AC', 'milliseconds_per_row',
           'mean_feedback_delay_rows']


def run_control(base, window_runner, exp, learner, domain, reference, method):
    if method == 'Random':
        return base.run_episode(exp, learner, domain, reference, 42,
                                training=False, mode='Random')
    if method == 'Uncertainty':
        return window_runner(exp, learner, domain, reference, 42,
                             'UncertaintyWindow', 5)
    raise ValueError('Unknown TEST control.')


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = output/f'seed{seed}'/'test_controls_v1'
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

    from enhancements.causal_window_query import run_window_episode
    for method in METHODS:
        location = root/method
        execution = dict(version=1, dataset=ds, training_seed=seed,
            evaluation_seed=42, method=method, levels=run.LEVELS, test_rows=0,
            plan_sha256=run.digest(output/'plan.json'), checkpoint_sha256=done['sha256'],
            beta=float(saved['beta']), TEST_data=plan['fixed_TEST_data'],
            budget_fraction=.2, reset_per_level=True,
            prediction='Context -> frozen CNN; prediction before feedback',
            query=('Bernoulli(.2), immediate feedback, strict cap, no quota filling'
                   if method == 'Random' else
                   'causal entropy ranking, arrived window=5, no threshold, exact floor cap'),
            scope='deployment query replacement, not independent no-RL training',
            endpoint_scope='target endpoints use adaptation-stream rows, not independent target gain',
            evaluator_sha256=run.digest(Path(__file__)),
            window_sha256=run.digest(run.ROOT/'enhancements'/'causal_window_query.py'),
            source_replay='canonical seed42 TRAIN pool; same as final DDQN TEST')
        ef = location/'execution.json'
        if ef.exists():
            if run.read(ef) != execution:
                raise ValueError('Existing TEST control recipe changed; preserve results.')
        elif location.exists() and any(location.iterdir()):
            raise ValueError('Unmatched TEST control directory.')
        else:
            run.write(ef, execution)
        rows = []
        for level in run.LEVELS:
            case = location/f'level_{level}'
            mf, tf, cf = [case/name for name in ('metrics.json', 'trace.csv', 'completed.json')]
            if cf.exists():
                record = run.read(cf)
                if (record['execution_sha256'] != run.digest(ef) or
                        any(run.digest(p) != record['sha256'][p.name] for p in (mf, tf))):
                    raise ValueError('Saved TEST control case changed.')
                result = run.read(mf)
            else:
                # Each episode restores Context/Adam/memory and uses the same stream order.
                base.pilot.base.legacy.core.set_seed(42)
                result, trace = run_control(base, run_window_episode, exp, learner,
                                            domains[level], reference, method)
                base.pilot.write_json(mf, result)
                base.pilot.write_csv(tf, trace)
                run.write(cf, dict(execution_sha256=run.digest(ef),
                    sha256={p.name:run.digest(p) for p in (mf, tf)}))
            if (result['samples'] != (968 if ds == 'IAT' else 801) or
                    result['samples'] != len(domains[level][1]) or
                    result['cap'] != int(.2*result['samples']) or
                    not 0 <= result['labels'] <= result['cap'] or result['td_updates'] != 0):
                raise RuntimeError('Full-TEST/frozen-Q/budget contract violated.')
            if method == 'Uncertainty' and result['labels'] != result['cap']:
                raise RuntimeError('Uncertainty window did not exhaust the exact legal cap.')
            row = dict(dataset=ds, method=method, training_seed=seed, evaluation_seed=42,
                level=level, **result['online'], samples=result['samples'],
                labels=result['labels'], cap=result['cap'], labels_percent=result['labels_percent'],
                forgetting=result['original_domain_forgetting']['AC'],
                source_initial_AC=result['source_before']['AC'],
                source_final_AC=result['source_after']['AC'],
                target_initial_AC=result['target_before']['AC'],
                target_final_AC=result['target_after']['AC'],
                milliseconds_per_row=result['milliseconds_per_row'],
                mean_feedback_delay_rows=result.get('mean_feedback_delay_rows', 0.))
            rows.append(row)
            print(f'TEST DATASET={ds} METHOD={method} TRAIN_SEED={seed} EVAL_SEED=42 '
                  f'LEVEL={level} AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% '
                  f'RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
                  f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
                  f'ORIGINAL_FORGETTING={row["forgetting"]:+.2f} pp '
                  f'FINAL_TARGET_AC={row["target_final_AC"]:.2f}%', flush=True)
        after = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
                 base.weight_hash(learner.policy.state_dict()), learner.beta_value(),
                 learner.steps, learner.updates)
        if before != after:
            raise RuntimeError('Control modified the initial CNN/Context or DDQN/beta.')
        if run.digest(model) != done['sha256']:
            raise RuntimeError('Control modified the saved checkpoint.')
        run.write(location/'summary_by_level.json', rows)
        run.csv_write(location/'summary_by_level.csv', rows)
        mean = {k:sum(r[k] for r in rows)/len(rows) for k in METRICS}
        run.write(location/'completed.json', dict(training_seed=seed, method=method,
            execution_sha256=run.digest(ef), plan_sha256=run.digest(output/'plan.json'),
            summary_sha256=run.digest(location/'summary_by_level.json'), mean=mean))
        print(f'SEED COMPLETE DATASET={ds} METHOD={method} TRAIN_SEED={seed} '
              +json.dumps(mean, sort_keys=True), flush=True)


def aggregate(output, plan, method):
    all_rows, per_seed = [], []
    for seed in plan['training_seeds']:
        root = output/f'seed{seed}'/'test_controls_v1'/method
        done = run.read(root/'completed.json')
        execution = run.read(root/'execution.json')
        if (done['training_seed'] != seed or done['method'] != method or
                done['plan_sha256'] != run.digest(output/'plan.json') or
                done['execution_sha256'] != run.digest(root/'execution.json') or
                done['summary_sha256'] != run.digest(root/'summary_by_level.json') or
                execution['evaluator_sha256'] != run.digest(Path(__file__))):
            raise ValueError('Incomplete or changed TEST control results.')
        rows = run.read(root/'summary_by_level.json')
        if ([r['level'] for r in rows] != run.LEVELS or
                any(r['training_seed'] != seed or r['method'] != method for r in rows)):
            raise ValueError('TEST levels/seeds/method differ.')
        all_rows.extend(rows)
        row = dict(dataset=plan['dataset'], method=method, training_seed=seed,
                   evaluation_seed=42, levels=len(rows))
        for key in METRICS:
            row[key] = sum(r[key] for r in rows)/len(rows)
        per_seed.append(row)
    by_level = []
    for level in run.LEVELS:
        selected = [r for r in all_rows if r['level'] == level]
        row = dict(dataset=plan['dataset'], method=method, level=level, seeds=len(selected))
        for key in METRICS:
            row[key+'_mean'], row[key+'_sd'] = run.moments([r[key] for r in selected])
        by_level.append(row)
    overall = dict(dataset=plan['dataset'], method=method, seeds=len(per_seed), levels=len(run.LEVELS))
    for key in METRICS:
        overall[key+'_mean'], overall[key+'_sd'] = run.moments([r[key] for r in per_seed])
    root = output/'Results_test_controls_v1'/method
    run.csv_write(root/'all_seed_levels.csv', all_rows)
    run.csv_write(root/'summary_per_seed.csv', per_seed)
    run.csv_write(root/'summary_by_level.csv', by_level)
    run.csv_write(root/'summary_overall.csv', [overall])
    run.write(root/'completed.json', dict(method=method, training_seeds=plan['training_seeds'],
        plan_sha256=run.digest(output/'plan.json'), evaluator_sha256=run.digest(Path(__file__)),
        overall=overall))
    print('ALL SEEDS COMPLETE '+json.dumps(overall, sort_keys=True), flush=True)
    print('OUTPUT: '+str(root), flush=True)


def main():
    global METHODS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    p.add_argument('--method', choices=['Both', 'Random', 'Uncertainty'], default='Both')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--action', choices=['Evaluate', 'Collect'], default='Evaluate')
    args = p.parse_args()
    METHODS = ('Random', 'Uncertainty') if args.method == 'Both' else (args.method,)
    output = (args.output_root or run.ROOT/f'ANT_Burst_{args.dataset}_TEST'/
              'protocol_runs'/'final_multiseed_fixedcnn_v1').resolve()
    plan = run.read(output/'plan.json')
    if (plan['dataset'] != args.dataset or plan['evaluation_seed'] != 42 or
            plan['test_rows'] != 0 or plan['budget_fraction'] != .2 or
            sorted(plan['training_seeds']) != [42, 48, 54, 62, 72]):
        p.error('Expected the completed final five-seed full-TEST plan.')
    old_evaluate, old_argv = worker.evaluate, sys.argv
    worker.evaluate = evaluate
    try:
        if args.action == 'Evaluate':
            for seed in plan['training_seeds']:
                print(f'=== TEST ONLY DATASET={args.dataset} METHODS={",".join(METHODS)} '
                      f'TRAIN_SEED={seed} FULL TEST BUDGET=20% NO RETRAINING ===', flush=True)
                sys.argv = ['amcal_final_multiseed_worker.py',
                            '--output-root', str(output), '--training-seed', str(seed),
                            '--stage', 'evaluate', '--device', args.device]
                worker.main()
    finally:
        worker.evaluate, sys.argv = old_evaluate, old_argv
    for method in METHODS:
        aggregate(output, plan, method)


if __name__ == '__main__':
    main()
