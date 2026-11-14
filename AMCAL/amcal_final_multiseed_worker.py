"""Isolated worker for fixed-CNN final AMCAL multiseed training."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import amcal_final_multiseed as run


def quarantine_partial(root, completion):
    """Preserve interrupted nonresumable stages; never delete their files."""
    if root.exists() and any(root.iterdir()) and not completion.exists():
        backup = root.with_name(root.name+'.partial_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.rename(backup)
        print('Preserved incomplete stage: '+str(backup), flush=True)


def invoke_main(function, arguments):
    original = sys.argv
    try:
        sys.argv = [function.__module__, *map(str, arguments)]
        function()
    finally:
        sys.argv = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--training-seed', type=int, required=True)
    p.add_argument('--stage', choices=[*run.STAGES, 'evaluate'], required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    output = args.output_root.resolve(); plan = run.read(output/'plan.json')
    seed = args.training_seed; ds = plan['dataset']; paths = run.seed_paths(output, seed)
    if seed not in plan['training_seeds']:
        p.error('Training seed is not in the immutable plan.')
    for name, expected in plan['implementation_sha256'].items():
        if run.digest(run.ROOT/name) != expected:
            p.error('Implementation changed: '+name)
    source = Path(plan['source'])
    for name, expected in plan['source_sha256'].items():
        if run.digest(source/'Models'/name) != expected:
            p.error('Shared initialization changed: '+name)
    import torch
    import amcal_two_action_paper_gain as base
    import amcal_meta_beta_ddqn as meta
    cnn = torch.load(source/'Models'/'initial_cnn_model.pth', map_location='cpu', weights_only=False)
    if cnn.get('architecture', 'legacy') != plan['architecture']:
        p.error('Expected original Frozen IAT CNN / Light PS CNN.')
    del cnn
    burst = run.ROOT/f'ANT_Burst_{ds}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{ds}'
    common = ['--dataset', ds, '--seed', str(seed), '--device', args.device]
    print(f'DATASET={ds} TRAIN_SEED={seed} STAGE={args.stage} '
          f'FROZEN_CNN={plan["architecture"]} | CNN weights are never fitted', flush=True)
    if args.stage in ('context_cache', 'context'):
        import amcal_minimax_paironly as pilot
        values = [*common, '--source-run', str(source), '--horizon', '16',
            '--reference-size', '32', '--cost', '.05', '--gain-weight', '1',
            '--forgetting-weight', '1', '--context-lr', '.00025', '--query-lr', '.0001',
            '--replay-weight', '.1', '--budget-fraction', '.2',
            '--epochs', str(plan['context_epochs']), '--patience', str(plan['context_patience'])]
        if args.stage == 'context_cache':
            root = paths['context_cache']
            quarantine_partial(root, root/'collection_complete.json')
            pilot.main([*values, '--action', 'Diagnose', '--diagnostic-phase', 'Collect',
                '--diagnostic-rows', '256', '--diagnostic-cache-rows', '1024', '--output', str(root)])
        else:
            root = paths['context']
            quarantine_partial(root, root/'training_complete.json')
            pilot.main([*values, '--action', 'Train', '--output', str(root),
                '--actor-train-cache', str(paths['context_cache']), '--actor-control', 'natural'])
            trained = torch.load(root/'Models'/'minimax_query.pth', map_location='cpu', weights_only=False)
            if trained['settings']['seed'] != seed or set(trained['training_levels']) != {0, 1, 3}:
                raise ValueError('Context development seed/domains differ.')
            if run.digest(root/'Models'/'initial_cnn_model.pth') != plan['source_sha256']['initial_cnn_model.pth']:
                raise RuntimeError('Context training replaced the frozen CNN.')
            initial = torch.load(source/'Models'/'enhanced_context.pth',
                                 map_location='cpu', weights_only=False)['initial_state_dict']
            changed = base.weight_hash(trained['context_state_dict']) != base.weight_hash(initial)
            print(f'CONTEXT FIT COMPLETE TRAIN_SEED={seed} SELECTED_EPOCH={trained["selected_epoch"]} '
                  f'CHANGED_FROM_SHARED_WARMSTART={changed}', flush=True)
    elif args.stage == 'selector':
        import amcal_competitive_selector as selector
        invoke_main(selector.main, [*common, '--action', 'Train',
            '--pilot-rows', str(plan['train_rows_per_level']),
            '--rounds', str(plan['selector_rounds']), '--steps', str(plan['selector_steps']),
            '--original-paper-reward', '--teacher', paths['context'], '--output', paths['selector']])
    elif args.stage == 'warmstart':
        a = SimpleNamespace(dataset=ds, seed=seed, device=args.device,
            pilot_rows=plan['train_rows_per_level'], epochs=plan['ddqn_epochs'],
            beta=plan['warmstart_beta'], budget_fraction=.2, burst_dir=burst,
            teacher=paths['context'], actor_run=paths['selector'], output=paths['warmstart'])
        exp = base.PaperGainExperiment(a)
        probes, retention, anchors, manifest = base.prepare_train_probes(exp, base.pilot, 24, 32)
        exp.train_anchors = anchors
        exp.reward_probe_keys = set().union(*(set(base.pilot.feature_keys(d[0])) for d in probes))
        exp.set_gain_reward(probes, retention, a.beta)
        sp = dict(version=1, dataset=ds, seed=seed, pilot_rows=a.pilot_rows, epochs=a.epochs,
            betas=[a.beta], architecture=plan['architecture'], training_levels=[0, 1, 3],
            actor_checkpoint_sha256=exp.actor_sha, healthy_teacher_sha256=exp.teacher_hash,
            source_models=exp.healthy['model_hashes'], train_files=exp.healthy['training_domain_sha256'],
            train_data=exp.data_manifest('train_ids', base.TRAIN),
            validation_data=exp.data_manifest('val_ids', base.TRAIN),
            probes=manifest, context_lr=a.context_lr, replay_weight=a.replay_weight,
            budget_fraction=.2, source_CNN_fixed=True,
            implementation_sha256=plan['implementation_sha256'],
            scope='same final-family DDQN fixed-beta warm start; no TEST beta selection')
        protocol = dict(protocol=sp, beta=a.beta)
        if not (a.output/'training_complete.json').exists() and not (
                a.output/'Models'/'trainer_state.pth').exists():
            quarantine_partial(a.output, a.output/'training_complete.json')
        base.pilot.base.legacy.core.set_seed(seed)
        learner = base.QueryDDQN(exp.spec, exp.actor_saved['state'],
                                 base.pilot.base.legacy.core.device, seed)
        base.train(exp, learner, protocol)
    elif args.stage == 'meta':
        invoke_main(meta.main, [*common, '--action', 'Train',
            '--source-run', paths['warmstart'], '--teacher', paths['context'],
            '--actor-run', paths['selector'], '--output', paths['meta'],
            '--epochs', str(plan['meta_epochs']), '--test-rows', '0', '--test-budgets', '.2'])
    else:
        evaluate(output, plan, seed, args.device, paths, base, meta, torch)
    for name, expected in plan['source_sha256'].items():
        if run.digest(source/'Models'/name) != expected:
            raise RuntimeError('Shared initialization was modified: '+name)
    print(f'STAGE COMPLETE DATASET={ds} TRAIN_SEED={seed} STAGE={args.stage}', flush=True)


def evaluate(output, plan, seed, device, paths, base, meta, torch):
    ds = plan['dataset']; root = paths['evaluation']
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
        beta=float(saved['beta']), TEST_data=plan['fixed_TEST_data'], test_rows=0,
        budget_fraction=.2, reset_per_level=True, policy='LearnedBetaDDQN',
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
            result, trace = base.run_episode(exp, learner, domains[level], reference,
                                             42, training=False, mode='DDQN')
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
        print(f'TEST DATASET={ds} TRAIN_SEED={seed} EVAL_SEED=42 LEVEL={level} '
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


if __name__ == '__main__':
    main()
