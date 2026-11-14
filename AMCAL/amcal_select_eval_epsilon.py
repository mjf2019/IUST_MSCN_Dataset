"""Choose a common inference epsilon on VAL using existing final checkpoints.

No retraining; no TEST evaluation or TEST-result reading. Earlier diagnostic TEST
inspection remains part of the experiment history. This is post-hoc VAL calibration.
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import amcal_final_multiseed as run

EPSILONS = [0., .01, .05]
LEVELS = [0, 1, 3]
EVALUATION_SEED = 42
NAME = 'epsilon_val_selection_v1'


def select_candidate(groups):
    """Equal weight per dataset, training seed and VAL domain."""
    reports = []
    for epsilon in EPSILONS:
        values = []
        for dataset in ('IAT', 'PS'):
            rows = groups[dataset]
            seeds = sorted({r['training_seed'] for r in rows})
            if len(seeds) != 5:
                raise ValueError('Expected five training seeds per dataset.')
            seed_values = []
            for seed in seeds:
                chosen = [r for r in rows if r['training_seed'] == seed and r['epsilon'] == epsilon]
                if sorted(r['level'] for r in chosen) != LEVELS or len(chosen) != 3:
                    raise ValueError('Expected exactly one result per VAL level.')
                seed_values.append({
                    'score': sum(r['AC']-max(0., r['forgetting']) for r in chosen)/3,
                    'AC': sum(r['AC'] for r in chosen)/3,
                    'F1': sum(r['F1'] for r in chosen)/3,
                    'labels_percent': sum(r['labels_percent'] for r in chosen)/3})
            values.append({k: sum(v[k] for v in seed_values)/len(seed_values)
                           for k in seed_values[0]})
        report = {'epsilon': epsilon}
        report.update({k: sum(v[k] for v in values)/2 for k in values[0]})
        if any(not math.isfinite(v) for v in report.values()):
            raise ValueError('Nonfinite validation selection metric.')
        reports.append(report)
    top = max(r['score'] for r in reports)
    tied = [r for r in reports if top-r['score'] <= 1e-9]
    fewest = min(r['labels_percent'] for r in tied)
    tied = [r for r in tied if r['labels_percent']-fewest <= 1e-9]
    return min(tied, key=lambda r: r['epsilon']), reports


def paths_for(dataset):
    output = run.ROOT/f'ANT_Burst_{dataset}_TEST'/'protocol_runs'/'final_multiseed_fixedcnn_v1'
    return output, output/NAME


def recipe_for(output, dataset):
    plan = run.read(output/'plan.json')
    if plan['dataset'] != dataset or plan['training_levels'] != LEVELS:
        raise ValueError('Unexpected saved training plan.')
    if sorted(plan['training_seeds']) != [42, 48, 54, 62, 72]:
        raise ValueError('Expected the existing five training seeds.')
    dependencies = {}
    for name, expected in plan['implementation_sha256'].items():
        actual = run.digest(run.ROOT/name)
        if actual != expected:
            raise ValueError('Training implementation changed: '+name)
        dependencies[name] = actual
    for name, expected in plan['source_sha256'].items():
        if run.digest(Path(plan['source'])/'Models'/name) != expected:
            raise ValueError('Shared initialization changed: '+name)
    checkpoints = {}
    for seed in plan['training_seeds']:
        meta = output/f'seed{seed}'/'meta'
        done = run.read(meta/'training_complete.json')
        sha = run.digest(meta/'Models'/'learned_beta.pth')
        if done['sha256'] != sha or done['protocol']['seed'] != seed:
            raise ValueError('Incomplete or changed learned-beta checkpoint.')
        checkpoints[str(seed)] = sha
    recipe = dict(version=1, dataset=dataset, training_seeds=plan['training_seeds'],
        evaluation_seed=EVALUATION_SEED, epsilons=EPSILONS, levels=LEVELS,
        rows_per_VAL_level=plan['train_rows_per_level'], budget_fraction=.2,
        plan_sha256=run.digest(output/'plan.json'), checkpoints=checkpoints,
        evaluator_sha256=run.digest(Path(__file__)),
        dependencies=dependencies, reset_per_case=True,
        selection='mean online AC minus positive original-domain forgetting; '
                  'equal dataset/seed/domain weight; ties favor fewer labels then smaller epsilon',
        Q_beta='fixed; only temporary Context adapts; no training changes',
        scope='VAL-only post-hoc calibration; earlier diagnostic TEST inspection is not erased')
    return plan, recipe


def explore(base, exp, learner, domain, reference, epsilon):
    original = learner.choose

    def choose(state, can_query, ignored_epsilon, rng):
        return original(state, can_query, epsilon, rng)

    learner.choose = choose
    try:
        result, trace = base.run_episode(exp, learner, domain, reference,
            EVALUATION_SEED, training=False, mode='DDQN')
    finally:
        learner.choose = original
    for row in trace:
        row['epsilon'] = epsilon
        row['evaluation_policy'] = 'VAL_EPSILON_GREEDY'
    return result, trace


def evaluate_dataset(dataset, device):
    import torch
    import amcal_two_action_paper_gain as base
    import amcal_meta_beta_ddqn as meta

    output, root = paths_for(dataset)
    plan, recipe = recipe_for(output, dataset)
    execution = root/'execution.json'
    if execution.exists():
        if run.read(execution) != recipe:
            raise ValueError('VAL calibration recipe changed; preserve existing results.')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Unmatched VAL output directory.')
    else:
        run.write(execution, recipe)
    rows = []
    for seed in plan['training_seeds']:
        paths = run.seed_paths(output, seed)
        saved = torch.load(paths['meta']/'Models'/'learned_beta.pth',
                           map_location='cpu', weights_only=False)
        done = run.read(paths['meta']/'training_complete.json')
        if saved['protocol'] != done['protocol']:
            raise ValueError('Model/completion protocol mismatch.')
        a = SimpleNamespace(dataset=dataset, seed=seed, device=device,
            pilot_rows=plan['train_rows_per_level'], beta=float(saved['beta']),
            budget_fraction=.2, burst_dir=run.ROOT/f'ANT_Burst_{dataset}_TEST'/
            'Dataset_official_v1'/f'AdvBurst_FTSC-{dataset}',
            teacher=paths['context'], actor_run=paths['selector'], output=root)
        exp = base.PaperGainExperiment(a)
        source_protocol = saved['protocol']['source_protocol']['protocol']
        if (exp.teacher_hash != source_protocol['healthy_teacher_sha256'] or
                exp.actor_sha != source_protocol['actor_checkpoint_sha256'] or
                exp.healthy['model_hashes'] != source_protocol['source_models']):
            raise ValueError('Context/selector/base lineage differs.')
        if exp.data_manifest('val_ids', LEVELS) != source_protocol['validation_data']:
            raise ValueError('VAL stream/reference differs from saved development protocol.')
        domains, reference = exp.split('val_ids', LEVELS)
        source = torch.load(paths['warmstart']/'Models'/'two_action_ddqn.pth',
                            map_location='cpu', weights_only=False)
        learner = meta.learner_from_source(exp, source)
        learner.policy.load_state_dict(saved['policy_state'])
        with torch.no_grad():
            learner.log_beta.copy_(learner.log_beta.new_tensor(math.log(float(saved['beta']))))
        learner.policy.eval().requires_grad_(False)
        learner.log_beta.requires_grad_(False)
        before = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
                  base.weight_hash(learner.policy.state_dict()), learner.beta_value(),
                  learner.steps, learner.updates)
        for epsilon in EPSILONS:
            for level in LEVELS:
                tag = format(epsilon, '.8g').replace('.', 'p')
                location = root/f'seed{seed}'/f'epsilon_{tag}'/f'level_{level}'
                metric, trace_file, complete = [location/f for f in
                    ('metrics.json', 'trace.csv', 'completed.json')]
                if complete.exists():
                    record = run.read(complete)
                    if (record['execution_sha256'] != run.digest(execution) or
                            any(run.digest(f) != record['sha256'][f.name] for f in (metric, trace_file))):
                        raise ValueError('Saved VAL case changed.')
                    result = run.read(metric)
                else:
                    base.pilot.base.legacy.core.set_seed(EVALUATION_SEED)
                    result, trace = explore(base, exp, learner, domains[level], reference, epsilon)
                    base.pilot.write_json(metric, result)
                    base.pilot.write_csv(trace_file, trace)
                    run.write(complete, dict(execution_sha256=run.digest(execution),
                        sha256={f.name:run.digest(f) for f in (metric, trace_file)}))
                if (result['samples'] != len(domains[level][1]) or
                        result['cap'] != int(.2*result['samples']) or
                        not 0 <= result['labels'] <= result['cap'] or result['td_updates'] != 0):
                    raise ValueError('VAL/frozen-Q/budget contract violated.')
                row = dict(dataset=dataset, training_seed=seed, evaluation_seed=EVALUATION_SEED,
                    epsilon=epsilon, **base.result_row('VAL_EPSILON_GREEDY', level, result))
                rows.append(row)
                print(f'VAL DATASET={dataset} TRAIN_SEED={seed} EPSILON={epsilon:g} LEVEL={level} '
                      f'AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% '
                      f'LABELS={result["labels"]}/{result["cap"]} '
                      f'FORGET={row["forgetting"]:+.2f} pp', flush=True)
        after = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
                 base.weight_hash(learner.policy.state_dict()), learner.beta_value(),
                 learner.steps, learner.updates)
        if before != after:
            raise RuntimeError('VAL modified frozen initial model/Q/beta.')
    run.csv_write(root/'all_val_cases.csv', rows)
    run.write(root/'all_val_cases.json', rows)
    run.write(root/'completed.json', dict(execution_sha256=run.digest(execution),
        summary_sha256=run.digest(root/'all_val_cases.json'), cases=len(rows)))
    print(f'VAL COMPLETE DATASET={dataset} CASES={len(rows)} OUTPUT={root}', flush=True)


def select():
    groups, provenance = {}, {}
    for dataset in ('IAT', 'PS'):
        output, root = paths_for(dataset)
        _, recipe = recipe_for(output, dataset)
        if run.read(root/'execution.json') != recipe:
            raise ValueError('VAL recipe changed.')
        done = run.read(root/'completed.json')
        if (done['execution_sha256'] != run.digest(root/'execution.json') or
                done['summary_sha256'] != run.digest(root/'all_val_cases.json') or done['cases'] != 45):
            raise ValueError('Both datasets must have all 45 VAL cases.')
        groups[dataset] = run.read(root/'all_val_cases.json')
        provenance[dataset] = dict(execution_sha256=done['execution_sha256'],
                                  summary_sha256=done['summary_sha256'])
    selected, reports = select_candidate(groups)
    target = run.ROOT/'final_epsilon_val_selection_v1'
    run.csv_write(target/'comparison.csv', reports)
    run.write(target/'selection.json', dict(selected_epsilon=selected['epsilon'],
        selected=selected, candidates=reports, provenance=provenance,
        scope='one common inference epsilon; VAL0/1/3 only; fixed existing checkpoints',
        history='post-hoc VAL calibration after earlier diagnostic TEST exploration'))
    for report in reports:
        print('VAL COMPARISON '+json.dumps(report, sort_keys=True), flush=True)
    print(f'SELECTED COMMON EPSILON={selected["epsilon"]:g} | NO RETRAINING | '
          f'SELECTION FILE={target/"selection.json"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS', 'Both'], default='Both')
    p.add_argument('--action', choices=['Evaluate', 'Select', 'All'], default='All')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    if args.action in ('Evaluate', 'All'):
        for dataset in (['IAT', 'PS'] if args.dataset == 'Both' else [args.dataset]):
            evaluate_dataset(dataset, args.device)
    if args.action in ('Select', 'All'):
        select()


if __name__ == '__main__':
    main()
