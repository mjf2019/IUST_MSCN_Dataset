"""Evaluation-only larger-budget sweep: original-paper DDQN versus uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import amcal_two_action_utility as engine

LEVELS = [5, 7, 10, 12, 15, 17, 20]
BUDGETS = [30, 40, 50, 60, 70, 80]
METHOD_NAMES = {'DDQN':'DDQN_PaperReward', 'Uncertainty':'Uncertainty'}


def check_paper_protocol(saved, completion):
    if saved.get('protocol') != completion.get('protocol'):
        raise ValueError('Paper DDQN checkpoint/completion protocols differ.')
    candidate = saved['protocol']
    if candidate.get('candidate') != 'paper' or candidate.get('utility') is not None:
        raise ValueError('Use the matched original-paper reward DDQN, not a utility candidate.')
    protocol = candidate.get('protocol', {})
    if 'synthetic' in protocol:
        raise ValueError('Synthetic-drift DDQN is not the requested checkpoint.')
    if protocol.get('budget_fraction') != .2:
        raise ValueError('Expected the completed 20%-TRAIN-budget paper DDQN.')
    if protocol.get('context_lr') != .00025 or protocol.get('replay_weight') != .1:
        raise ValueError('Paper DDQN update settings differ from the matched comparison.')
    if saved.get('selected_epoch') != completion.get('selected_epoch'):
        raise ValueError('Selected paper DDQN epoch mismatch.')
    return protocol


def evaluate_case(experiment, learner, method, budget, level, domain, reference, directory, protocol,
                  case_number=1, total_cases=1):
    if method not in METHOD_NAMES or budget not in BUDGETS or level not in LEVELS:
        raise ValueError('Unsupported method, budget or test level.')
    name = METHOD_NAMES[method]
    a = experiment.a
    a.budget_fraction = budget/100.
    count = len(domain[1]); cap = int(a.budget_fraction*count)
    prefix = f'[{case_number}/{total_cases}] DATASET={a.dataset} | METHOD={name} | TEST_BUDGET={budget}% | LEVEL={level}'
    print('\nSTART '+prefix+f' | MAX_LABELS={cap}/{count} | RESET=Context/Adam/memory', flush=True)
    case_protocol = dict(protocol=protocol, method=name, budget_percent=budget, level=level,
        stream_sha256=engine.domain_hash(domain), reference_sha256=engine.domain_hash(reference),
        samples=count, cap=cap, seed=a.seed)
    directory.mkdir(parents=True, exist_ok=True)
    execution_file = directory/'execution.json'
    completion_file = directory/'completed.json'
    metrics_file, trace_file = directory/'metrics.json', directory/'trace.csv'
    if execution_file.exists():
        if json.loads(execution_file.read_text(encoding='utf-8')) != case_protocol:
            raise ValueError('Budget case protocol changed; preserve this output and use a fresh --output.')
    elif any(directory.iterdir()):
        raise ValueError('Unmatched partial budget case preserved; choose another output.')
    else:
        engine.pilot.write_json(execution_file, case_protocol)
    reused = completion_file.exists()
    if reused:
        completion = json.loads(completion_file.read_text(encoding='utf-8'))
        if completion.get('execution_sha256') != engine.pilot.digest(execution_file):
            raise ValueError('Completed budget execution record changed.')
        if any(not p.is_file() or engine.pilot.digest(p) != completion.get('sha256', {}).get(p.name)
                for p in (metrics_file, trace_file)):
            raise ValueError('Completed budget metrics/trace are missing or changed.')
        result = json.loads(metrics_file.read_text(encoding='utf-8'))
    else:
        before_q = engine.weight_hash(learner.policy.state_dict())
        before_steps, before_updates = learner.steps, learner.updates
        result, trace = engine.run_episode(experiment, learner, domain, reference, a.seed,
            training=False, mode='DDQN' if method == 'DDQN' else 'Uncertainty')
        if (before_q != engine.weight_hash(learner.policy.state_dict()) or learner.steps != before_steps
                or learner.updates != before_updates):
            raise RuntimeError('Evaluation changed the query learner.')
        if result['cap'] != cap or result['labels'] > cap or result['td_updates'] != 0:
            raise RuntimeError('Evaluation violated the budget or frozen-DDQN contract.')
        engine.pilot.write_csv(trace_file, trace)
        engine.pilot.write_json(metrics_file, result)
        engine.pilot.write_json(completion_file, dict(execution_sha256=engine.pilot.digest(execution_file),
            sha256={p.name:engine.pilot.digest(p) for p in (metrics_file, trace_file)}))
    row = dict(dataset=a.dataset, budget_percent=budget, **engine.result_row(name, level, result))
    for key in ('PR', 'RC', 'F1'):
        row['target_final_'+key] = result['target_after'][key]
        row['forgetting_'+key] = result['original_domain_forgetting'][key]
    print(('REUSED ' if reused else 'DONE ')+prefix+
        f' | ONLINE AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}%'
        f' | LABELS={result["labels"]}/{cap} ({row["labels_percent"]:.2f}% of stream)'
        f' | FINAL_TARGET_AC={row["target_final_AC"]:.2f}%'
        f' FINAL_TARGET_F1={row["target_final_F1"]:.2f}%'
        f' | ORIGINAL_DOMAIN_FORGETTING={row["forgetting"]:+.2f} pp', flush=True)
    return row


def summarize(rows):
    summary = []
    print('\nSUMMARY: EACH ROW IS ONE DATASET / METHOD / TEST BUDGET, MEAN OVER SEVEN LEVELS', flush=True)
    for budget in sorted({r['budget_percent'] for r in rows}):
        for method in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['budget_percent'] == budget and r['method'] == method]
            record = dict(dataset=group[0]['dataset'], method=method, budget_percent=budget, levels=len(group))
            for key in ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting', 'target_final_AC',
                    'target_final_PR', 'target_final_RC', 'target_final_F1', 'source_final_AC',
                    'forgetting_PR', 'forgetting_RC', 'forgetting_F1'):
                record[key] = float(np.mean([r[key] for r in group]))
            summary.append(record)
            print(f'DATASET={record["dataset"]} | METHOD={method} | TEST_BUDGET={budget}% | '+
                ' | '.join(f'{key}={record[key]:.2f}' for key in
                    ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'target_final_AC', 'target_final_F1', 'forgetting')), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    parser.add_argument('--method', choices=['Both', 'DDQN', 'Uncertainty'], default='Both')
    parser.add_argument('--budgets', type=int, nargs='+', choices=BUDGETS, default=BUDGETS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300, help='TEST rows per intensity; zero evaluates all official TEST rows.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--paper-run', type=Path)
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 0 or 0 < a.pilot_rows < 32:
        parser.error('Use at least 32 TEST rows per intensity, or zero for full TEST.')
    a.budgets = sorted(set(a.budgets))
    test_rows = a.pilot_rows
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.paper_run = a.paper_run or root/'protocol_runs'/f'two_action_utility_seed{a.seed}_p300_e3_v1'/'Candidates'/'paper'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.output = a.output or root/'protocol_runs'/f'two_action_paper_budget_seed{a.seed}_p{test_rows}_{a.method.lower()}_v1'
    model_file = a.paper_run/'Models'/'two_action_ddqn.pth'
    completion_file = a.paper_run/'training_complete.json'
    if not model_file.is_file() or not completion_file.is_file():
        raise ValueError('Completed no-synthetic-drift paper DDQN required: '+str(a.paper_run))
    saved = torch.load(model_file, map_location='cpu', weights_only=False)
    completion = json.loads(completion_file.read_text(encoding='utf-8'))
    trained = check_paper_protocol(saved, completion)
    if engine.pilot.digest(model_file) != completion.get('checkpoint_sha256'):
        raise ValueError('Paper DDQN checkpoint checksum mismatch.')
    if trained.get('dataset') != a.dataset or trained.get('seed') != a.seed:
        raise ValueError('Paper DDQN dataset/seed mismatch.')
    for name, expected in trained['implementation_sha256'].items():
        if engine.pilot.digest(Path(__file__).parent/name) != expected:
            raise ValueError('Paper DDQN implementation provenance changed: '+name)
    # Reconstruct and validate its training lineage before changing only TEST row count/cap.
    a.pilot_rows = trained['pilot_rows']; a.budget_fraction = .2
    experiment = engine.Experiment(a)
    if (trained['healthy_teacher_sha256'] != experiment.teacher_hash or
            trained['actor_checkpoint_sha256'] != experiment.actor_sha or
            trained['source_models'] != experiment.healthy['model_hashes'] or
            trained['train_files'] != experiment.healthy['training_domain_sha256']):
        raise ValueError('Paper DDQN and healthy CNN/Context/Actor lineage differ.')
    if (trained['train_data'] != experiment.data_manifest('train_ids', engine.TRAIN) or
            trained['validation_data'] != experiment.data_manifest('val_ids', engine.TRAIN)):
        raise ValueError('Paper DDQN TRAIN/VAL input provenance differs.')
    learner = engine.QueryDDQN(experiment.spec, experiment.actor_saved['state'], engine.pilot.base.legacy.core.device, a.seed)
    learner.policy.load_state_dict(saved['policy_state'])
    learner.policy.eval().requires_grad_(False)
    a.pilot_rows = test_rows
    domains, reference = experiment.split('test_ids', LEVELS)
    methods = ['DDQN', 'Uncertainty'] if a.method == 'Both' else [a.method]
    cnn_hash = engine.weight_hash(experiment.cnn.state_dict())
    context_hash = engine.weight_hash(experiment.template.state_dict())
    code_files = ['amcal_two_action_budget_sweep.py', 'amcal_two_action_utility.py',
        'enhancements/two_action_ddqn.py', 'enhancements/competitive_selector.py']
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, methods=methods, budgets=a.budgets,
        TEST_rows_per_level=test_rows, TEST_data=experiment.data_manifest('test_ids', LEVELS),
        training_budget_fraction=.2, training_pilot_rows=trained['pilot_rows'], selected_epoch=saved['selected_epoch'],
        paper_checkpoint_sha256=engine.pilot.digest(model_file), paper_protocol=saved['protocol'],
        healthy_checkpoint_sha256=experiment.teacher_hash, actor_checkpoint_sha256=experiment.actor_sha,
        implementation_sha256={name:engine.pilot.digest(Path(__file__).parent/name) for name in code_files},
        uncertainty_threshold=experiment.threshold, uncertainty='unchanged saved TRAIN threshold; no TEST fitting or forced spending',
        context_lr=a.context_lr, replay_weight=a.replay_weight, paid_replay='batch5',
        prediction='Context -> Frozen; predict current row before its feedback',
        reset='fresh healthy Context, Adam, source/paid memory and cap for every method/budget/intensity',
        query_policy='frozen DDQN trained at 20%; greedy inference at larger caps; no retraining or online query learning',
        scope='full TEST' if test_rows == 0 else 'small-data TEST pilot; larger caps are evaluated without retraining')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol_file = a.output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Budget sweep protocol differs; preserve outputs and choose a new --output.')
    elif any(a.output.iterdir()):
        raise ValueError('Unmatched partial budget sweep preserved; use a fresh output.')
    else: engine.pilot.write_json(protocol_file, protocol)
    print(f'DATASET={a.dataset} | TRAIN_BUDGET=20% | TEST_BUDGETS={a.budgets} | METHOD={a.method}', flush=True)
    print(f'PAPER_CHECKPOINT={model_file} | SELECTED_EPOCH={saved["selected_epoch"]} | NO RETRAINING', flush=True)
    print(f'FIXED_UNCERTAINTY_THRESHOLD={experiment.threshold:.6f} | IAT=original CNN / PS=Light CNN', flush=True)
    rows = []; total = len(a.budgets)*len(methods)*len(LEVELS)
    for budget in a.budgets:
        for method in methods:
            for level in LEVELS:
                row = evaluate_case(experiment, learner, method, budget, level, domains[level], reference,
                    a.output/f'budget_{budget}'/method/f'level_{level}', protocol, len(rows)+1, total)
                rows.append(row)
                engine.pilot.write_csv(a.output/'summary_by_level.csv', rows)
            print('\nBUDGET/METHOD COMPLETE: '+f'DATASET={a.dataset} METHOD={METHOD_NAMES[method]} BUDGET={budget}%', flush=True)
            summarize([r for r in rows if r['budget_percent'] == budget and r['method'] == METHOD_NAMES[method]])
    engine.pilot.write_csv(a.output/'summary_mean.csv', summarize(rows))
    if cnn_hash != engine.weight_hash(experiment.cnn.state_dict()) or context_hash != engine.weight_hash(experiment.template.state_dict()):
        raise RuntimeError('Shared initial CNN/Context changed during the budget sweep.')
    if engine.pilot.digest(model_file) != completion['checkpoint_sha256']:
        raise RuntimeError('Paper DDQN model file changed during evaluation.')
    engine.pilot.write_json(a.output/'completed.json', dict(cases=len(rows),
        protocol_sha256=engine.pilot.digest(protocol_file),
        summary_sha256={name:engine.pilot.digest(a.output/name) for name in ('summary_by_level.csv','summary_mean.csv')}))
    print('BUDGET SWEEP COMPLETE: '+str(a.output), flush=True)


if __name__ == '__main__': main()
