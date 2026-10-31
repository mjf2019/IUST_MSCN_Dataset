"""Evaluation-only periodic feedback controls on the strong two-action Context."""
import argparse
import json
from pathlib import Path

import numpy as np

import amcal_two_action_utility as engine

LEVELS = [5, 7, 10, 12, 15, 17, 20]


class PeriodicQuery:
    """Schedule only; no Actor/Q forward, reward optimization or TD updates."""
    def __init__(self, skip_count):
        if skip_count not in (1, 2): raise ValueError('Use one or two skipped rows between queries.')
        self.skip_count = skip_count
        self.position = self.queries = self.steps = self.updates = 0

    def choose(self, state, can_query, epsilon, rng):
        if epsilon != 0: raise ValueError('Periodic controls are deterministic evaluation only.')
        self.position += 1
        action = int(can_query and (self.position-1) % (self.skip_count+1) == 0)
        self.queries += action
        return action, None

    def remember(self, *args):
        raise RuntimeError('Periodic feedback controls must never train a query policy.')


def validate_periodic_trace(trace, skip_count, cap):
    expected = set(list(range(1, len(trace)+1, skip_count+1))[:cap])
    used = 0
    for position, row in enumerate(trace, 1):
        query = position in expected
        used += int(query)
        if (row['step'] != position or row['queried'] != query or row['action'] != int(query)
                or row['route'] != 1 or row['labels_used'] != used):
            raise RuntimeError('Periodic schedule, causal label count or fixed Context route mismatch.')
    if used > cap: raise RuntimeError('Periodic control exceeded its feedback cap.')
    return sorted(expected)


def evaluate_level(experiment, skip_count, level, domain, reference, directory, protocol,
                   case_number=1, total_cases=1):
    if skip_count not in (1, 2) or level not in LEVELS: raise ValueError('Unsupported periodic control.')
    a = experiment.a; a.budget_fraction = .2
    method = f'PeriodicSkip{skip_count}'
    n = len(domain[1]); cap = int(.2*n)
    tag = f'[{case_number}/{total_cases}] DATASET={a.dataset} | METHOD={method} | BUDGET=20% | LEVEL={level}'
    print('\nSTART '+tag+f' | PATTERN=query1/skip{skip_count} | CAP={cap}/{n} | RESET=Context/Adam/memory', flush=True)
    execution = dict(protocol=protocol, method=method, level=level, skip_count=skip_count,
        stream_sha256=engine.domain_hash(domain), reference_sha256=engine.domain_hash(reference),
        samples=n, cap=cap, seed=a.seed)
    directory.mkdir(parents=True, exist_ok=True)
    execution_file, completed_file = directory/'execution.json', directory/'completed.json'
    metrics_file, trace_file = directory/'metrics.json', directory/'trace.csv'
    if execution_file.exists():
        if json.loads(execution_file.read_text(encoding='utf-8')) != execution:
            raise ValueError('Periodic execution differs; preserve outputs and use a fresh --output.')
    elif any(directory.iterdir()):
        raise ValueError('Unmatched partial periodic output preserved; choose a fresh output.')
    else: engine.pilot.write_json(execution_file, execution)
    reused = completed_file.exists()
    if reused:
        completed = json.loads(completed_file.read_text(encoding='utf-8'))
        if completed.get('execution_sha256') != engine.pilot.digest(execution_file):
            raise ValueError('Completed periodic execution changed.')
        if any(not path.is_file() or engine.pilot.digest(path) != completed.get('sha256', {}).get(path.name)
                for path in (metrics_file, trace_file)):
            raise ValueError('Completed periodic metrics/trace are missing or changed.')
        result = json.loads(metrics_file.read_text(encoding='utf-8'))
    else:
        schedule = PeriodicQuery(skip_count)
        # Reuse the exact existing prediction/update environment. The choose()
        # adapter supplies a periodic action; no pretrained query model is called.
        result, trace = engine.run_episode(experiment, schedule, domain, reference, a.seed,
            training=False, mode='DDQN')
        positions = validate_periodic_trace(trace, skip_count, cap)
        if (result['labels'] != len(positions) or result['cap'] != cap or result['td_updates'] != 0
                or schedule.updates != 0 or schedule.steps != 0 or schedule.position != n):
            raise RuntimeError('Periodic control budget/learning contract failed.')
        for row in trace:
            row['observer_paper_reward'] = row['reward']
            row['reward'] = 0.
            row['query_control'] = method
        result['positive_observer_paper_rewards'] = result.pop('positive_paid_rewards')
        result.update(query_control=method, query_positions=positions,
            Q_learning='disabled; deterministic periodic actions; no Actor/Q inference or query training',
            pattern=f'query first row, then skip {skip_count}; stop queries at the 20% cap',
            observer_reward='paper-table diagnostic only; no reward is used to choose requests')
        engine.pilot.write_csv(trace_file, trace)
        engine.pilot.write_json(metrics_file, result)
        engine.pilot.write_json(completed_file, dict(execution_sha256=engine.pilot.digest(execution_file),
            sha256={path.name:engine.pilot.digest(path) for path in (metrics_file, trace_file)}))
    row = dict(dataset=a.dataset, budget_percent=20, skip_count=skip_count,
        **engine.result_row(method, level, result))
    for key in ('PR', 'RC', 'F1'):
        row['target_final_'+key] = result['target_after'][key]
        row['forgetting_'+key] = result['original_domain_forgetting'][key]
    positions = result['query_positions']
    row['last_query_step'] = positions[-1] if positions else 0
    print(('REUSED ' if reused else 'DONE ')+tag+
        f' | AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}%'
        f' | LABELS={result["labels"]}/{cap} ({row["labels_percent"]:.2f}%)'
        f' | LAST_QUERY_STEP={row["last_query_step"]}'
        f' | FINAL_TARGET_AC={row["target_final_AC"]:.2f}% FINAL_TARGET_F1={row["target_final_F1"]:.2f}%'
        f' | ORIGINAL_DOMAIN_FORGETTING={row["forgetting"]:+.2f} pp', flush=True)
    return row


def summarize(rows):
    summaries = []
    print('\nPERIODIC CONTROL SUMMARY: MEAN OVER TEST LEVELS; ALL CAPS 20%', flush=True)
    for method in dict.fromkeys(row['method'] for row in rows):
        group = [row for row in rows if row['method'] == method]
        report = dict(dataset=group[0]['dataset'], method=method, budget_percent=20, levels=len(group))
        for key in ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting', 'source_final_AC',
                'target_final_AC', 'target_final_PR', 'target_final_RC', 'target_final_F1',
                'forgetting_PR', 'forgetting_RC', 'forgetting_F1', 'last_query_step'):
            report[key] = float(np.mean([row[key] for row in group]))
        summaries.append(report)
        print(f'DATASET={report["dataset"]} | METHOD={method} | BUDGET=20% | '+
            ' | '.join(f'{key}={report[key]:.2f}' for key in ('AC', 'PR', 'RC', 'F1',
                'labels_percent', 'target_final_AC', 'target_final_F1', 'forgetting', 'last_query_step')), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    parser.add_argument('--skip-counts', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pilot-rows', type=int, default=300, help='TEST rows per level; zero uses full official TEST.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.pilot_rows < 0 or 0 < a.pilot_rows < 32: parser.error('Use zero or at least 32 TEST rows per level.')
    a.skip_counts = sorted(set(a.skip_counts)); a.budget_fraction = .2
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    suffix = '_'.join(map(str, a.skip_counts))
    a.output = a.output or root/'protocol_runs'/f'periodic_query_seed{a.seed}_p{a.pilot_rows}_skip{suffix}_b20_v1'
    experiment = engine.Experiment(a)
    domains, reference = experiment.split('test_ids', LEVELS)
    initial = dict(cnn=engine.weight_hash(experiment.cnn.state_dict()),
        context=engine.weight_hash(experiment.template.state_dict()))
    files = ['amcal_two_action_periodic.py', 'amcal_two_action_utility.py',
        'amcal_competitive_selector.py', 'amcal_budget_actor.py', 'amcal_minimax_paironly.py',
        'enhancements/minimax_query.py', 'enhancements/competitive_selector.py']
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, TEST_rows_per_level=a.pilot_rows,
        skip_counts=a.skip_counts, budget_fraction=.2, test_data=experiment.data_manifest('test_ids', LEVELS),
        healthy_checkpoint_sha256=experiment.teacher_hash, parent_actor_sha256=experiment.actor_sha,
        source_models=experiment.healthy['model_hashes'], initial_state_hashes=initial,
        implementation_sha256={name:engine.pilot.digest(Path(__file__).parent/name) for name in files},
        schedule='query positions 1,3,5,... or 1,4,7,... until floor(0.2*N) paid labels; no later requests',
        query_inference='none: no Actor or DDQN forward; no uncertainty threshold; no reward optimization',
        model='same healthy original IAT CNN / Light PS CNN and Context as the paper DDQN comparison',
        context_lr=a.context_lr, replay_weight=a.replay_weight, update='unchanged batch5; current label after prediction',
        reset='Context/Adam/source-and-paid memory reset for every control and intensity',
        scope='inference-only replacement of query selection; initial pretrained Context retained; not a retrained no-RL ablation')
    a.output.mkdir(parents=True, exist_ok=True)
    protocol_file = a.output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Periodic protocol changed; preserve outputs and use a fresh --output.')
    elif any(a.output.iterdir()): raise ValueError('Unmatched periodic output preserved; use a fresh output.')
    else: engine.pilot.write_json(protocol_file, protocol)
    print(f'DATASET={a.dataset} | PERIODIC QUERY CONTROLS={a.skip_counts} | BUDGET=20% | NO RETRAINING', flush=True)
    print('Skip1: query rows 1,3,5,... | Skip2: query rows 1,4,7,... | stop at cap | Context -> Frozen', flush=True)
    rows = []; total = len(a.skip_counts)*len(LEVELS)
    for skip_count in a.skip_counts:
        for level in LEVELS:
            row = evaluate_level(experiment, skip_count, level, domains[level], reference,
                a.output/f'PeriodicSkip{skip_count}'/f'level_{level}', protocol, len(rows)+1, total)
            rows.append(row)
            engine.pilot.write_csv(a.output/'summary_by_level.csv', rows)
        summarize([row for row in rows if row['skip_count'] == skip_count])
    engine.pilot.write_csv(a.output/'summary_mean.csv', summarize(rows))
    if initial != dict(cnn=engine.weight_hash(experiment.cnn.state_dict()),
            context=engine.weight_hash(experiment.template.state_dict())):
        raise RuntimeError('The shared initial CNN/Context was changed.')
    if (engine.pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash or
            engine.pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha):
        raise RuntimeError('A source checkpoint file changed during evaluation.')
    engine.pilot.write_json(a.output/'completed.json', dict(cases=len(rows), protocol_sha256=engine.pilot.digest(protocol_file),
        summary_sha256={name:engine.pilot.digest(a.output/name) for name in ('summary_by_level.csv', 'summary_mean.csv')}))
    print('PERIODIC EVALUATION COMPLETE: '+str(a.output), flush=True)


if __name__ == '__main__': main()
