"""Evaluate existing final Light supplied-dataset checkpoints with exploration.

Clean-only training/checkpoints unchanged. Online and independent target-holdout
protocols are preserved and results are written to separate epsilon folders.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import amcal_clean_meta_beta as shared
import amcal_supplied_light_meta_beta as light
from amcal_clean_meta_beta import (
    load_selected, attack_cases, split_targets, purge_replay,
    summary_row, METRICS, ATTACKS, csv, write, read, run_holdout_case)

api = shared.api
base = shared.base
EPSILON = None


def epsilon_tag(epsilon):
    return format(epsilon, '.8g').replace('.', 'p')


def exploration_runner(original, epsilon):
    def episode(experiment, learner, *args, **kwargs):
        if kwargs.get('training', False) or (len(args) >= 4 and args[3]):
            raise ValueError('This wrapper only allows evaluation; no training.')
        choose = learner.choose

        def explore(state, can_query, ignored_epsilon, rng):
            return choose(state, can_query, epsilon, rng)

        learner.choose = explore
        try:
            result, trace = original(experiment, learner, *args, **kwargs)
        finally:
            learner.choose = choose
        for row in trace:
            row['epsilon'] = epsilon
            row['evaluation_policy'] = 'DDQN_EPSILON_GREEDY'
        result['evaluation_epsilon'] = epsilon
        return result, trace
    return episode


def evaluate(exp, protocol):
    a = exp.a; learner, done = load_selected(exp, protocol)
    domains, descriptions = attack_cases(exp)
    scopes = ['Online', 'Independent'] if a.protocol == 'Both' else [a.protocol]
    full_anchors = exp.anchors
    policy_before = base.weight_hash(learner.policy.state_dict())
    for scope in scopes:
        exp.anchors = full_anchors
        root = a.output/f'Results_eps{epsilon_tag(EPSILON)}_v1'/scope
        if scope == 'Independent':
            streams, audits, split = split_targets(domains, a.seed)
            exp.anchors, purge = purge_replay(full_anchors, audits)
        else:
            streams, audits, split, purge = domains, {}, None, None
        execution = dict(protocol=protocol, checkpoint_sha256=done['sha256'],
            test_scope=scope, budget_fraction=.2, model_reset_per_intensity=True,
            evaluation_epsilon=EPSILON, evaluation_policy='DDQN_EPSILON_GREEDY',
            evaluator_sha256=api.digest(Path(__file__)),
            source_reference=base.domain_hash(exp.validation),
            targets={str(k): base.domain_hash(d) for k, d in domains.items()},
            split=split, replay_purge=purge,
            metric_order=list(METRICS), beta=learner.beta_value())
        ef = root/'execution.json'
        if ef.exists() and read(ef) != execution:
            raise ValueError('Existing TEST recipe differs; results preserved.')
        write(ef, execution)
        rows = []
        for case, description in descriptions.items():
            location = root/description['attack']/('level_'+str(description['level']))
            mf, tf, cf = (location/name for name in ('metrics.json','trace.csv','completed.json'))
            if cf.is_file():
                record = read(cf)
                if record['execution_sha256'] != api.digest(ef) or any(
                        api.digest(p) != record['sha256'][p.name] for p in (mf, tf)):
                    raise ValueError('Completed target case changed.')
                result = read(mf)
            else:
                if scope == 'Independent':
                    result, trace = run_holdout_case(exp, learner, streams[case], audits[case],
                        exp.validation, a.seed, 'LearnedBetaDDQN', 1)
                else:
                    result, trace = base.run_episode(exp, learner, streams[case], exp.validation,
                        a.seed, training=False)
                    result['adaptation_stream_seen_before'] = result.pop('target_before')
                    result['adaptation_stream_seen_after'] = result.pop('target_after')
                    result['target_final_definition'] = 'not reported; Online uses all TEST rows for pre-feedback scoring'
                result['case'] = description
                write(mf, result); csv(tf, trace)
                write(cf, dict(execution_sha256=api.digest(ef),
                    sha256={p.name: api.digest(p) for p in (mf, tf)}))
            if (result['samples'] != len(streams[case][1]) or result['td_updates'] != 0
                    or result['cap'] != int(.2*len(streams[case][1]))
                    or not 0 <= result['labels'] <= result['cap']):
                raise RuntimeError('TEST sample, frozen-Q or budget contract violated.')
            row = summary_row(a.dataset, scope, description, result, learner.beta_value())
            rows.append(row)
            values = ' '.join(f'{k}={row[k]:.2f}%' for k in METRICS)
            final = f' FINAL_HOLDOUT_AC={row["target_final_AC"]:.2f}%' if scope == 'Independent' else ''
            print(f'TEST EPSILON={EPSILON:g} POLICY=DDQN_EPSILON_GREEDY DATASET={a.dataset} SCOPE={scope} ATTACK={description["attack"]} '
                f'STRENGTH={description["strength"]} {values} N={row["samples"]} '
                f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
                f'CAP_USED={row["budget_used_percent"]:.2f}% '
                f'ORIGINAL_FORGETTING={row["original_forgetting_AC"]:+.2f}{final}', flush=True)
        csv(root/'summary_by_intensity.csv', rows)
        means = []
        for attack in [*ATTACKS, 'ALL']:
            selected = [r for r in rows if attack == 'ALL' or r['attack'] == attack]
            mean = dict(dataset=a.dataset, protocol=scope, attack=attack, intensities=len(selected))
            numeric = [k for k in rows[0] if k not in ('dataset','protocol','method','attack','strength','level')]
            mean.update({k: float(np.mean([r[k] for r in selected])) for k in numeric})
            means.append(mean)
            print('MEAN '+json.dumps(mean, sort_keys=True), flush=True)
        csv(root/'summary_means.csv', means)
        write(root/'completed.json', dict(execution_sha256=api.digest(ef),
            cases=len(rows), summary_sha256=api.digest(root/'summary_by_intensity.csv')))
    exp.anchors = full_anchors
    if policy_before != base.weight_hash(learner.policy.state_dict()) or learner.steps or learner.updates:
        raise RuntimeError('TEST updated the frozen DDQN.')
    if not math.isclose(learner.beta_value(), done['selected_beta'], rel_tol=1e-6):
        raise RuntimeError('TEST updated the frozen beta.')


def main():
    global EPSILON
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=list(shared.DATASETS), required=True)
    p.add_argument('--epsilon', type=float, default=.05)
    p.add_argument('--protocol', choices=['Both', 'Online', 'Independent'], default='Both')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--source-root', type=Path)
    p.add_argument('--model-run', type=Path)
    a = p.parse_args()
    if not math.isfinite(a.epsilon) or not 0 < a.epsilon <= 1:
        p.error('Use a finite epsilon in (0,1].')
    EPSILON = a.epsilon
    original_evaluate, original_episode, original_argv = shared.evaluate, base.run_episode, sys.argv
    arguments = ['amcal_supplied_light_meta_beta.py', '--dataset', a.dataset,
                 '--action', 'Evaluate', '--protocol', a.protocol,
                 '--seed', str(a.seed), '--device', a.device]
    if a.source_root is not None:
        arguments += ['--source-root', str(a.source_root)]
    if a.model_run is not None:
        arguments += ['--output', str(a.model_run)]
    shared.evaluate = evaluate
    base.run_episode = exploration_runner(original_episode, EPSILON)
    print(f'=== TEST ONLY DATASET={a.dataset} EPSILON={EPSILON:g} '
          f'PROTOCOL={a.protocol} BUDGET=20% FULL TEST '
          'EXISTING CHECKPOINT | NO RETRAINING | Q/BETA FIXED ===', flush=True)
    try:
        sys.argv = arguments
        light.main()
    finally:
        shared.evaluate, base.run_episode, sys.argv = original_evaluate, original_episode, original_argv


if __name__ == '__main__':
    main()
