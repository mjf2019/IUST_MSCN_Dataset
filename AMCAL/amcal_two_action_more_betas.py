"""Train/evaluate additional beta 25/30/35/40 candidates without TEST selection."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

import amcal_two_action_paper_gain as base

BETAS = [25., 30., 35., 40.]
LEVELS = [5, 7, 10, 12, 15, 17, 20]


def validate_parent(protocol, dataset, seed):
    if protocol.get('dataset') != dataset or protocol.get('seed') != seed:
        raise ValueError('Parent experiment dataset/seed mismatch.')
    if not {0., 10., 100.}.issubset(protocol.get('betas', [])) or 'synthetic' in protocol:
        raise ValueError('Use the completed original paper-plus-gain experiment with beta 0/10/100.')
    if (protocol.get('budget_fraction') != .2 or protocol.get('context_lr') != .00025
            or protocol.get('replay_weight') != .1 or protocol.get('pilot_rows', 0) < 32):
        raise ValueError('Parent budget, update settings or TRAIN pilot differ.')
    if not protocol.get('reward', '').startswith('query: original paper reward + beta *'):
        raise ValueError('Parent reward is not original-paper plus same-domain TRAIN gain.')


def candidate_location(parent_root, output, parent_protocol, beta):
    if beta not in BETAS: raise ValueError('Unsupported beta.')
    reused = beta in parent_protocol['betas']
    root = parent_root if reused else output
    return root/'Candidates'/base.beta_name(beta), reused


def evaluate_case(experiment, learner, beta, level, domain, reference, directory, protocol,
                  model_sha256, case_number=1, total_cases=28):
    if beta not in BETAS or level not in LEVELS: raise ValueError('Unsupported beta or test level.')
    a = experiment.a; a.beta = beta; a.budget_fraction = .2
    tag = f'[{case_number}/{total_cases}] DATASET={a.dataset} | BETA={beta:g} | LEVEL={level} | BUDGET=20%'
    cap = int(.2*len(domain[1]))
    print('\nSTART '+tag+f' | ROWS={len(domain[1])} | CAP={cap} | RESET=Context/Adam/memory', flush=True)
    execution = dict(protocol=protocol, beta=beta, level=level, model_sha256=model_sha256,
        stream_sha256=base.domain_hash(domain), source_audit_sha256=base.domain_hash(reference), cap=cap)
    directory.mkdir(parents=True, exist_ok=True)
    execution_file, complete_file = directory/'execution.json', directory/'completed.json'
    metrics_file, trace_file = directory/'metrics.json', directory/'trace.csv'
    if execution_file.exists():
        if json.loads(execution_file.read_text(encoding='utf-8')) != execution:
            raise ValueError('Case provenance differs; preserve outputs and use a fresh --output.')
    elif any(directory.iterdir()): raise ValueError('Unmatched partial beta case preserved.')
    else: base.pilot.write_json(execution_file, execution)
    reused = complete_file.exists()
    if reused:
        completion = json.loads(complete_file.read_text(encoding='utf-8'))
        if completion.get('execution_sha256') != base.pilot.digest(execution_file):
            raise ValueError('Completed beta execution changed.')
        if any(not path.is_file() or base.pilot.digest(path) != completion.get('sha256', {}).get(path.name)
                for path in (metrics_file, trace_file)):
            raise ValueError('Completed beta metrics/trace are missing or changed.')
        result = json.loads(metrics_file.read_text(encoding='utf-8'))
    else:
        weights = base.weight_hash(learner.policy.state_dict()); updates, steps = learner.updates, learner.steps
        result, trace = base.run_episode(experiment, learner, domain, reference, a.seed, training=False, mode='DDQN')
        if (weights != base.weight_hash(learner.policy.state_dict()) or updates != learner.updates
                or steps != learner.steps or result['td_updates'] != 0 or result['cap'] != cap or result['labels'] > cap):
            raise RuntimeError('TEST changed query weights or violated the feedback cap.')
        base.pilot.write_csv(trace_file, trace); base.pilot.write_json(metrics_file, result)
        base.pilot.write_json(complete_file, dict(execution_sha256=base.pilot.digest(execution_file),
            sha256={path.name:base.pilot.digest(path) for path in (metrics_file, trace_file)}))
    row = dict(dataset=a.dataset, beta=beta, budget_percent=20,
        **base.result_row(f'DDQN_beta{beta:g}', level, result))
    for key in ('PR', 'RC', 'F1'):
        row['target_final_'+key] = result['target_after'][key]
        row['forgetting_'+key] = result['original_domain_forgetting'][key]
    print(('REUSED ' if reused else 'DONE ')+tag+
        f' | ONLINE AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}%'
        f' | LABELS={result["labels"]}/{cap} ({row["labels_percent"]:.2f}%)'
        f' | FINAL_TARGET_AC={row["target_final_AC"]:.2f}% FINAL_TARGET_F1={row["target_final_F1"]:.2f}%'
        f' | ORIGINAL_DOMAIN_FORGETTING={row["forgetting"]:+.2f} pp', flush=True)
    return row


def summarize(rows):
    reports = []
    print('\nADDITIONAL-BETA TEST SUMMARY; DESCRIPTIVE COMPARISON ONLY, NO TEST MODEL SELECTION', flush=True)
    for beta in sorted({row['beta'] for row in rows}):
        group = [row for row in rows if row['beta'] == beta]
        report = dict(dataset=group[0]['dataset'], beta=beta, budget_percent=20, levels=len(group))
        for key in ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting', 'source_final_AC',
                'target_final_AC', 'target_final_PR', 'target_final_RC', 'target_final_F1',
                'forgetting_PR', 'forgetting_RC', 'forgetting_F1'):
            report[key] = float(np.mean([row[key] for row in group]))
        reports.append(report)
        print(f'DATASET={report["dataset"]} | BETA={beta:g} | BUDGET=20% | '+
            ' | '.join(f'{key}={report[key]:.2f}' for key in ('AC','PR','RC','F1','labels_percent',
                'target_final_AC','target_final_F1','forgetting')), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['IAT','PS'], required=True)
    parser.add_argument('--action', choices=['All','Train','Evaluate'], default='All')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-rows', type=int, default=300, help='Rows per TEST level; zero uses the full official TEST.')
    parser.add_argument('--probe-rows', type=int, default=24, help='Must match the parent TRAIN probe recipe.')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='cuda')
    parser.add_argument('--parent-run', type=Path)
    parser.add_argument('--candidate-run', type=Path, help='Separate reusable training folder for beta 15/20.')
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--actor-run', type=Path)
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    if a.test_rows < 0 or 0 < a.test_rows < 32: parser.error('Use zero or at least 32 TEST rows per level.')
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.parent_run = a.parent_run or root/'protocol_runs'/f'two_action_paper_gain_seed{a.seed}_p300_e3_v1'
    parent_file = a.parent_run/'protocol.json'
    parent_protocol = json.loads(parent_file.read_text(encoding='utf-8'))
    validate_parent(parent_protocol, a.dataset, a.seed)
    for name, expected in parent_protocol['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name) != expected:
            raise ValueError('Parent training implementation changed: '+name)
    a.pilot_rows = parent_protocol['pilot_rows']; a.epochs = parent_protocol['epochs']; a.budget_fraction = .2
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = a.output or root/'protocol_runs'/f'two_action_paper_gain_more25_30_35_40_seed{a.seed}_trainp{a.pilot_rows}_testp{a.test_rows}_v1'
    a.candidate_run = a.candidate_run or root/'protocol_runs'/f'two_action_paper_gain_extra_seed{a.seed}_p{a.pilot_rows}_e{a.epochs}_v1'
    output = a.output
    experiment = base.PaperGainExperiment(a)
    if (experiment.teacher_hash != parent_protocol['healthy_teacher_sha256'] or
            experiment.actor_sha != parent_protocol['actor_checkpoint_sha256'] or
            experiment.healthy['model_hashes'] != parent_protocol['source_models'] or
            experiment.healthy['training_domain_sha256'] != parent_protocol['train_files']):
        raise ValueError('Parent and current healthy model/Actor lineage differ.')
    if (experiment.data_manifest('train_ids', base.TRAIN) != parent_protocol['train_data'] or
            experiment.data_manifest('val_ids', base.TRAIN) != parent_protocol['validation_data']):
        raise ValueError('Parent TRAIN/VAL inputs differ.')
    probes, excluded_reference, anchors, probe_manifest = base.prepare_train_probes(experiment, base.pilot, a.probe_rows, 32)
    probe_manifest.update(generalization_sha256=[base.domain_hash(data) for data in probes],
        TRAIN_anchors_sha256=base.domain_hash(anchors),
        excluded_reference_use='reserved/excluded from adaptive updates; no reference loss is measured for query reward',
        loss='ordinary mean cross entropy on the single matching TRAIN level, not a mixed-domain or macro-class loss')
    if probe_manifest != parent_protocol['probes']:
        raise ValueError('Parent probe/anchor recipe differs; supply its matching --probe-rows.')
    experiment.train_anchors = anchors
    experiment.reward_probe_keys = set().union(*(set(base.pilot.feature_keys(data[0])) for data in probes))
    extension = deepcopy(parent_protocol)
    extension.update(betas=BETAS, extension='new beta candidates; same training/VAL recipe; all candidates TEST, no TEST selection',
        parent_protocol_sha256=base.pilot.digest(parent_file))
    extension['implementation_sha256']['amcal_two_action_more_betas.py'] = base.pilot.digest(Path(__file__))
    base_args = deepcopy(a)

    def configure(beta):
        experiment.a = deepcopy(base_args)
        path, reused = candidate_location(a.parent_run, a.candidate_run, parent_protocol, beta)
        experiment.a.output = path
        experiment.set_gain_reward(probes, excluded_reference, beta)
        return dict(protocol=parent_protocol if reused else extension, beta=beta), reused

    # Fail before training anything if existing beta models are unavailable or inconsistent.
    for beta in BETAS:
        if beta in parent_protocol['betas']:
            candidate_protocol, _ = configure(beta)
            base.load_selected(experiment, candidate_protocol)
    # TRAIN-only and All invocations share the same protocol without inspecting TEST for selection.
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed, betas=BETAS, test_rows_per_level=a.test_rows,
        train_rows_per_level=a.pilot_rows, epochs=a.epochs, budget_fraction=.2,
        candidate_training_folder=str(a.candidate_run),
        parent_protocol_sha256=base.pilot.digest(parent_file), extension_training_protocol=extension,
        healthy_checkpoint_sha256=experiment.teacher_hash, actor_checkpoint_sha256=experiment.actor_sha,
        scope='evaluate every beta including zero-query candidates; no cross-beta TEST selection',
        reset='healthy Context/Adam/memory/cap per beta and intensity; query weights frozen on TEST')
    output.mkdir(parents=True, exist_ok=True)
    protocol_file = output/'protocol.json'
    if protocol_file.exists():
        if json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
            raise ValueError('All-beta protocol differs; preserve outputs and use a new --output.')
    elif any(output.iterdir()): raise ValueError('Unmatched all-beta output preserved.')
    else: base.pilot.write_json(protocol_file, protocol)
    if a.action in ('Train','All'):
        for beta in BETAS:
            candidate_protocol, reused = configure(beta)
            if reused:
                print(f'DATASET={a.dataset} | BETA={beta:g} | REUSE completed parent checkpoint; no retraining', flush=True)
                continue
            print(f'\nDATASET={a.dataset} | BETA={beta:g} | PREPARE extra candidate (reuse/resume if complete) | EPOCHS={a.epochs} | TRAIN_ROWS={a.pilot_rows}', flush=True)
            base.pilot.base.legacy.core.set_seed(a.seed)
            learner = base.QueryDDQN(experiment.spec, experiment.actor_saved['state'], base.pilot.base.legacy.core.device, a.seed)
            base.train(experiment, learner, candidate_protocol)
    if a.action in ('Evaluate','All'):
        # TEST is opened only after the requested training has completed.
        experiment.a.pilot_rows = a.test_rows
        domains, reference = experiment.split('test_ids', LEVELS)
        test_manifest = experiment.data_manifest('test_ids', LEVELS)
        rows = []; models = {}
        initial = (base.weight_hash(experiment.cnn.state_dict()), base.weight_hash(experiment.template.state_dict()))
        for beta in BETAS:
            candidate_protocol, reused = configure(beta)
            learner = base.load_selected(experiment, candidate_protocol)
            learner.policy.eval().requires_grad_(False)
            model_file = experiment.a.output/'Models'/'two_action_ddqn.pth'
            model_hash = base.pilot.digest(model_file); models[base.beta_name(beta)] = dict(
                path=str(model_file), sha256=model_hash, reused_parent=reused,
                selected_epoch=json.loads((experiment.a.output/'training_complete.json').read_text(encoding='utf-8'))['selected_epoch'])
            experiment.a.pilot_rows = a.test_rows
            print(f'\nTEST ALL LEVELS | DATASET={a.dataset} | BETA={beta:g} | TEST_ROWS={a.test_rows or "FULL"} | NO QUERY LEARNING', flush=True)
            for level in LEVELS:
                rows.append(evaluate_case(experiment, learner, beta, level, domains[level], reference,
                    output/'Results'/base.beta_name(beta)/f'level_{level}', protocol, model_hash, len(rows)+1))
                base.pilot.write_csv(output/'summary_by_level.csv', rows)
            summarize([row for row in rows if row['beta'] == beta])
        base.pilot.write_csv(output/'summary_mean.csv', summarize(rows))
        base.pilot.write_json(output/'test_manifest.json', dict(test_data=test_manifest, model_checkpoints=models,
            purpose='descriptive all-beta comparison; existing VAL-selected configurations remain unchanged'))
        if initial != (base.weight_hash(experiment.cnn.state_dict()), base.weight_hash(experiment.template.state_dict())):
            raise RuntimeError('The shared healthy CNN/initial Context changed during TEST.')
        base.pilot.write_json(output/'completed.json', dict(cases=len(rows), betas=BETAS,
            protocol_sha256=base.pilot.digest(protocol_file),
            summary_sha256={name:base.pilot.digest(output/name) for name in ('summary_by_level.csv','summary_mean.csv')}))
    if (base.pilot.digest(a.teacher/'Models'/'minimax_query.pth') != experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth') != experiment.actor_sha):
        raise RuntimeError('An original source checkpoint changed.')
    print('ADDITIONAL-BETA OUTPUT: '+str(output), flush=True)


if __name__ == '__main__': main()
