"""Evaluation-only non-window random querying on the FINAL AMCAL checkpoint.

Same healthy CNN/initial Context, official full streams, reference, seed, batch5,
LR and replay as the selected learned-beta run. Only the deployment query rule
changes. This is NOT independently retrained no-RL Context development.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import amcal_two_action_paper_gain as base

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    runs = ROOT/f'ANT_Burst_{a.dataset}_TEST'/'protocol_runs'
    model_run = runs/f'meta_beta_ddqn_seed{a.seed}_p300_e3_v1'
    model_file = model_run/'Models'/'learned_beta.pth'
    done = read(model_run/'training_complete.json')
    saved = torch.load(model_file, map_location='cpu', weights_only=False)
    learned_protocol = saved['protocol']
    if (done['protocol'] != learned_protocol or done['sha256'] != base.pilot.digest(model_file)
            or learned_protocol['dataset'] != a.dataset or learned_protocol['seed'] != a.seed):
        raise ValueError('Final learned-beta checkpoint/completion mismatch.')
    sp = learned_protocol['source_protocol']['protocol']
    for hashes in (sp['implementation_sha256'], learned_protocol['implementation_sha256']):
        for name, expected in hashes.items():
            if base.pilot.digest(ROOT/name) != expected:
                raise ValueError('Source implementation changed: '+name)
    a.pilot_rows = sp['pilot_rows']
    a.beta = float(saved['beta'])
    a.budget_fraction = .2
    a.burst_dir = ROOT/f'ANT_Burst_{a.dataset}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = runs/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = runs/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = (a.output or runs/f'final_random_full_seed{a.seed}_v1').resolve()
    for source in (model_run, a.teacher, a.actor_run):
        source = source.resolve()
        if a.output == source or source in a.output.parents or a.output in source.parents:
            p.error('Output must be separate from source checkpoints.')
    exp = base.PaperGainExperiment(a)
    if (exp.teacher_hash != sp['healthy_teacher_sha256'] or exp.actor_sha != sp['actor_checkpoint_sha256']
            or exp.healthy['model_hashes'] != sp['source_models']
            or exp.healthy['training_domain_sha256'] != sp['train_files']):
        raise ValueError('Final CNN/Context/Actor lineage differs.')
    if (exp.data_manifest('train_ids', base.TRAIN) != sp['train_data']
            or exp.data_manifest('val_ids', base.TRAIN) != sp['validation_data']):
        raise ValueError('Source TRAIN/VAL data differ.')
    learner = base.QueryDDQN(exp.spec, exp.actor_saved['state'],
                             base.pilot.base.legacy.core.device, a.seed)
    learner.policy.load_state_dict(saved['policy_state'])
    learner.policy.eval().requires_grad_(False)
    a.pilot_rows = 0
    domains, reference = exp.split('test_ids', base.TEST)
    test_manifest = exp.data_manifest('test_ids', base.TEST)
    main_execution = read(model_run/'Results'/'testp0_budget20'/'LearnedBeta'/'execution.json')
    if (main_execution['TEST_data'] != test_manifest
            or main_execution['budget_fraction'] != .2
            or main_execution['learned_checkpoint_sha256'] != done['sha256']):
        raise ValueError('Full final-method TEST stream/reference does not match.')
    protocol = dict(version=1, dataset=a.dataset, seed=a.seed,
        method='RandomNonWindow', learned_checkpoint_sha256=done['sha256'],
        learned_training_protocol=learned_protocol, TEST_data=test_manifest,
        levels=base.TEST, test_rows=0, budget_fraction=.2,
        query='independent Bernoulli(p=.2) per arrived sample until strict floor(.2*N) cap; no quota filling',
        feedback='immediately after the queried sample prediction; no window',
        prediction='fixed Context -> Frozen; predict before label acquisition and updating',
        reset='initial Context/Adam/memory/feedback counters reset for every level',
        learning='no Q/beta updates; only paid-label Context adaptation',
        scope='deployment query replacement, NOT independently retrained no-RL development',
        target_endpoints='diagnostics on the adaptation stream, including paid rows; not independent target gain',
        implementation_sha256=base.pilot.digest(Path(__file__)))
    pf = a.output/'protocol.json'
    if pf.exists():
        if read(pf) != protocol:
            raise ValueError('Existing protocol differs; preserve output and choose fresh --output.')
    elif a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Unmatched partial output preserved; choose fresh --output.')
    else:
        a.output.mkdir(parents=True, exist_ok=True)
        base.pilot.write_json(pf, protocol)
    frozen = (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
              base.weight_hash(learner.policy.state_dict()))
    counters = (learner.steps, learner.updates)
    rows = []
    print(f'DATASET={a.dataset} POLICY=RandomNonWindow | FULL TEST | BUDGET=20% | '
          'IMMEDIATE FEEDBACK | SAME FINAL INITIALIZATION | NO TRAINING', flush=True)
    for level in base.TEST:
        root = a.output/'Results'/f'level_{level}'
        mf, tf, cf = (root/name for name in ('metrics.json', 'trace.csv', 'completed.json'))
        if cf.exists():
            complete = read(cf)
            if (complete['protocol_sha256'] != base.pilot.digest(pf)
                    or any(base.pilot.digest(path) != complete['sha256'][path.name] for path in (mf, tf))):
                raise ValueError('Completed random-control case changed.')
            result = read(mf)
        elif mf.exists() or tf.exists():
            raise ValueError('Incomplete case preserved; use a fresh --output: '+str(root))
        else:
            result, trace = base.run_episode(exp, learner, domains[level], reference,
                                             a.seed, training=False, mode='Random')
            root.mkdir(parents=True, exist_ok=True)
            base.pilot.write_json(mf, result)
            base.pilot.write_csv(tf, trace)
            base.pilot.write_json(cf, dict(protocol_sha256=base.pilot.digest(pf),
                sha256={path.name: base.pilot.digest(path) for path in (mf, tf)}))
        if (result['samples'] != len(domains[level][1]) or result['td_updates'] != 0
                or result['cap'] != int(.2*result['samples'])
                or not 0 <= result['labels'] <= result['cap']):
            raise RuntimeError('Full TEST, cap or frozen-Q contract violated.')
        row = dict(dataset=a.dataset, method='RandomNonWindow', level=level,
            **result['online'], samples=result['samples'], labels=result['labels'],
            cap=result['cap'], labels_percent=result['labels_percent'],
            forgetting=result['original_domain_forgetting']['AC'],
            milliseconds_per_row=result['milliseconds_per_row'])
        rows.append(row)
        print(f'DATASET={a.dataset} POLICY=RandomNonWindow LEVEL={level} '
              f'AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
              f'N={row["samples"]} LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
              f'ORIGINAL_FORGETTING={row["forgetting"]:+.2f} pp', flush=True)
    if (frozen != (base.weight_hash(exp.cnn.state_dict()), base.weight_hash(exp.template.state_dict()),
                   base.weight_hash(learner.policy.state_dict()))
            or counters != (learner.steps, learner.updates)):
        raise RuntimeError('Source model or frozen-Q state changed.')
    for name, digest in sp['source_models'].items():
        if base.pilot.digest(a.models/name) != digest:
            raise RuntimeError('Source model file changed: '+name)
    if base.pilot.digest(model_file) != done['sha256']:
        raise RuntimeError('Learned-beta source file changed.')
    if (base.pilot.digest(a.teacher/'Models'/'minimax_query.pth') != exp.teacher_hash
            or base.pilot.digest(a.actor_run/'Models'/'selector.pth') != exp.actor_sha):
        raise RuntimeError('Teacher or Actor checkpoint file changed.')
    base.pilot.write_csv(a.output/'Results'/'summary_by_level.csv', rows)
    mean = dict(dataset=a.dataset, method='RandomNonWindow', levels=len(rows))
    for key in ('AC', 'PR', 'RC', 'F1', 'labels_percent', 'forgetting'):
        mean[key] = float(np.mean([row[key] for row in rows]))
    base.pilot.write_csv(a.output/'Results'/'summary_mean.csv', [mean])
    base.pilot.write_json(a.output/'Results'/'completed.json', dict(
        protocol_sha256=base.pilot.digest(pf), levels=base.TEST,
        summary_sha256=base.pilot.digest(a.output/'Results'/'summary_mean.csv')))
    print('MEAN '+json.dumps(mean, sort_keys=True), flush=True)
    print('OUTPUT: '+str(a.output), flush=True)


if __name__ == '__main__':
    main()
