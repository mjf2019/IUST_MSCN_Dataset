"""Evaluate existing post-selector bundles at multiple budgets; no offline training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import amcal_post_selector_cnn13 as saved_runner

base = saved_runner.base

def compare_case(experiment, learner, variant, level, domain, reference, common_reference,
                 directory, execution):
    a = experiment.a; a.budget_fraction = float(execution['budget_fraction'])
    if not 0 < a.budget_fraction <= 1: raise ValueError('Invalid TEST budget.')
    tag = f'DATASET={a.dataset} BETA={a.beta:g} VARIANT={variant} LEVEL={level} BUDGET={100*a.budget_fraction:g}%'
    print('\nSTART '+tag+' | RESET=Context/Adam/memory', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    ef, cf, mf, tf = directory/'execution.json', directory/'completed.json', directory/'metrics.json', directory/'trace.csv'
    if ef.exists():
        if json.loads(ef.read_text(encoding='utf-8')) != execution: raise ValueError('CNN comparison case changed.')
    elif any(directory.iterdir()): raise ValueError('Unmatched partial CNN comparison case preserved.')
    else: base.pilot.write_json(ef, execution)
    reused = cf.exists()
    if reused:
        done = json.loads(cf.read_text(encoding='utf-8'))
        if done.get('execution_sha256') != base.pilot.digest(ef) or any(not p.is_file() or
                base.pilot.digest(p) != done.get('sha256',{}).get(p.name) for p in (mf,tf)):
            raise ValueError('Completed CNN comparison metrics/trace changed.')
        result = json.loads(mf.read_text(encoding='utf-8'))
    else:
        before = base.weight_hash(learner.policy.state_dict()); updates, steps = learner.updates, learner.steps
        result, trace = base.run_episode(experiment, learner, domain, reference, a.seed, training=False, mode='DDQN')
        if before != base.weight_hash(learner.policy.state_dict()) or updates != learner.updates or steps != learner.steps:
            raise RuntimeError('Comparison TEST updated the DDQN.')
        if result['td_updates'] != 0 or result['cap'] != int(a.budget_fraction*len(domain[1])) or result['labels'] > result['cap']:
            raise RuntimeError('CNN comparison violated frozen-query or budget accounting.')
        result['cnn_finetune_source_drop'] = {k:common_reference[k]-result['source_before'][k] for k in common_reference}
        result['total_source_drop_from_current_bundle'] = {k:common_reference[k]-result['source_after'][k] for k in common_reference}
        base.pilot.write_csv(tf,trace); base.pilot.write_json(mf,result)
        base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(ef),sha256={p.name:base.pilot.digest(p) for p in (mf,tf)}))
    row = dict(variant=variant,beta=a.beta,**base.result_row(variant,level,result),
        cnn_finetune_source_drop=result['cnn_finetune_source_drop']['AC'],
        total_source_drop=result['total_source_drop_from_current_bundle']['AC'])
    for key in ('PR','RC','F1'):
        row['target_final_'+key] = result['target_after'][key]
        row['total_source_drop_'+key] = result['total_source_drop_from_current_bundle'][key]
    print(('REUSED ' if reused else 'DONE ')+tag+
        f' | AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% LABELS={row["labels_percent"]:.2f}% '
        f'FINAL_AC={row["target_final_AC"]:.2f}% | CNN_FT_SOURCE_DROP={row["cnn_finetune_source_drop"]:+.2f} pp '
        f'ONLINE_SOURCE_DROP={row["forgetting"]:+.2f} pp TOTAL_SOURCE_DROP={row["total_source_drop"]:+.2f} pp',flush=True)
    return row



def budget_directory(rows, budget):
    return f'testp{rows}_budget{100*budget:g}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT','PS'], required=True)
    p.add_argument('--beta', type=float)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--budgets', type=float, nargs='+', default=[.3,.4,.5])
    p.add_argument('--variants', choices=saved_runner.VARIANTS, nargs='+', default=['Current'])
    p.add_argument('--test-rows', type=int, default=300)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='cuda')
    p.add_argument('--bundle-run', type=Path)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--actor-run', type=Path)
    a = p.parse_args()
    if a.beta is None: a.beta = 26. if a.dataset == 'IAT' else 55.
    if (not np.isfinite(a.beta) or a.beta < 0 or
            a.test_rows < 0 or 0 < a.test_rows < 32 or
            any(not np.isfinite(b) or not 0 < b <= 1 for b in a.budgets)):
        p.error('Use nonnegative beta, row limit zero or at least 32 and budgets in (0,1].')
    a.budgets = list(dict.fromkeys(a.budgets))
    a.variants = list(dict.fromkeys(a.variants))
    root = Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    bundle_run = a.bundle_run or root/'protocol_runs'/f'post_selector_cnn13_seed{a.seed}_{base.beta_name(a.beta)}_trainp300_v1'
    protocol_file = bundle_run/'protocol.json'
    protocol = json.loads(protocol_file.read_text(encoding='utf-8'))
    complete = json.loads((bundle_run/'training_complete.json').read_text(encoding='utf-8'))
    if (complete['protocol'] != protocol or protocol['dataset'] != a.dataset or
            protocol['seed'] != a.seed or protocol['beta'] != a.beta):
        raise ValueError('Existing bundle dataset/seed/beta/protocol mismatch.')
    if base.pilot.digest(Path(saved_runner.__file__)) != protocol['implementation_sha256']:
        raise ValueError('Original bundle implementation changed; preserve the source driver.')
    source = protocol['source_query_protocol']['protocol']
    for name, expected in source['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name) != expected:
            raise ValueError('Source selector implementation changed: '+name)
    a.pilot_rows = source['pilot_rows']; a.budget_fraction = .2
    a.burst_dir = root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher = a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run = a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output = bundle_run
    experiment = base.PaperGainExperiment(a)
    if (experiment.teacher_hash != protocol['healthy_teacher_sha256'] or
            experiment.actor_sha != protocol['actor_checkpoint_sha256'] or
            experiment.healthy['model_hashes'] != source['source_models'] or
            experiment.healthy['training_domain_sha256'] != source['train_files']):
        raise ValueError('CNN/Context/selector lineage differs from the saved comparison.')
    if (base.weight_hash(experiment.cnn.state_dict()) != protocol['initial_CNN_sha256'] or
            base.weight_hash(experiment.template.state_dict()) != protocol['unchanged_context_sha256']):
        raise ValueError('Original CNN/Context changed.')
    paths = {'Current':bundle_run/'Models'/'current_bundle.pth',
             'CNN13':bundle_run/'Models'/'cnn13_bundle.pth'}
    requested = {variant:paths[variant] for variant in a.variants}
    for variant, path in requested.items():
        if base.pilot.digest(path) != complete['bundle_sha256'][variant]:
            raise ValueError('Saved bundle changed: '+variant)
    learner = base.QueryDDQN(experiment.spec, experiment.actor_saved['state'],
        base.pilot.base.legacy.core.device, a.seed)
    a.pilot_rows = a.test_rows
    domains, reference = experiment.split('test_ids', saved_runner.LEVELS)
    manifest = experiment.data_manifest('test_ids', saved_runner.LEVELS)
    common = base.endpoint(experiment, experiment.template, reference)
    for budget in a.budgets:
        a.budget_fraction = budget
        results = bundle_run/'Results'/budget_directory(a.test_rows,budget)
        rows = []
        for variant, path in requested.items():
            bundle = saved_runner.load_variant(experiment,learner,path,protocol,
                protocol['unchanged_context_sha256'],protocol['unchanged_query_sha256'])
            if bundle['variant'] != variant: raise ValueError('Bundle labels were swapped.')
            print(f'\nEVALUATION ONLY | DATASET={a.dataset} BETA={a.beta:g} VARIANT={variant} '
                f'BUDGET={100*budget:g}% | reset Context/Adam/memory per level | no retraining',flush=True)
            for level in saved_runner.LEVELS:
                execution = dict(protocol=protocol,budget_fraction=budget,
                    evaluator_sha256=base.pilot.digest(Path(__file__)),
                    bundle_sha256=complete['bundle_sha256'][variant],variant=variant,level=level,
                    TEST_data=manifest,common_source_reference=common)
                row = compare_case(experiment,learner,variant,level,domains[level],reference,common,
                    results/variant/f'level_{level}',execution)
                row['budget_fraction'] = budget
                rows.append(row)
                base.pilot.write_csv(results/'summary_by_level.csv',rows)
        summary = saved_runner.summarize(rows)
        for row in summary: row['budget_fraction'] = budget
        base.pilot.write_csv(results/'summary_mean.csv',summary)
        base.pilot.write_json(results/'completed.json',dict(cases=len(rows),budget_fraction=budget,
            TEST_data=manifest,common_source_reference=common,protocol_sha256=base.pilot.digest(protocol_file),
            evaluator_sha256=base.pilot.digest(Path(__file__)),
            summary_sha256={name:base.pilot.digest(results/name) for name in ('summary_by_level.csv','summary_mean.csv')}))
    for variant, path in requested.items():
        if base.pilot.digest(path) != complete['bundle_sha256'][variant]:
            raise RuntimeError('An original bundle file changed.')
    print('BUDGET EVALUATION OUTPUT: '+str(bundle_run/'Results'),flush=True)


if __name__ == '__main__': main()

