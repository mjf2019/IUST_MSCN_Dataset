"""Fixed-beta sweep on every eligible TEST row, with a 20 percent per-level cap."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

import amcal_two_action_paper_gain as base
from amcal_selected_beta_budgets import compare_case

BETAS = [10.,15.,20.,25.,30.]


def candidate_roots(runs,seed,own,beta):
    name=base.beta_name(beta)
    return [runs/f'two_action_paper_gain_seed{seed}_p300_e3_v1'/'Candidates'/name,
        runs/f'two_action_paper_gain_extra_seed{seed}_p300_e3_v1'/'Candidates'/name,
        own/'Candidates'/name]


def locate_candidate(runs,seed,own,beta):
    for root in candidate_roots(runs,seed,own,beta):
        if (root/'Models'/'two_action_ddqn.pth').is_file() and (root/'training_complete.json').is_file():
            return root
    return None


def validate_candidate(saved,completed,parent,beta):
    protocol=saved['protocol']; source=protocol['protocol']
    if completed['protocol']!=protocol or protocol['beta']!=beta:
        raise ValueError('Candidate beta/protocol mismatch.')
    for key in ('dataset','seed','pilot_rows','epochs','healthy_teacher_sha256','actor_checkpoint_sha256',
                'source_models','train_files','train_data','validation_data','probes','budget_fraction',
                'context_lr','replay_weight','architecture','reward','paper_reward','measurement','DDQN'):
        if source.get(key)!=parent.get(key): raise ValueError('Candidate recipe differs: '+key)
    return protocol


def report_mean(rows):
    report=dict(dataset=rows[0]['dataset'],beta=rows[0]['beta'],levels=len(rows),budget_percent=20,
        total_samples=sum(r['samples'] for r in rows),total_labels=sum(r['labels'] for r in rows))
    for key in ('AC','PR','RC','F1','labels_percent','budget_used_percent','target_final_AC',
                'target_final_PR','target_final_RC','target_final_F1','source_final_AC','forgetting'):
        report[key]=float(np.mean([r[key] for r in rows]))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['IAT','PS'],required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    a=parser.parse_args()
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    runs=root/'protocol_runs'
    parent_root=runs/f'two_action_paper_gain_seed{a.seed}_p300_e3_v1'
    pf=parent_root/'protocol.json'
    parent=json.loads(pf.read_text(encoding='utf-8'))
    if parent['dataset']!=a.dataset or parent['seed']!=a.seed or parent['budget_fraction']!=.2 or parent['pilot_rows']!=300:
        raise ValueError('Expected matching source recipe: TRAIN p300, budget .2, same dataset and seed.')
    for name,sha in parent['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Source implementation changed: '+name)
    a.output=runs/f'fixed_beta_fulltest_seed{a.seed}_v1'; output=a.output
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher=runs/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run=runs/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.pilot_rows=parent['pilot_rows']; a.epochs=parent['epochs']; a.budget_fraction=.2; a.beta=0.
    experiment=base.PaperGainExperiment(a)
    if (experiment.teacher_hash!=parent['healthy_teacher_sha256'] or experiment.actor_sha!=parent['actor_checkpoint_sha256']
            or experiment.healthy['model_hashes']!=parent['source_models']
            or experiment.healthy['training_domain_sha256']!=parent['train_files']):
        raise ValueError('Source CNN/Context/Actor lineage differs.')
    if (experiment.data_manifest('train_ids',base.TRAIN)!=parent['train_data'] or
            experiment.data_manifest('val_ids',base.TRAIN)!=parent['validation_data']):
        raise ValueError('Source TRAIN/VAL streams differ.')
    original_weights=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()))
    sources={}
    for beta in BETAS:
        path=locate_candidate(runs,a.seed,output,beta)
        if path is not None:
            saved=torch.load(path/'Models'/'two_action_ddqn.pth',map_location='cpu',weights_only=False)
            completed=json.loads((path/'training_complete.json').read_text(encoding='utf-8'))
            validate_candidate(saved,completed,parent,beta)
            if base.pilot.digest(path/'Models'/'two_action_ddqn.pth')!=completed['checkpoint_sha256']:
                raise ValueError('Existing candidate checkpoint changed: '+str(path))
            for name,sha in saved['protocol']['protocol']['implementation_sha256'].items():
                if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Candidate implementation changed: '+name)
            sources[beta]=path
        else:
            raise ValueError(f'Missing complete local checkpoint for beta {beta:g}. No training was started.')
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,betas=BETAS,
        test_rows_per_level=0,TEST_levels=base.TEST,budget_fraction=.2,
        parent_protocol_sha256=base.pilot.digest(pf),
        implementation_sha256={name:base.pilot.digest(Path(__file__).parent/name) for name in
            ('amcal_fixed_beta_fulltest.py','amcal_selected_beta_budgets.py')},
        training='none; reuse existing validated candidates only',
        evaluation='all eligible official TEST rows; same audit exclusions, order and seed; reset Context/Adam/memory per beta and level',
        selection='TEST is descriptive for every beta; no automatic TEST-based model selection',
        original_models='source CNN and initial Context unchanged; no CNN fine-tuning')
    own_pf=output/'protocol.json'
    if own_pf.exists():
        if json.loads(own_pf.read_text(encoding='utf-8'))!=protocol: raise ValueError('Full TEST protocol differs.')
    elif output.exists() and any(output.iterdir()): raise ValueError('Unmatched partial output preserved.')
    else: base.pilot.write_json(own_pf,protocol)
    args=deepcopy(a)
    experiment.a=deepcopy(args)
    # Only the TEST row limit changes: zero means every eligible row.
    experiment.a.pilot_rows=0
    domains,reference=experiment.split('test_ids',base.TEST)
    test_manifest=experiment.data_manifest('test_ids',base.TEST)
    common=base.endpoint(experiment,experiment.template,reference)
    means=[]; all_rows=[]
    for beta in BETAS:
        print(f'REUSE DATASET={a.dataset} BETA={beta:g} | existing Q checkpoint | no training',flush=True)
        path=sources[beta]; model_file=path/'Models'/'two_action_ddqn.pth'
        saved=torch.load(model_file,map_location='cpu',weights_only=False)
        completed=json.loads((path/'training_complete.json').read_text(encoding='utf-8'))
        validate_candidate(saved,completed,parent,beta)
        model_hash=base.pilot.digest(model_file)
        if model_hash!=completed['checkpoint_sha256']: raise ValueError('Candidate checkpoint changed.')
        learner=base.QueryDDQN(experiment.spec,experiment.actor_saved['state'],base.pilot.base.legacy.core.device,a.seed)
        learner.policy.load_state_dict(saved['policy_state']); learner.policy.eval().requires_grad_(False)
        experiment.a.beta=beta; rows=[]
        for level in base.TEST:
            n=len(domains[level][1])
            print(f'FULL TEST DATASET={a.dataset} BETA={beta:g} LEVEL={level} ROWS={n} CAP={int(.2*n)} | no TEST subsampling',flush=True)
            execution=dict(protocol=protocol,budget_fraction=.2,beta=beta,level=level,
                query_checkpoint_sha256=model_hash,query_training_protocol=saved['protocol'],
                TEST_data=test_manifest,common_source_reference=common)
            row=compare_case(experiment,learner,'Current',level,domains[level],reference,common,
                output/'Results'/base.beta_name(beta)/f'level_{level}',execution)
            row.update(dataset=a.dataset,budget_percent=20,budget_used_percent=100*row['labels']/max(1,int(.2*n)))
            rows.append(row); all_rows.append(row)
        mean=report_mean(rows); means.append(mean)
        base.pilot.write_csv(output/'Results'/base.beta_name(beta)/'summary_by_level.csv',rows)
        base.pilot.write_csv(output/'Results'/'summary_mean.csv',means)
        base.pilot.write_csv(output/'Results'/'summary_by_level.csv',all_rows)
        print('MEAN FULL TEST '+json.dumps(mean,sort_keys=True),flush=True)
        if base.pilot.digest(model_file)!=model_hash: raise RuntimeError('Source Q checkpoint changed.')
    if original_weights!=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict())):
        raise RuntimeError('Original CNN/Context changed.')
    base.pilot.write_json(output/'Results'/'completed.json',dict(protocol=protocol,TEST_data=test_manifest,
        cases=len(all_rows),summary_sha256={name:base.pilot.digest(output/'Results'/name)
            for name in ('summary_mean.csv','summary_by_level.csv')}))
    print('FULL TEST OUTPUT: '+str(output),flush=True)


if __name__=='__main__': main()

