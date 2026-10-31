"""Short q75-aligned DDQN continuation; fixed reward/beta and independent target TEST."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

import amcal_two_action_paper_gain as base
import amcal_meta_beta_ddqn as source_recipe
from enhancements.q75_masked_ddqn import Q75MaskedDDQN
from enhancements.q75_query_guide import run_guided_case
from enhancements.independent_target_holdout import split_targets,purge_replay

METHODS=['AlignedQ75DDQN','InferenceOnlyQ75','LearnedBetaDDQN','Q75Only']


def new_agent(experiment,policy_state,masked):
    a=experiment.a
    common=(experiment.spec,experiment.actor_saved['state'],base.pilot.base.legacy.core.device,a.seed)
    learner=(Q75MaskedDDQN(*common,threshold=experiment.threshold,classes=len(a.classes))
        if masked else base.QueryDDQN(*common))
    learner.policy.load_state_dict(policy_state)
    learner.target.load_state_dict(learner.policy.state_dict())
    return learner


def train_aligned(experiment,learner,protocol):
    a=experiment.a; root=a.output; candidate=root/'AlignedTraining'
    trainer=candidate/'Models'/'trainer_state.pth'
    model=candidate/'Models'/'two_action_ddqn.pth'
    done=candidate/'training_complete.json'; intent=candidate/'initialization.json'
    if not done.exists() and not trainer.exists():
        if intent.exists():
            if json.loads(intent.read_text(encoding='utf-8'))!=protocol: raise ValueError('Initialization protocol changed.')
        elif candidate.exists() and any(candidate.iterdir()): raise ValueError('Unmatched partial aligned run preserved.')
        else: base.pilot.write_json(intent,protocol)
        base.pilot.write_json(candidate/'protocol.json',protocol)
        # Include the inference-only hybrid as epoch0; further Q training must win on VAL.
        rows=base.evaluate(experiment,learner,'val_ids',base.TRAIN,candidate/'VAL_epoch_00',controls=False)
        score=float(np.mean([r['AC']-max(0.,r['forgetting']) for r in rows]))
        labels=float(np.mean([r['labels_percent'] for r in rows]))
        history=[dict(epoch=0,training=[],validation=rows,selection_score=score,
            validation_labels_percent=labels,role='source policy under q75 mask; no continuation updates')]
        base.parent.save_model(model,dict(protocol=protocol,selected_epoch=0,
            policy_state=deepcopy(learner.policy.state_dict())))
        base.parent.save_model(trainer,dict(protocol=protocol,epoch=0,learner=learner.snapshot(),
            history=history,best=[score,-labels]))
        print(f'VAL BASELINE EPOCH=0 SCORE={score:.2f} LABELS={labels:.2f}% | keep unless continuation improves VAL',flush=True)
    a.output=candidate
    try: base.train(experiment,learner,protocol)
    finally: a.output=root
    selected=json.loads(done.read_text(encoding='utf-8'))
    print(f'ALIGNED TRAINING SELECTED_EPOCH={selected["selected_epoch"]} | 0 means original hybrid retained by VAL',flush=True)


def run_case(experiment,learner,stream,audit,source_reference,method):
    a=experiment.a
    if method=='AlignedQ75DDQN':
        learner.decision_log=[]
        try:
            result,trace=run_guided_case(experiment,learner,stream,audit,source_reference,a.seed,'LearnedBetaDDQN',5)
            if len(trace)!=len(learner.decision_log): raise RuntimeError('Aligned decision/trace mismatch.')
            for row,decision in zip(trace,learner.decision_log): row.update(decision)
            result['query_rule']='AlignedQ75DDQN'
            return result,trace
        finally: learner.decision_log=None
    translated={'InferenceOnlyQ75':'Q75GuidedDDQN','LearnedBetaDDQN':'LearnedBetaDDQN','Q75Only':'Q75Only'}
    return run_guided_case(experiment,learner,stream,audit,source_reference,a.seed,translated[method],5)


def evaluate(experiment,source,protocol):
    a=experiment.a; root=a.output
    selected_path=root/'AlignedTraining'/'Models'/'two_action_ddqn.pth'
    done=json.loads((root/'AlignedTraining'/'training_complete.json').read_text(encoding='utf-8'))
    selected=torch.load(selected_path,map_location='cpu',weights_only=False)
    if done['protocol']!=protocol or selected['protocol']!=protocol or done['checkpoint_sha256']!=base.pilot.digest(selected_path):
        raise ValueError('Aligned selected checkpoint provenance mismatch.')
    a.pilot_rows=0
    domains,reference=experiment.split('test_ids',base.TEST)
    streams,audits,split_manifest=split_targets(domains,a.seed)
    experiment.anchors,replay_manifest=purge_replay(experiment.anchors,audits)
    previous=root.parent/f'target_holdout20_learned_vs_uncertainty_seed{a.seed}_w5_full_v1'
    if (previous/'target_split_manifest.json').exists():
        if json.loads((previous/'target_split_manifest.json').read_text(encoding='utf-8'))!=split_manifest:
            raise ValueError('Independent target split differs from prior experiment.')
    if (previous/'protocol.json').exists():
        old=json.loads((previous/'protocol.json').read_text(encoding='utf-8'))
        if old['source_checkpoint_sha256']!=protocol['source_checkpoint_sha256'] or old['source_replay_purge']!=replay_manifest:
            raise ValueError('Original query checkpoint or replay differs from prior holdout experiment.')
    execution=dict(training_protocol=protocol,selected_epoch=selected['selected_epoch'],
        selected_checkpoint_sha256=done['checkpoint_sha256'],target_split=split_manifest,
        source_replay_purge=replay_manifest,methods=METHODS,feedback='immediate for all methods',
        test='all eligible target rows allocated; group-disjoint20% holdout,80% stream; no TEST truncation',
        budget='20% of adaptation stream; nominal16% of all eligible target rows')
    ef=root/'evaluation_protocol.json'
    if ef.exists():
        if json.loads(ef.read_text(encoding='utf-8'))!=execution: raise ValueError('Aligned TEST protocol changed.')
    else: base.pilot.write_json(ef,execution)
    base.pilot.write_json(root/'target_split_manifest.json',split_manifest)
    all_rows=[]; summaries=[]
    for method in METHODS:
        learner=new_agent(experiment,selected['policy_state'] if method=='AlignedQ75DDQN' else source['policy_state'],
            method=='AlignedQ75DDQN')
        learner.policy.eval().requires_grad_(False)
        initial=base.weight_hash(learner.policy.state_dict()); rows=[]
        for level in base.TEST:
            directory=root/'Results'/method/f'level_{level}'
            case_file=directory/'execution.json'; mf=directory/'metrics.json'; tf=directory/'trace.csv'; cf=directory/'completed.json'
            case=dict(execution=execution,method=method,level=level)
            if case_file.exists():
                if json.loads(case_file.read_text(encoding='utf-8'))!=case: raise ValueError('Aligned TEST case changed.')
            elif directory.exists() and any(directory.iterdir()): raise ValueError('Unmatched partial TEST case preserved.')
            else: base.pilot.write_json(case_file,case)
            n=len(streams[level][1]); h=len(audits[level][1])
            print(f'START DATASET={a.dataset} METHOD={method} LEVEL={level} STREAM={n} HOLDOUT={h} '
                f'BETA={a.beta:.6f} Q75={experiment.threshold:.8f} SELECTED_EPOCH={selected["selected_epoch"]}',flush=True)
            if cf.exists():
                complete=json.loads(cf.read_text(encoding='utf-8'))
                if complete['execution_sha256']!=base.pilot.digest(case_file) or any(not f.is_file() or
                        base.pilot.digest(f)!=complete['sha256'][f.name] for f in (mf,tf)):
                    raise ValueError('Completed aligned TEST metrics/trace changed.')
                result=json.loads(mf.read_text(encoding='utf-8'))
            else:
                base.pilot.base.legacy.core.set_seed(a.seed)
                result,trace=run_case(experiment,learner,streams[level],audits[level],reference,method)
                if result['td_updates']!=0 or result['cap']!=int(.2*n) or result['labels']>result['cap']:
                    raise RuntimeError('Aligned TEST budget or frozen-Q violation.')
                if set(map(int,audits[level][2])) & {int(r['source_row']) for r in trace}:
                    raise RuntimeError('Target holdout row leaked into adaptation.')
                base.pilot.write_json(mf,result); base.pilot.write_csv(tf,trace)
                base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(case_file),
                    sha256={f.name:base.pilot.digest(f) for f in (mf,tf)}))
            row=dict(dataset=a.dataset,beta=a.beta,selected_epoch=selected['selected_epoch'],**base.result_row(method,level,result))
            row.update(target_initial_AC=result['target_before']['AC'],
                target_holdout_gain_AC=result['target_holdout_gain']['AC'],
                target_holdout_rows=h,labels_percent_of_all_target_rows=result['labels_percent_of_all_target_rows'],
                budget_used_percent=100*result['labels']/max(1,result['cap']))
            for key in ('PR','RC','F1'):
                row['target_initial_'+key]=result['target_before'][key]
                row['target_final_'+key]=result['target_after'][key]
                row['target_holdout_gain_'+key]=result['target_holdout_gain'][key]
            rows.append(row); all_rows.append(row)
            base.pilot.write_csv(root/'Results'/'summary_by_level.csv',all_rows)
            print(f'DONE DATASET={a.dataset} METHOD={method} LEVEL={level} ONLINE_AC={row["AC"]:.2f}% '
                f'ONLINE_F1={row["F1"]:.2f}% HOLDOUT_AC={row["target_final_AC"]:.2f}% '
                f'HOLDOUT_F1={row["target_final_F1"]:.2f}% LABELS_STREAM={row["labels_percent"]:.2f}% '
                f'LABELS_ALL={row["labels_percent_of_all_target_rows"]:.2f}% CAP_USED={row["budget_used_percent"]:.2f}% '
                f'FORGET={row["forgetting"]:+.2f} pp',flush=True)
        if initial!=base.weight_hash(learner.policy.state_dict()) or learner.steps!=0 or learner.updates!=0:
            raise RuntimeError('TEST changed the Q policy.')
        report=dict(dataset=a.dataset,method=method,beta=a.beta,selected_epoch=selected['selected_epoch'],levels=len(rows))
        for key in ('AC','PR','RC','F1','target_initial_AC','target_final_AC','target_final_PR','target_final_RC','target_final_F1',
                'labels_percent','labels_percent_of_all_target_rows','budget_used_percent','source_final_AC','forgetting',
                'target_initial_PR','target_initial_RC','target_initial_F1','target_holdout_gain_AC',
                'target_holdout_gain_PR','target_holdout_gain_RC','target_holdout_gain_F1'):
            report[key]=float(np.mean([r[key] for r in rows]))
        summaries.append(report); base.pilot.write_csv(root/'Results'/'summary_mean.csv',summaries)
        print('MEAN '+json.dumps(report,sort_keys=True),flush=True)
    base.pilot.write_json(root/'Results'/'completed.json',dict(evaluation_sha256=base.pilot.digest(ef),cases=len(all_rows),
        summary_sha256={name:base.pilot.digest(root/'Results'/name) for name in ('summary_mean.csv','summary_by_level.csv')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['IAT','PS'],required=True)
    parser.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    a=parser.parse_args()
    if a.epochs<1: parser.error('Use positive continuation epochs.')
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'; runs=root/'protocol_runs'
    source_run=runs/f'meta_beta_ddqn_seed{a.seed}_p300_e3_v1'
    source_path=source_run/'Models'/'learned_beta.pth'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    complete=json.loads((source_run/'training_complete.json').read_text(encoding='utf-8'))
    source_protocol=source['protocol']; sp=source_protocol['source_protocol']['protocol']
    if (complete['protocol']!=source_protocol or complete['sha256']!=base.pilot.digest(source_path)
            or source_protocol['dataset']!=a.dataset or source_protocol['seed']!=a.seed):
        raise ValueError('Source learned checkpoint mismatch.')
    for implementation in (sp['implementation_sha256'],source_protocol['implementation_sha256']):
        for name,sha in implementation.items():
            if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Source code changed: '+name)
    a.beta=float(source['beta']); a.budget_fraction=.2; a.pilot_rows=sp['pilot_rows']
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher=runs/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run=runs/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output=runs/f'q75_aligned_ddqn_seed{a.seed}_e{a.epochs}_full_v1'
    experiment=base.PaperGainExperiment(a)
    if (experiment.teacher_hash!=sp['healthy_teacher_sha256'] or experiment.actor_sha!=sp['actor_checkpoint_sha256']
            or experiment.healthy['model_hashes']!=sp['source_models']
            or experiment.healthy['training_domain_sha256']!=sp['train_files']):
        raise ValueError('Source CNN/Context/Actor lineage changed.')
    if (experiment.data_manifest('train_ids',base.TRAIN)!=sp['train_data'] or
            experiment.data_manifest('val_ids',base.TRAIN)!=sp['validation_data']):
        raise ValueError('Source TRAIN/VAL streams changed.')
    inner,retention,anchors,manifest=base.prepare_train_probes(experiment,base.pilot,24,32)
    experiment.train_anchors=anchors
    _,meta_probes,anchors,meta_manifest=source_recipe.reserve_meta(experiment,inner)
    if (manifest!=source_protocol['inner_probe_manifest'] or meta_manifest!=source_protocol['meta_TRAIN_manifest']
            or [base.domain_hash(d) for d in inner]!=source_protocol['inner_probe_sha256']
            or base.domain_hash(anchors)!=source_protocol['meta_replay_sha256']):
        raise ValueError('Original independent TRAIN probes/replay reservation changed.')
    experiment.train_anchors=anchors
    experiment.reward_probe_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in [*inner,*meta_probes.values()]))
    experiment.set_gain_reward(inner,retention,a.beta)
    initial=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()))
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,epochs=a.epochs,
        source_checkpoint_sha256=complete['sha256'],source_protocol=source_protocol,
        beta=a.beta,beta_update='none; selected learned beta fixed',threshold=float(experiment.threshold),
        threshold_source='saved source Actor TRAIN q75; no VAL/TEST threshold calibration',
        behavior='budget AND q75 mask before greedy or epsilon action selection',
        bootstrap='same mask applied to successor legality before online argmax and target gather',
        reward='unchanged paper reward plus frozen learned beta times independent same-domain TRAIN gain; skip zero',
        replay='fresh Q replay/Adam at continuation start; current legal transitions only; original TRAIN anchors/probes reused',
        initialization='source learned Q transferred; target network synchronized; CNN and initial Context unchanged',
        validation='epoch0 inference-only hybrid eligible; max mean online AC minus positive source forgetting, fewer labels tie-break',
        selection='TRAIN/VAL only; no TEST data read in Train action',
        train_rows_per_level=sp['pilot_rows'],budget_fraction=.2,context_lr=a.context_lr,replay_weight=a.replay_weight,
        source_TRAIN_probe_manifest=manifest,source_meta_reservation=meta_manifest,
        train_replay_sha256=base.domain_hash(anchors),
        implementation_sha256={name:base.pilot.digest(Path(__file__).parent/name) for name in
            ('amcal_q75_aligned_ddqn.py','enhancements/q75_masked_ddqn.py','enhancements/q75_query_guide.py',
             'enhancements/independent_target_holdout.py')},
        theory='existing competitive Context update and closed-loop DDQN; added eligibility constraint, no new saddle-equivalence claim')
    pf=a.output/'protocol.json'
    if pf.exists():
        if json.loads(pf.read_text(encoding='utf-8'))!=protocol: raise ValueError('Aligned training protocol changed.')
    elif a.output.exists() and any(a.output.iterdir()): raise ValueError('Unmatched partial root preserved.')
    else:
        if a.action=='Evaluate': raise ValueError('Complete aligned TRAIN/VAL selection first.')
        base.pilot.write_json(pf,protocol)
    if a.action in ('Train','All'):
        base.pilot.base.legacy.core.set_seed(a.seed)
        train_aligned(experiment,new_agent(experiment,source['policy_state'],True),protocol)
    if a.action in ('Evaluate','All'): evaluate(experiment,source,protocol)
    if initial!=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict())):
        raise RuntimeError('Initial CNN/Context changed.')
    if (base.pilot.digest(source_path)!=complete['sha256'] or
            base.pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth')!=experiment.actor_sha):
        raise RuntimeError('An original checkpoint file changed.')
    print('Q75 ALIGNED OUTPUT: '+str(a.output),flush=True)


if __name__=='__main__': main()

