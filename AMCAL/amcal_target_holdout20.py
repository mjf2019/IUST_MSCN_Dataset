"""Learned-beta DDQN versus window uncertainty with a group-disjoint 20% target holdout."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import amcal_two_action_paper_gain as base
from enhancements.independent_target_holdout import split_targets,purge_replay,run_holdout_case

METHODS=['LearnedBetaDDQN','UncertaintyWindow']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    a=p.parse_args(); a.window_size=5; a.budget_fraction=.2
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'; runs=root/'protocol_runs'
    model_run=runs/f'meta_beta_ddqn_seed{a.seed}_p300_e3_v1'
    model_file=model_run/'Models'/'learned_beta.pth'
    saved=torch.load(model_file,map_location='cpu',weights_only=False)
    complete=json.loads((model_run/'training_complete.json').read_text(encoding='utf-8'))
    learned_protocol=saved['protocol']
    if (complete['protocol']!=learned_protocol or complete['sha256']!=base.pilot.digest(model_file)
            or learned_protocol['dataset']!=a.dataset or learned_protocol['seed']!=a.seed):
        raise ValueError('Learned-beta checkpoint/completion provenance mismatch.')
    sp=learned_protocol['source_protocol']['protocol']
    for implementation in (sp['implementation_sha256'],learned_protocol['implementation_sha256']):
        for name,sha in implementation.items():
            if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Source implementation changed: '+name)
    a.beta=float(saved['beta']); a.pilot_rows=sp['pilot_rows']
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher=runs/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run=runs/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output=runs/f'target_holdout20_learned_vs_uncertainty_seed{a.seed}_w5_full_v1'
    experiment=base.PaperGainExperiment(a)
    if (experiment.teacher_hash!=sp['healthy_teacher_sha256'] or experiment.actor_sha!=sp['actor_checkpoint_sha256']
            or experiment.healthy['model_hashes']!=sp['source_models']
            or experiment.healthy['training_domain_sha256']!=sp['train_files']):
        raise ValueError('Learned-beta CNN/Context/Actor lineage differs.')
    if (experiment.data_manifest('train_ids',base.TRAIN)!=sp['train_data'] or
            experiment.data_manifest('val_ids',base.TRAIN)!=sp['validation_data']):
        raise ValueError('Source TRAIN/VAL streams differ.')
    learner=base.QueryDDQN(experiment.spec,experiment.actor_saved['state'],base.pilot.base.legacy.core.device,a.seed)
    learner.policy.load_state_dict(saved['policy_state']); learner.policy.eval().requires_grad_(False)
    source_hashes=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()),
        base.weight_hash(learner.policy.state_dict()))
    a.pilot_rows=0
    domains,source_reference=experiment.split('test_ids',base.TEST)
    official_manifest=experiment.data_manifest('test_ids',base.TEST)
    streams,audits,split_manifest=split_targets(domains,a.seed)
    experiment.anchors,replay_manifest=purge_replay(experiment.anchors,audits)
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,beta=a.beta,source_checkpoint_sha256=complete['sha256'],
        learned_training_protocol=learned_protocol,official_TEST=official_manifest,target_split=split_manifest,
        source_replay_purge=replay_manifest,methods=METHODS,window_size=5,threshold=None,
        budget_fraction_of_adaptation_stream=.2,nominal_budget_fraction_of_all_target_rows=.16,
        test_rows_per_level='all eligible rows allocated to stream or holdout; no pilot sampling',
        final_target='same-level independent holdout before/after adaptation; never an update/replay target',
        online='pre-feedback predictions on adaptation stream only; independent holdout is not part of stream',
        source_forgetting='original source reference accuracy before minus after adaptation',
        reset='initial Context/Adam/feedback memory/cap reset per method and intensity; CNN/Q/beta frozen',
        delay='LearnedBetaDDQN immediate; UncertaintyWindow releases feedback at window close; retain this distinction',
        current_independence='holdout IDs/features excluded from current streams and source replay across all levels',
        historical_pretraining='existing models reused; this split cannot retroactively remove historical feature duplicates',
        implementation_sha256={name:base.pilot.digest(Path(__file__).parent/name) for name in
            ('amcal_target_holdout20.py','enhancements/independent_target_holdout.py','enhancements/causal_window_query.py')})
    pf=a.output/'protocol.json'
    if pf.exists():
        if json.loads(pf.read_text(encoding='utf-8'))!=protocol: raise ValueError('Independent target TEST protocol changed.')
    elif a.output.exists() and any(a.output.iterdir()): raise ValueError('Unmatched partial holdout run preserved.')
    else: base.pilot.write_json(pf,protocol)
    base.pilot.write_json(a.output/'target_split_manifest.json',split_manifest)
    for level in base.TEST:
        n=len(domains[level][1]); s=len(streams[level][1]); h=len(audits[level][1])
        print(f'SPLIT DATASET={a.dataset} LEVEL={level} TOTAL={n} STREAM={s} '
            f'INDEPENDENT_TEST={h} ({100*h/n:.2f}%) FEEDBACK_CAP={int(.2*s)} '
            f'({100*int(.2*s)/n:.2f}% of total)',flush=True)
    all_rows=[]; means=[]
    for method in METHODS:
        rows=[]
        for level in base.TEST:
            directory=a.output/'Results'/method/f'level_{level}'
            ef=directory/'execution.json'; mf=directory/'metrics.json'; tf=directory/'trace.csv'; cf=directory/'completed.json'
            execution=dict(protocol=protocol,method=method,level=level,
                stream_sha256=base.domain_hash(streams[level]),holdout_sha256=base.domain_hash(audits[level]))
            if ef.exists():
                if json.loads(ef.read_text(encoding='utf-8'))!=execution: raise ValueError('Independent TEST case changed.')
            elif directory.exists() and any(directory.iterdir()): raise ValueError('Unmatched partial TEST case preserved.')
            else: base.pilot.write_json(ef,execution)
            print(f'START DATASET={a.dataset} METHOD={method} LEVEL={level} BETA={a.beta:.6f} '
                'STREAM_BUDGET=20% | independent target TEST never queried',flush=True)
            if cf.exists():
                done=json.loads(cf.read_text(encoding='utf-8'))
                if done['execution_sha256']!=base.pilot.digest(ef) or any(not f.is_file() or
                        base.pilot.digest(f)!=done['sha256'][f.name] for f in (mf,tf)):
                    raise ValueError('Completed independent target case changed.')
                result=json.loads(mf.read_text(encoding='utf-8'))
            else:
                base.pilot.base.legacy.core.set_seed(a.seed)
                result,trace=run_holdout_case(experiment,learner,streams[level],audits[level],source_reference,
                    a.seed,method,5)
                cap=int(.2*len(streams[level][1]))
                if result['td_updates']!=0 or result['cap']!=cap or result['labels']>cap:
                    raise RuntimeError('Independent TEST budget/frozen-Q contract failed.')
                # A direct ID audit includes unqueried stream rows as well as paid rows.
                if set(map(int,audits[level][2])) & {int(r['source_row']) for r in trace}:
                    raise RuntimeError('Independent holdout row appeared in adaptation trace.')
                base.pilot.write_json(mf,result); base.pilot.write_csv(tf,trace)
                base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(ef),
                    sha256={f.name:base.pilot.digest(f) for f in (mf,tf)}))
            row=dict(dataset=a.dataset,beta=a.beta,**base.result_row(method,level,result))
            row.update(target_total_rows=result['target_total_rows'],target_holdout_rows=result['target_holdout_rows'],
                target_holdout_percent=100*result['target_holdout_rows']/result['target_total_rows'],
                target_initial_AC=result['target_before']['AC'],
                target_holdout_gain_AC=result['target_holdout_gain']['AC'],
                labels_percent_of_all_target_rows=result['labels_percent_of_all_target_rows'],
                budget_used_percent=100*result['labels']/max(1,result['cap']))
            for key in ('PR','RC','F1'):
                row['target_initial_'+key]=result['target_before'][key]
                row['target_final_'+key]=result['target_after'][key]
                row['target_holdout_gain_'+key]=result['target_holdout_gain'][key]
            rows.append(row); all_rows.append(row)
            base.pilot.write_csv(a.output/'Results'/'summary_by_level.csv',all_rows)
            print(f'DONE DATASET={a.dataset} METHOD={method} LEVEL={level} '
                f'ONLINE_AC={row["AC"]:.2f}% ONLINE_F1={row["F1"]:.2f}% '
                f'INDEPENDENT_TEST_AC={row["target_initial_AC"]:.2f}% -> {row["target_final_AC"]:.2f}% '
                f'INDEPENDENT_TEST_F1={row["target_final_F1"]:.2f}% '
                f'LABELS_STREAM={row["labels_percent"]:.2f}% LABELS_ALL={row["labels_percent_of_all_target_rows"]:.2f}% '
                f'CAP_USED={row["budget_used_percent"]:.2f}% SOURCE_FORGET={row["forgetting"]:+.2f} pp',flush=True)
        mean=dict(dataset=a.dataset,method=method,beta=a.beta,levels=len(rows))
        for key in ('AC','PR','RC','F1','labels_percent','labels_percent_of_all_target_rows','budget_used_percent',
                'forgetting','source_final_AC','target_initial_AC','target_final_AC','target_initial_PR','target_final_PR',
                'target_initial_RC','target_final_RC','target_initial_F1','target_final_F1',
                'target_holdout_gain_AC','target_holdout_gain_PR','target_holdout_gain_RC','target_holdout_gain_F1'):
            mean[key]=float(np.mean([r[key] for r in rows]))
        means.append(mean); base.pilot.write_csv(a.output/'Results'/'summary_mean.csv',means)
        print('MEAN INDEPENDENT TEST '+json.dumps(mean,sort_keys=True),flush=True)
    current=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()),
        base.weight_hash(learner.policy.state_dict()))
    if current!=source_hashes or learner.updates!=0 or learner.steps!=0:
        raise RuntimeError('Evaluation changed initial source models or Q counters.')
    if (base.pilot.digest(model_file)!=complete['sha256'] or
            base.pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth')!=experiment.actor_sha):
        raise RuntimeError('An original checkpoint file changed.')
    base.pilot.write_json(a.output/'Results'/'completed.json',dict(protocol_sha256=base.pilot.digest(pf),cases=len(all_rows),
        summary_sha256={name:base.pilot.digest(a.output/'Results'/name) for name in ('summary_mean.csv','summary_by_level.csv')}))
    print('INDEPENDENT TARGET TEST OUTPUT: '+str(a.output),flush=True)


if __name__=='__main__': main()

