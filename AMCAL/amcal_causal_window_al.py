"""Full TEST comparison of causal window uncertainty, random and DDQN selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import amcal_two_action_paper_gain as base
from enhancements.causal_window_query import WINDOW_MODES,run_window_episode

MODES=['DDQNOriginal',*WINDOW_MODES]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--beta',type=float)
    p.add_argument('--window-size',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.beta is None: a.beta=26. if a.dataset=='IAT' else 55.
    if a.window_size<1 or not np.isfinite(a.beta) or a.beta<0: p.error('Use positive window size and finite nonnegative beta.')
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'; runs=root/'protocol_runs'
    candidates=[runs/f'two_action_paper_gain_seed{a.seed}_p300_e3_v1'/'Candidates'/base.beta_name(a.beta),
        runs/f'two_action_paper_gain_extra_seed{a.seed}_p300_e3_v1'/'Candidates'/base.beta_name(a.beta)]
    source=next((path for path in candidates if (path/'Models'/'two_action_ddqn.pth').is_file()
        and (path/'training_complete.json').is_file()),None)
    if source is None: raise ValueError('Matching local source checkpoint missing; no training is started.')
    model=source/'Models'/'two_action_ddqn.pth'; saved=torch.load(model,map_location='cpu',weights_only=False)
    complete=json.loads((source/'training_complete.json').read_text(encoding='utf-8'))
    if saved['protocol']!=complete['protocol'] or base.pilot.digest(model)!=complete['checkpoint_sha256']:
        raise ValueError('Source checkpoint/completion changed.')
    sp=saved['protocol']['protocol']
    if sp['dataset']!=a.dataset or sp['seed']!=a.seed or saved['protocol']['beta']!=a.beta:
        raise ValueError('Source dataset/seed/beta mismatch.')
    for name,sha in sp['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Source implementation changed: '+name)
    a.teacher=runs/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run=runs/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.output=runs/f'causal_window_al_seed{a.seed}_{base.beta_name(a.beta)}_w{a.window_size}_full_v1'
    a.pilot_rows=sp['pilot_rows']; a.budget_fraction=.2
    experiment=base.PaperGainExperiment(a)
    if (experiment.teacher_hash!=sp['healthy_teacher_sha256'] or experiment.actor_sha!=sp['actor_checkpoint_sha256']
            or experiment.healthy['model_hashes']!=sp['source_models']
            or experiment.healthy['training_domain_sha256']!=sp['train_files']):
        raise ValueError('Source CNN/Context/Actor provenance differs.')
    if (experiment.data_manifest('train_ids',base.TRAIN)!=sp['train_data'] or
            experiment.data_manifest('val_ids',base.TRAIN)!=sp['validation_data']):
        raise ValueError('Source TRAIN/VAL streams differ.')
    learner=base.QueryDDQN(experiment.spec,experiment.actor_saved['state'],base.pilot.base.legacy.core.device,a.seed)
    learner.policy.load_state_dict(saved['policy_state']); learner.policy.eval().requires_grad_(False)
    initial_hashes=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()),
        base.weight_hash(learner.policy.state_dict()))
    a.pilot_rows=0  # Full eligible TEST, without changing historical TRAIN sizes.
    domains,reference=experiment.split('test_ids',base.TEST)
    manifest=experiment.data_manifest('test_ids',base.TEST)
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,beta=a.beta,window_size=a.window_size,
        source_checkpoint_sha256=complete['checkpoint_sha256'],source_training_protocol=saved['protocol'],
        budget_fraction=.2,test_rows_per_level=0,TEST_data=manifest,levels=base.TEST,modes=MODES,
        threshold=None,entropy='normalized predictive entropy of Context -> Frozen, scored on arrival',
        windows='predict entire arrived window before requesting top quota labels; no future-window data',
        quota='floor(.2*end)-floor(.2*start); floor total cap, including partial final window',
        matching='UncertaintyWindow, RandomWindow and DDQNWindowRank share window quotas and feedback release times; per-sample delays can differ',
        original='DDQNOriginal retains original immediate decisions/feedback; difference includes delay and pacing',
        ddqn_window='rank Q_query-Q_skip within arrived window; forced quota, not the original DDQN argmax policy',
        context='same initial Context, batch5 paid replay, LR .00025 and replay .1; fixed Context -> Frozen',
        scope='query policy replacement on the same pretrained Context; not independently retrained AL architectures',
        feedback_order='selected rows updated in arrival order, preserving all original window predictions',
        implementation_sha256={name:base.pilot.digest(Path(__file__).parent/name)
            for name in ('amcal_causal_window_al.py','enhancements/causal_window_query.py')})
    pf=a.output/'protocol.json'
    if pf.exists():
        if json.loads(pf.read_text(encoding='utf-8'))!=protocol: raise ValueError('Causal-window protocol changed.')
    elif a.output.exists() and any(a.output.iterdir()): raise ValueError('Unmatched partial output preserved.')
    else: base.pilot.write_json(pf,protocol)
    all_rows=[]; means=[]
    for mode in MODES:
        rows=[]
        for level in base.TEST:
            directory=a.output/'Results'/mode/f'level_{level}'
            ef=directory/'execution.json'; mf=directory/'metrics.json'; tf=directory/'trace.csv'; cf=directory/'completed.json'
            execution=dict(protocol=protocol,mode=mode,level=level,stream_sha256=base.domain_hash(domains[level]))
            if ef.exists():
                if json.loads(ef.read_text(encoding='utf-8'))!=execution: raise ValueError('Window case protocol changed.')
            elif directory.exists() and any(directory.iterdir()): raise ValueError('Unmatched partial case preserved.')
            else: base.pilot.write_json(ef,execution)
            n=len(domains[level][1])
            print(f'START DATASET={a.dataset} METHOD={mode} BETA={a.beta:g} LEVEL={level} '
                f'ROWS={n} CAP={int(.2*n)} WINDOW={a.window_size} | full TEST, no training',flush=True)
            if cf.exists():
                done=json.loads(cf.read_text(encoding='utf-8'))
                if done['execution_sha256']!=base.pilot.digest(ef) or any(
                        not path.is_file() or base.pilot.digest(path)!=done['sha256'][path.name] for path in (mf,tf)):
                    raise ValueError('Completed causal-window case changed.')
                result=json.loads(mf.read_text(encoding='utf-8'))
            else:
                base.pilot.base.legacy.core.set_seed(a.seed)
                if mode=='DDQNOriginal':
                    result,trace=base.run_episode(experiment,learner,domains[level],reference,a.seed,training=False,mode='DDQN')
                    for row in trace:
                        row.update(feedback_step=row['step'] if row['queried'] else None,
                            feedback_delay_rows=0 if row['queried'] else None)
                else:
                    result,trace=run_window_episode(experiment,learner,domains[level],reference,a.seed,mode,a.window_size)
                if result['cap']!=int(.2*n) or result['labels']>result['cap'] or result['td_updates']!=0:
                    raise RuntimeError('Window case violated the budget/frozen-Q contract.')
                base.pilot.write_json(mf,result); base.pilot.write_csv(tf,trace)
                base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(ef),
                    sha256={path.name:base.pilot.digest(path) for path in (mf,tf)}))
            report=dict(dataset=a.dataset,beta=a.beta,window_size=a.window_size,budget_percent=20,
                **base.result_row(mode,level,result))
            report['budget_used_percent']=100*result['labels']/max(1,result['cap'])
            report['mean_feedback_delay_rows']=result.get('mean_feedback_delay_rows',0.)
            report['max_feedback_delay_rows']=result.get('max_feedback_delay_rows',0)
            for key in ('PR','RC','F1'): report['target_final_'+key]=result['target_after'][key]
            rows.append(report); all_rows.append(report)
            print(f'DONE DATASET={a.dataset} METHOD={mode} LEVEL={level} AC={report["AC"]:.2f}% '
                f'PR={report["PR"]:.2f}% RC={report["RC"]:.2f}% F1={report["F1"]:.2f}% '
                f'FINAL_AC={report["target_final_AC"]:.2f}% LABELS={report["labels_percent"]:.2f}% '
                f'CAP_USED={report["budget_used_percent"]:.2f}% FORGET={report["forgetting"]:+.2f} pp '
                f'MEAN_FEEDBACK_DELAY={report["mean_feedback_delay_rows"]:.2f} rows',flush=True)
            base.pilot.write_csv(a.output/'Results'/'summary_by_level.csv',all_rows)
        mean=dict(dataset=a.dataset,method=mode,beta=a.beta,window_size=a.window_size,levels=len(rows),budget_percent=20)
        for key in ('AC','PR','RC','F1','labels_percent','budget_used_percent','source_final_AC','forgetting',
                'target_final_AC','target_final_PR','target_final_RC','target_final_F1','mean_feedback_delay_rows'):
            mean[key]=float(np.mean([r[key] for r in rows]))
        means.append(mean)
        base.pilot.write_csv(a.output/'Results'/'summary_mean.csv',means)
        print('MEAN '+json.dumps(mean,sort_keys=True),flush=True)
    final_hashes=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()),
        base.weight_hash(learner.policy.state_dict()))
    if initial_hashes!=final_hashes or learner.updates!=0 or learner.steps!=0:
        raise RuntimeError('Evaluation changed an original model or DDQN counters.')
    if (base.pilot.digest(model)!=complete['checkpoint_sha256'] or
            base.pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth')!=experiment.actor_sha):
        raise RuntimeError('A source checkpoint file changed.')
    base.pilot.write_json(a.output/'Results'/'completed.json',dict(protocol_sha256=base.pilot.digest(pf),cases=len(all_rows),
        summary_sha256={name:base.pilot.digest(a.output/'Results'/name) for name in ('summary_mean.csv','summary_by_level.csv')}))
    print('CAUSAL WINDOW OUTPUT: '+str(a.output),flush=True)


if __name__=='__main__': main()

