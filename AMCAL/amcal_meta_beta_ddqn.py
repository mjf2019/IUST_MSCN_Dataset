"""Experimental TRAIN-meta learned reward weight; original CNN/Context/DDQN retained."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
import amcal_two_action_paper_gain as base
from enhancements.meta_beta_ddqn import MetaBetaDDQN, MetaRecorder, LearnedGainReward


def reserve_meta(experiment,inner_probes,rows=100,probe_rows=24):
    api = base.pilot; a = experiment.a
    blocked = [*experiment.train.values(),experiment.reference,*inner_probes]
    ids = {int(i) for d in blocked for i in d[2]}
    keys = set().union(*(set(api.feature_keys(d[0])) for d in blocked))
    streams,probes,manifest = {},{},{}
    for level in base.TRAIN:
        full = api.base.legacy.read_domain(a.burst_dir,experiment.artifact,level,'train_ids')
        for name,size,seed in [('stream',rows,a.seed+9601+level),('probe',probe_rows,a.seed+9701+level)]:
            keep = np.asarray([int(i) not in ids and key not in keys
                for i,key in zip(full[2],api.feature_keys(full[0]))])
            pool = api.subset(full,keep)
            if len(pool[1]) < size or set(np.unique(pool[1])) != set(range(len(a.classes))):
                raise ValueError('Insufficient disjoint TRAIN meta data; no VAL/TEST fallback.')
            # Feature deduplication precedes stratified sampling.
            unique = np.zeros(len(pool[1]),dtype=bool); seen = set()
            for i,key in enumerate(api.feature_keys(pool[0])):
                if key not in seen: unique[i] = True; seen.add(key)
            pool = api.subset(pool,unique)
            if len(pool[1]) < size: raise ValueError('Too few unique TRAIN meta rows.')
            at = api.pilot_indices(pool[1],size,seed)
            mask = np.zeros(len(pool[1]),dtype=bool); mask[at] = True
            data = api.subset(pool,mask)
            if len(data[1]) != size or set(np.unique(data[1])) != set(range(len(a.classes))):
                raise ValueError('Meta TRAIN sampling lost rows/classes.')
            ids.update(map(int,data[2])); keys.update(api.feature_keys(data[0]))
            (streams if name == 'stream' else probes)[level] = data
            manifest[f'{name}_{level}'] = dict(rows=len(data[1]),sha256=base.domain_hash(data),
                ids=list(map(int,data[2])))
    held = [*streams.values(),*probes.values()]
    held_keys = set().union(*(set(api.feature_keys(d[0])) for d in held))
    held_ids = {int(i) for d in held for i in d[2]}
    anchors = experiment.train_anchors
    keep = np.asarray([int(i) not in held_ids and key not in held_keys
        for i,key in zip(anchors[2],api.feature_keys(anchors[0]))])
    anchors = api.subset(anchors,keep)
    if not len(anchors[1]) or set(np.unique(anchors[1])) != set(range(len(a.classes))):
        raise ValueError('Meta reservation removed source replay classes.')
    return streams,probes,anchors,manifest


def learner_from_source(experiment,saved):
    learner = MetaBetaDDQN(experiment.spec,experiment.actor_saved['state'],
        base.pilot.base.legacy.core.device,experiment.a.seed)
    learner.policy.load_state_dict(saved['policy_state'])
    learner.target.load_state_dict(learner.policy.state_dict())
    return learner


def bind_rewards(experiment,learner,inner_probes,meta_streams,meta_probes,retention):
    experiment.gain_by_domain = {
        id(experiment.train[level]):LearnedGainReward(probe,retention,learner,experiment.a)
        for level,probe in zip(base.TRAIN,inner_probes)}
    experiment.gain_by_domain.update({id(meta_streams[level]):
        LearnedGainReward(meta_probes[level],retention,learner,experiment.a) for level in base.TRAIN})


def train(experiment,learner,protocol,inner,meta_streams,meta_probes,retention):
    a = experiment.a; out = a.output
    state_file = out/'Models'/'trainer.pth'; model_file = out/'Models'/'learned_beta.pth'
    completion = out/'training_complete.json'
    if completion.exists():
        done = json.loads(completion.read_text(encoding='utf-8'))
        model = torch.load(model_file,map_location='cpu',weights_only=False)
        if done['protocol'] != protocol or model['protocol'] != protocol or done['sha256'] != base.pilot.digest(model_file):
            raise ValueError('Completed learned-beta model changed.')
        print('Using verified learned-beta training; no retraining.',flush=True); return
    if state_file.exists():
        saved = torch.load(state_file,map_location='cpu',weights_only=False)
        if saved['protocol'] != protocol: raise ValueError('Resume protocol changed.')
        learner.restore(saved['learner']); history=saved['history']; best=tuple(saved['best']); start=saved['epoch']+1
    else: history=[]; best=(-float('inf'),-float('inf')); start=1
    bind_rewards(experiment,learner,inner,meta_streams,meta_probes,retention)
    for epoch in range(start,a.epochs+1):
        base.pilot.base.legacy.core.set_seed(a.seed+epoch)
        a.budget_fraction=.2
        collector = MetaRecorder(learner)
        for level in base.TRAIN:
            result,trace = base.run_episode(experiment,collector,meta_streams[level],None,
                a.seed+epoch*1009+level*101+70000,training=True,epsilon=.3)
            base.pilot.write_csv(out/'META'/f'epoch_{epoch}'/f'level_{level}_trace.csv',trace)
            print(f'META TRAIN DATASET={a.dataset} epoch={epoch} level={level} '
                f'labels={result["labels"]}/{result["cap"]} | no live Q/beta updates',flush=True)
        learner.set_meta_rows(collector.rows)
        print(f'TRAIN-ONLY beta calibration={learner.beta_value():.6f}; meta rows={len(collector.rows)}; '
            f'gain RMS={learner.gain_scale:.6g}',flush=True)
        epsilon=.3-.25*(epoch-1)/max(a.epochs-1,1); reports=[]
        for level in base.TRAIN:
            a.beta=learner.beta_value()
            result,trace=base.run_episode(experiment,learner,experiment.train[level],None,
                a.seed+epoch*1009+level*101,training=True,epsilon=epsilon)
            base.pilot.write_csv(out/'TRAIN'/f'epoch_{epoch}'/f'level_{level}_trace.csv',trace)
            reports.append(dict(level=level,**result))
            print(f'TRAIN DATASET={a.dataset} EPOCH={epoch}/{a.epochs} LEVEL={level} '
                f'LEARNED_BETA={learner.beta_value():.6f} AC={result["online"]["AC"]:.2f}% '
                f'F1={result["online"]["F1"]:.2f}% LABELS={result["labels"]}/{result["cap"]} '
                f'TD_UPDATES={learner.updates} META_UPDATES={learner.meta_updates}',flush=True)
        a.beta=learner.beta_value()
        rows=base.evaluate(experiment,learner,'val_ids',base.TRAIN,out/'VAL'/f'epoch_{epoch}',controls=False)
        score=float(np.mean([r['AC']-max(0.,r['forgetting']) for r in rows]))
        labels=float(np.mean([r['labels_percent'] for r in rows])); rank=(score,-labels)
        history.append(dict(epoch=epoch,training=reports,validation=rows,score=score,
            labels_percent=labels,beta=learner.beta_value(),meta_updates=learner.meta_updates))
        if rank>best:
            best=rank
            base.parent.save_model(model_file,dict(protocol=protocol,epoch=epoch,
                policy_state=deepcopy(learner.policy.state_dict()),beta=learner.beta_value(),
                gain_scale=learner.gain_scale))
        base.parent.save_model(state_file,dict(protocol=protocol,epoch=epoch,
            learner=learner.snapshot(),history=history,best=list(best)))
        base.pilot.write_json(out/'training.json',dict(protocol=protocol,epochs=history))
        base.pilot.write_csv(out/'beta_learning.csv',learner.meta_history)
        print(f'VAL DATASET={a.dataset} EPOCH={epoch} BETA={learner.beta_value():.6f} '
            f'SCORE={score:.2f} LABELS={labels:.2f}% | checkpoint selection uses VAL only',flush=True)
    model=torch.load(model_file,map_location='cpu',weights_only=False)
    base.pilot.write_json(completion,dict(protocol=protocol,selected_epoch=model['epoch'],
        selected_beta=model['beta'],sha256=base.pilot.digest(model_file)))


def evaluate(experiment,source,protocol):
    a=experiment.a; out=a.output
    done=json.loads((out/'training_complete.json').read_text(encoding='utf-8'))
    path=out/'Models'/'learned_beta.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if done['protocol']!=protocol or saved['protocol']!=protocol or done['sha256']!=base.pilot.digest(path):
        raise ValueError('Learned-beta TEST checkpoint/completion mismatch.')
    learned=learner_from_source(experiment,source)
    learned.policy.load_state_dict(saved['policy_state'])
    with torch.no_grad(): learned.log_beta.copy_(learned.log_beta.new_tensor(np.log(saved['beta'])))
    learned.policy.eval().requires_grad_(False); learned.log_beta.requires_grad_(False)
    fixed=learner_from_source(experiment,source); fixed.policy.eval().requires_grad_(False)
    with torch.no_grad(): fixed.log_beta.copy_(fixed.log_beta.new_tensor(np.log(protocol['source_beta'])))
    fixed.log_beta.requires_grad_(False)
    a.pilot_rows=a.test_rows
    domains,reference=experiment.split('test_ids',base.TEST)
    test_manifest=experiment.data_manifest('test_ids',base.TEST)
    summaries=[]
    for budget in a.test_budgets:
        a.budget_fraction=budget
        for name,agent,mode,beta in [('LearnedBeta',learned,'DDQN',saved['beta']),
                ('SourceFixed',fixed,'DDQN',protocol['source_beta']),
                ('Uncertainty',fixed,'Uncertainty',protocol['source_beta'])]:
            a.beta=beta
            root=out/'Results'/f'testp{a.test_rows}_budget{100*budget:g}'/name
            execution=dict(protocol=protocol,policy=name,budget_fraction=budget,
                learned_checkpoint_sha256=done['sha256'],TEST_data=test_manifest,
                beta=beta,evaluator_sha256=base.pilot.digest(Path(__file__)))
            ef=root/'execution.json'; cf=root/'completed.json'
            if ef.exists():
                if json.loads(ef.read_text(encoding='utf-8'))!=execution: raise ValueError('TEST recipe changed.')
            elif root.exists() and any(root.iterdir()): raise ValueError('Unmatched partial TEST result.')
            else: base.pilot.write_json(ef,execution)
            rows=[]; before=base.weight_hash(agent.policy.state_dict())
            frozen_beta=agent.beta_value(); updates,steps=agent.updates,agent.steps
            for level in base.TEST:
                mf=root/f'level_{level}'/'metrics.json'; tf=root/f'level_{level}'/'trace.csv'; lf=root/f'level_{level}'/'completed.json'
                if lf.exists():
                    record=json.loads(lf.read_text(encoding='utf-8'))
                    if record['execution_sha256']!=base.pilot.digest(ef) or any(
                            base.pilot.digest(f)!=record['sha256'][f.name] for f in (mf,tf)):
                        raise ValueError('Completed TEST case changed.')
                    result=json.loads(mf.read_text(encoding='utf-8'))
                else:
                    result,trace=base.run_episode(experiment,agent,domains[level],reference,a.seed,training=False,mode=mode)
                    base.pilot.write_json(mf,result); base.pilot.write_csv(tf,trace)
                    base.pilot.write_json(lf,dict(execution_sha256=base.pilot.digest(ef),
                        sha256={f.name:base.pilot.digest(f) for f in (mf,tf)}))
                if (result['td_updates']!=0 or result['cap']!=int(budget*len(domains[level][1])) or
                        result['labels']>result['cap']): raise RuntimeError('TEST feedback cap violated.')
                row=dict(dataset=a.dataset,budget_percent=100*budget,beta=beta,**base.result_row(name,level,result))
                row['budget_used_percent']=100*result['labels']/max(1,result['cap'])
                for key in ('PR','RC','F1'): row['target_final_'+key]=result['target_after'][key]
                rows.append(row)
                print(f'TEST DATASET={a.dataset} POLICY={name} BETA={beta:.6f} BUDGET={100*budget:g}% LEVEL={level} '
                    f'AC={row["AC"]:.2f}% PR={row["PR"]:.2f}% RC={row["RC"]:.2f}% F1={row["F1"]:.2f}% '
                    f'FINAL_AC={row["target_final_AC"]:.2f}% '
                    f'LABELS={row["labels_percent"]:.2f}% CAP_USED={row["budget_used_percent"]:.2f}% '
                    f'FORGET={row["forgetting"]:+.2f} pp',flush=True)
            if (before!=base.weight_hash(agent.policy.state_dict()) or frozen_beta!=agent.beta_value()
                    or updates!=agent.updates or steps!=agent.steps):
                raise RuntimeError('TEST changed Q/beta/training counters.')
            base.pilot.write_csv(root/'summary_by_level.csv',rows)
            mean=dict(dataset=a.dataset,method=name,beta=beta,budget_percent=100*budget,levels=len(rows))
            for key in ('AC','PR','RC','F1','labels_percent','budget_used_percent','forgetting','source_final_AC',
                    'target_final_AC','target_final_PR','target_final_RC','target_final_F1'):
                mean[key]=float(np.mean([r[key] for r in rows]))
            summaries.append(mean)
            base.pilot.write_json(cf,dict(execution_sha256=base.pilot.digest(ef),
                summary_sha256=base.pilot.digest(root/'summary_by_level.csv')))
            print('MEAN '+json.dumps(mean,sort_keys=True),flush=True)
    budget_tag='_'.join(format(100*b,'g') for b in a.test_budgets)
    base.pilot.write_csv(out/'Results'/f'comparison_testp{a.test_rows}_budgets_{budget_tag}.csv',summaries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--action',choices=['Train','Evaluate','All'],default='All')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--test-rows',type=int,default=300)
    p.add_argument('--test-budgets',type=float,nargs='+',default=[.2])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--source-run',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--actor-run',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.epochs<1 or a.test_rows<0 or 0<a.test_rows<32 or any(not np.isfinite(b) or not 0<b<=1 for b in a.test_budgets):
        p.error('Use positive epochs, rows zero or at least 32 and TEST budgets in (0,1].')
    a.test_budgets=list(dict.fromkeys(a.test_budgets))
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    source_beta=26. if a.dataset=='IAT' else 55.
    a.source_run=a.source_run or root/'protocol_runs'/f'two_action_paper_gain_extra_seed{a.seed}_p300_e3_v1'/'Candidates'/base.beta_name(source_beta)
    source_path=a.source_run/'Models'/'two_action_ddqn.pth'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    source_done=json.loads((a.source_run/'training_complete.json').read_text(encoding='utf-8'))
    sp=source['protocol']['protocol']
    if (source_done['protocol']!=source['protocol'] or source_done['checkpoint_sha256']!=base.pilot.digest(source_path)
            or sp['dataset']!=a.dataset or sp['seed']!=a.seed or source['protocol']['beta']!=source_beta):
        raise ValueError('Source selected-beta checkpoint mismatch.')
    for name,sha in sp['implementation_sha256'].items():
        if base.pilot.digest(Path(__file__).parent/name)!=sha: raise ValueError('Source implementation changed: '+name)
    a.pilot_rows=sp['pilot_rows']; a.budget_fraction=.2; a.beta=1.
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.teacher=a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.actor_run=a.actor_run or root/'protocol_runs'/f'competitive_selector_seed{a.seed}_p300_r2_v1_paper_reward'
    a.output=a.output or root/'protocol_runs'/f'meta_beta_ddqn_seed{a.seed}_p{a.pilot_rows}_e{a.epochs}_v1'
    experiment=base.PaperGainExperiment(a)
    source_weight_hashes=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict()))
    if (experiment.teacher_hash!=sp['healthy_teacher_sha256'] or experiment.actor_sha!=sp['actor_checkpoint_sha256']
            or experiment.healthy['model_hashes']!=sp['source_models']):
        raise ValueError('Source CNN/Context/Actor lineage differs.')
    if (experiment.data_manifest('train_ids',base.TRAIN)!=sp['train_data'] or
            experiment.data_manifest('val_ids',base.TRAIN)!=sp['validation_data']):
        raise ValueError('Source TRAIN/VAL streams differ.')
    inner,retention,anchors,probe_manifest=base.prepare_train_probes(experiment,base.pilot,24,32)
    experiment.train_anchors=anchors
    meta_streams,meta_probes,anchors,meta_manifest=reserve_meta(experiment,inner)
    experiment.train_anchors=anchors
    experiment.reward_probe_keys=set().union(*(set(base.pilot.feature_keys(d[0])) for d in [*inner,*meta_probes.values()]))
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,epochs=a.epochs,pilot_rows=a.pilot_rows,
        source_beta=source_beta,source_checkpoint_sha256=base.pilot.digest(source_path),source_protocol=source['protocol'],
        inner_probe_manifest=probe_manifest,meta_TRAIN_manifest=meta_manifest,
        inner_probe_sha256=[base.domain_hash(d) for d in inner],meta_replay_sha256=base.domain_hash(anchors),
        budget_fraction=.2,context_lr=a.context_lr,replay_weight=a.replay_weight,
        implementation_sha256={name:base.pilot.digest(Path(__file__).parent/name) for name in
            ('amcal_meta_beta_ddqn.py','enhancements/meta_beta_ddqn.py')},
        beta='positive trainable log-weight; TRAIN RMS initialization; Adam .01; log range [-8,8] numerical guard',
        meta='held-out TRAIN paid transitions and separate same-domain probes; magnitude-preserving signed gain objective; '
            'one virtual SGD critic step; first-order local optimizer approximation to actual Adam; meta every 15 TD steps',
        replay='raw paper/gain components; rewards recomputed with current detached beta; no stale scalar rewards',
        preserved='source CNN, initial Context, competitive Context update, two-action DDQN gamma .95; fixed Context path',
        theory='local competitive Context update and closed-loop DDQN preserved; no exact shared min-max equivalence claim',
        validation='epoch selection by mean online AC minus positive source forgetting; fewer labels break ties',
        scope='experimental learned weight, warm-start selected DDQN; comparison not an independent no-RL retraining',
        labels='20 percent per inner/meta trajectory; probes and prior training additional TRAIN supervision; TEST cap per level')
    pf=a.output/'protocol.json'
    if pf.exists():
        if json.loads(pf.read_text(encoding='utf-8'))!=protocol: raise ValueError('Protocol changed; use a fresh output.')
    elif a.output.exists() and any(a.output.iterdir()): raise ValueError('Unmatched partial output preserved.')
    else:
        if a.action=='Evaluate': raise ValueError('Train the separate learned-beta checkpoint first.')
        base.pilot.write_json(pf,protocol)
    learner=learner_from_source(experiment,source)
    if a.action in ('Train','All'): train(experiment,learner,protocol,inner,meta_streams,meta_probes,retention)
    if a.action in ('Evaluate','All'): evaluate(experiment,source,protocol)
    if source_weight_hashes!=(base.weight_hash(experiment.cnn.state_dict()),base.weight_hash(experiment.template.state_dict())):
        raise RuntimeError('Original CNN or initial Context weights changed.')
    if base.pilot.digest(source_path)!=protocol['source_checkpoint_sha256']:
        raise RuntimeError('Original source checkpoint changed.')
    if (base.pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=experiment.teacher_hash or
            base.pilot.digest(a.actor_run/'Models'/'selector.pth')!=experiment.actor_sha):
        raise RuntimeError('A source teacher/Actor checkpoint changed.')
    print('OUTPUT: '+str(a.output),flush=True)


if __name__=='__main__': main()

