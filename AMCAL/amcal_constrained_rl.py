"""Opt-in constrained sequential DDQN; existing AMCAL entry points are unchanged."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

import amcal_budget_actor as budget
from enhancements.constrained_agent import ConstrainedDDQN,constraint_values,dual_ascent,training_indices
from enhancements.constrained_episode import run_episode,probabilities,entropy,context_accuracy

pilot=budget.pilot
LEVELS=[0,1,3]
TEST_LEVELS=[5,7,10,12,15,17,20]
MODES=['FrontLoadedContext','Uncertainty','Random','NoQueryContext']


def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


def policy_hash(agent):
    h=hashlib.sha256()
    for key,value in sorted(agent.policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def domain_hash(domain):
    h=hashlib.sha256(domain[0].detach().cpu().contiguous().numpy().tobytes())
    for values in domain[1:]:h.update(np.asarray(values).tobytes())
    return h.hexdigest()


def clean_domain(domain,blocked_ids,blocked_keys):
    keep=np.asarray([int(i) not in blocked_ids and key not in blocked_keys
        for i,key in zip(domain[2],pilot.feature_keys(domain[0]))])
    return pilot.subset(domain,keep)


def sample(domain,count,seed):
    selected=pilot.pilot_indices(domain[1],count,seed)
    mask=np.zeros(len(domain[1]),dtype=bool);mask[selected]=True
    return pilot.subset(domain,mask)


def row(method,level,result):
    return dict(method=method,level=level,**{k:100*result[k] for k in ('AC','PR','RC','F1')},
        samples=result['samples'],labels=result['labels'],cap=result['cap'],labels_percent=100*result['label_fraction'],
        source_initial_AC=100*result['source_initial_AC'],source_final_AC=100*result['source_final_AC'],
        source_system_final_AC=100*result.get('source_system_final_AC',result['source_final_AC']),
        forgetting_AC=100*result['forgetting'],milliseconds_per_row=result['milliseconds_per_row'])


def show(method,level,result):
    print(f"{method} level {level}: AC {100*result['AC']:.2f}% | PR {100*result['PR']:.2f}% | "
        f"RC {100*result['RC']:.2f}% | F1 {100*result['F1']:.2f}% | labels {result['labels']}/{result['cap']} "
        f"({100*result['label_fraction']:.2f}%) | source {100*result['source_initial_AC']:.2f}% -> "
        f"{100*result['source_final_AC']:.2f}% | forgetting {100*result['forgetting']:+.2f} pp",flush=True)
    if 'source_system_final_AC' in result:
        print(f"  Routed system source final AC {100*result['source_system_final_AC']:.2f}% (observer only, no queries)",flush=True)


def print_table(rows):
    print('\nMEAN AC/PR/RC/F1 AND LABELS IN PERCENT; FORGETTING IN POINTS',flush=True)
    print('Method                    AC      PR      RC      F1   Labels   Forget',flush=True)
    for method in dict.fromkeys(r['method'] for r in rows):
        group=[r for r in rows if r['method']==method]
        averages=[np.mean([r[key] for r in group]) for key in ('AC','PR','RC','F1','labels_percent','forgetting_AC')]
        print(f'{method:<23}'+''.join(f'{value:8.2f}' for value in averages),flush=True)


class Experiment:
    def __init__(self,a):
        self.a=a
        self.healthy,self.artifact,anchors,reference,self.teacher_hash=budget.setup(a)
        saved_cnn=torch.load(a.models/'initial_cnn_model.pth',map_location='cpu',weights_only=False)
        expected='legacy' if a.dataset=='IAT' else 'light_ps_cnn_v1'
        if saved_cnn.get('architecture','legacy')!=expected:
            raise ValueError('Expected original IAT CNN / Light PS CNN. Existing architecture is preserved; choose the matching teacher.')
        self.cnn=pilot.base.load_cnn(a.models,self.artifact)
        self.template=pilot.base.build_bank(a.models,self.artifact,a.config,self.healthy['context_state_dict'],slots=1)[0]
        self.reference=sample(reference,128,a.seed+701)
        self.reserved=set(self.healthy['provenance']['reference_source_ids'])
        self.ref_keys=set(pilot.feature_keys(reference[0]))
        self.anchors=clean_domain(anchors,self.reserved,self.ref_keys)
        self.anchor_keys=set(pilot.feature_keys(self.anchors[0]))
        _,h=pilot.base.legacy.base_cache(self.cnn,self.anchors[0]);h=h.detach().cpu().numpy()
        self.stats=dict(mean=h.mean(0),scale=np.maximum(h.std(0),.1))
        self.classes=len(a.classes);self.width=24+2*self.classes
        self.domains={};self.references={};self.audit_info={};self.initial_accuracy={}
        self.train={level:sample(clean_domain(pilot.base.legacy.read_domain(a.burst_dir,self.artifact,level,'train_ids'),
            self.reserved,self.ref_keys),a.pilot_rows,a.seed+702) for level in LEVELS}
        self.TRAIN_losses={}
        with torch.no_grad():
            for level,(x,y,_) in self.train.items():
                values=[]
                for start in range(0,len(y),128):
                    labels=torch.as_tensor(y[start:start+128],dtype=torch.long,device=x.device)
                    values.extend(torch.nn.functional.cross_entropy(self.cnn(self.template(x[start:start+128])),
                        labels,reduction='none').cpu().tolist())
                self.TRAIN_losses[level]=np.asarray(values)
        self.split('val_ids',LEVELS)
        self.threshold=self.calibrate()

    def episode_domain(self,level,seed,count=None):
        domain=self.train[level]
        indices=training_indices(domain[1],self.TRAIN_losses[level],self.a.episode_rows if count is None else count,seed,self.a.hard_fraction)
        mask=np.zeros(len(domain[1]),dtype=bool);mask[indices]=True
        return pilot.subset(domain,mask)

    def split(self,split,levels):
        if split in self.domains:return self.domains[split],self.references[split]
        source=pilot.base.legacy.read_domain(self.a.burst_dir,self.artifact,0,split)
        # Audit features must not also be available in the fixed TRAIN replay pool.
        eligible=clean_domain(source,set(),self.anchor_keys)
        _,reference,reserved=pilot.reference_partition(eligible,self.a.seed)
        ref_keys=set(pilot.feature_keys(reference[0]))
        domains={level:sample(clean_domain(pilot.base.legacy.read_domain(self.a.burst_dir,self.artifact,level,split),
            reserved,ref_keys),self.a.pilot_rows,self.a.seed+(703 if split=='val_ids' else 704)) for level in levels}
        self.references[split]=reference;self.domains[split]=domains
        self.audit_info[split]=dict(source_audit_ids=list(map(int,reference[2])),
            candidate_rows={str(level):len(domain[1]) for level,domain in domains.items()},
            rule='source audit IDs/features excluded from all target streams; audit features disjoint from TRAIN replay anchors')
        return domains,reference

    def calibrate(self):
        values=[]
        with torch.no_grad():
            for x,_,_ in self.domains['val_ids'].values():
                for start in range(0,len(x),128):
                    inputs=x[start:start+128]
                    p=probabilities(torch.stack([self.cnn(inputs),self.cnn(self.template(inputs))],1).cpu().numpy())
                    h=entropy(p);values.extend(h.min(1).tolist())
        return float(np.quantile(values,.75))

    def run(self,domain,reference,agent,mode,seed,epsilon=0.):
        key=id(reference)
        if key not in self.initial_accuracy:self.initial_accuracy[key]=context_accuracy(self.cnn,self.template,reference)
        return run_episode(self.cnn,lambda:deepcopy(self.template),
            lambda anchors,seed:pilot.Replay(anchors,seed+61,self.a.config,self.classes),
            pilot.base.legacy.base_cache,
            lambda context,opt,cnn,current,paid,replay:pilot.context_min_step(context,opt,cnn,current,paid,replay,self.a.replay_weight),
            domain,self.anchors,reference,self.stats,self.classes,agent,mode,seed,self.a.budget_fraction,
            self.threshold,self.a.context_lr,epsilon,self.initial_accuracy[key])

    def cached(self,directory,domain,reference,agent,mode,seed):
        directory.mkdir(parents=True,exist_ok=True)
        execution=dict(mode=mode,seed=seed,policy=policy_hash(agent) if mode=='RL' else None,
            ids=list(map(int,domain[2])),threshold=self.threshold,budget=self.a.budget_fraction,
            domain_sha256=domain_hash(domain),reference_sha256=domain_hash(reference))
        record=directory/'metrics.json';trace=directory/'trace.csv'
        if record.exists():
            old=json.loads(record.read_text(encoding='utf-8'))
            if old['execution']!=execution or pilot.digest(trace)!=old['trace_sha256']:
                raise ValueError('Cached evaluation changed: '+str(directory))
            return old['result']
        result,rows,_,_=self.run(domain,reference,agent,mode,seed)
        pilot.write_csv(trace,rows)
        pilot.write_json(record,dict(execution=execution,result=result,trace_sha256=pilot.digest(trace)))
        return result


def validate(experiment,agent,directory,references,tolerances):
    domains,reference=experiment.split('val_ids',LEVELS);rows=[];results={}
    before=policy_hash(agent)
    for level,domain in domains.items():
        r=experiment.cached(directory/f'level_{level}',domain,reference,agent,'RL',experiment.a.seed)
        results[level]=r;rows.append(row('RL',level,r));show('VAL RL',level,r)
    if policy_hash(agent)!=before:raise RuntimeError('Validation fitted the policy.')
    violations=np.stack([constraint_values(results[level],references[level],tolerances) for level in LEVELS])
    maximum=float(np.maximum(violations,0).max());feasible=maximum<=1e-10
    mean_labels=float(np.mean([r['label_fraction'] for r in results.values()]))
    mean_ac=float(np.mean([r['AC'] for r in results.values()]))
    key=(0,mean_labels,-mean_ac) if feasible else (1,maximum,mean_labels,-mean_ac)
    pilot.write_csv(directory/'summary.csv',rows)
    report=dict(feasible=feasible,worst_violation=maximum,violations_by_level=violations.tolist(),
        mean_label_fraction=mean_labels,mean_AC=mean_ac,rows=rows)
    pilot.write_json(directory/'constraints.json',report)
    print(f'VAL constraints: feasible={feasible} | worst violation {100*maximum:.2f} pp | labels {100*mean_labels:.2f}%',flush=True)
    return key,report


def train(a,experiment,agent,protocol):
    completed=a.output/'training_complete.json';model=a.output/'Models'/'constrained_policy.pth'
    if completed.exists():
        c=json.loads(completed.read_text(encoding='utf-8'))
        if c['protocol']!=protocol or pilot.digest(model)!=c['checkpoint_sha256']:raise ValueError('Completed training changed.')
        print('Using completed constrained RL training.',flush=True)
        print_table(json.loads((a.output/'selected_validation.json').read_text())['rows']);return
    models=a.output/'Models';models.mkdir(exist_ok=True)
    val_domains,val_reference=experiment.split('val_ids',LEVELS)
    references={};control_rows=[]
    for mode in MODES:
        for level,domain in val_domains.items():
            r=experiment.cached(a.output/'Controls'/mode/f'level_{level}',domain,val_reference,agent,mode,a.seed)
            show('VAL '+mode,level,r);control_rows.append(row(mode,level,r))
            if mode=='FrontLoadedContext':references[level]=r
    pilot.write_csv(a.output/'validation_controls.csv',control_rows);print_table(control_rows)
    trainer=models/'trainer_state.pth'
    if trainer.exists():
        saved=torch.load(trainer,map_location='cpu',weights_only=False)
        if saved['protocol']!=protocol:raise ValueError('Resume protocol differs; use a new RunTag.')
        agent.restore(saved['agent']);duals=np.asarray(saved['duals']);log=saved['log'];start=saved['episode']+1
        best=tuple(saved['best']) if saved['best'] is not None else None
        print(f'Resuming at episode {start}/{a.episodes}',flush=True)
    else:
        if model.exists():raise ValueError('Unmatched partial model preserved; use a new RunTag.')
        examples=[]
        for level in LEVELS:
            domain=experiment.episode_domain(level,a.seed+1000+level)
            _,_,_,demo=experiment.run(domain,experiment.reference,agent,'Uncertainty',a.seed+1000+level)
            examples.extend(demo)
        warm_loss=agent.warm_start(examples,100)
        pilot.write_json(a.output/'warmup.json',dict(TRAIN_examples=len(examples),loss=warm_loss,
            teacher='uncertainty query and lower-entropy routing; no TRAIN label target used for imitation',
            note='q75 cutoff calibrated on unlabelled VAL features; demonstrations themselves are TRAIN only'))
        print(f'Uncertainty initialization: {len(examples)} TRAIN states | loss {warm_loss:.4f}',flush=True)
        duals=np.full(3,5.);log=[];start=1;best=None
    tolerances=np.array([a.ac_tolerance,a.f1_tolerance,a.forgetting_tolerance])
    for episode in range(start,a.episodes+1):
        level=LEVELS[(episode-1)%3];seed=a.seed+episode*7919
        domain=experiment.episode_domain(level,seed,a.pilot_rows if episode<=3 else a.episode_rows)
        baseline,_,_,_=experiment.run(domain,experiment.reference,agent,'FrontLoadedContext',seed)
        no_query,_,_,_=experiment.run(domain,experiment.reference,agent,'NoQueryContext',seed)
        epsilon=.3-(.25*(episode-1)/max(a.episodes-1,1))
        r,trace,experience,_=experiment.run(domain,experiment.reference,agent,'RL',seed,epsilon)
        agent.add_episode(*experience);loss=agent.learn(duals,a.updates_per_episode)
        g=constraint_values(r,baseline,tolerances);before=duals.copy();duals=dual_ascent(duals,g,a.dual_lr)
        record=dict(episode=episode,epsilon=epsilon,TD_loss=loss,
            dual_before=before.tolist(),dual_after=duals.tolist(),violations=g.tolist(),
            no_query_AC=100*no_query['AC'],no_query_F1=100*no_query['F1'],
            reference_AC_gain_pp=100*(baseline['AC']-no_query['AC']),
            reference_F1_gain_pp=100*(baseline['F1']-no_query['F1']),
            **row('RL',level,r))
        log.append(record)
        traces=a.output/'TRAIN';traces.mkdir(exist_ok=True)
        pilot.write_csv(traces/f'episode_{episode:03d}.csv',trace)
        pilot.write_json(a.output/'training.json',dict(protocol=protocol,episodes=log))
        show(f'TRAIN episode {episode}/{a.episodes}',level,r)
        print('  multipliers AC/F1/forget: '+', '.join(f'{v:.3f}' for v in duals)+f' | TD loss {loss}',flush=True)
        print(f"  TRAIN adaptation headroom: reference minus no-query AC {record['reference_AC_gain_pp']:+.2f} pp | "
            f"F1 {record['reference_F1_gain_pp']:+.2f} pp",flush=True)
        if episode%a.validate_every==0 or episode==a.episodes:
            key,report=validate(experiment,agent,a.output/'VAL'/f'episode_{episode:03d}',references,tolerances)
            if best is None or key<best:
                best=key
                save_torch(model,dict(version=1,protocol=protocol,policy=deepcopy(agent.policy.state_dict()),
                    duals=duals.tolist(),selected_episode=episode,validation=report,
                    source_statistics={k:v.tolist() for k,v in experiment.stats.items()},threshold=experiment.threshold))
                pilot.write_json(a.output/'selected_validation.json',dict(selected_episode=episode,**report))
        save_torch(trainer,dict(protocol=protocol,agent=agent.snapshot(),duals=duals.tolist(),episode=episode,
            log=log,best=best))
    selected=json.loads((a.output/'selected_validation.json').read_text(encoding='utf-8'))
    pilot.write_json(completed,dict(protocol=protocol,checkpoint_sha256=pilot.digest(model),
        selected_episode=selected['selected_episode'],feasible=selected['feasible']))
    print_table(control_rows+selected['rows'])
    print(f"Selected episode {selected['selected_episode']}; constraints feasible={selected['feasible']}. No feasibility or dominance guarantee is implied.",flush=True)


def evaluate(a,experiment,agent,protocol):
    path=a.output/'Models'/'constrained_policy.pth'
    completion=json.loads((a.output/'training_complete.json').read_text(encoding='utf-8'))
    if completion['protocol']!=protocol or completion['checkpoint_sha256']!=pilot.digest(path):raise ValueError('Training checkpoint changed.')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    agent.policy.load_state_dict(saved['policy'])
    domains,reference=experiment.split('test_ids',TEST_LEVELS);rows=[]
    for mode in ['RL',*MODES]:
        for level,domain in domains.items():
            result=experiment.cached(a.output/'Results'/mode/f'level_{level}',domain,reference,agent,mode,a.seed)
            show('TEST '+mode,level,result);rows.append(row(mode,level,result))
    pilot.write_csv(a.output/'Results'/'summary.csv',rows)
    pilot.write_json(a.output/'Results'/'execution.json',dict(protocol=protocol,audit=experiment.audit_info,
        selected_episode=saved['selected_episode'],validation_feasible=saved['validation']['feasible'],
        label='matched deployment controls; same healthy Context, not full retraining ablations'))
    print_table(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--action',choices=['Train','Evaluate','All'],default='Train')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--pilot-rows',type=int,default=300)
    p.add_argument('--episode-rows',type=int,default=96);p.add_argument('--episodes',type=int,default=12)
    p.add_argument('--validate-every',type=int,default=4);p.add_argument('--updates-per-episode',type=int,default=20)
    p.add_argument('--policy-lr',type=float,default=.001);p.add_argument('--dual-lr',type=float,default=.5)
    p.add_argument('--hard-fraction',type=float,default=.5)
    p.add_argument('--ac-tolerance',type=float,default=.01);p.add_argument('--f1-tolerance',type=float,default=.01)
    p.add_argument('--forgetting-tolerance',type=float,default=.005)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--run-tag',default='pilot_v1');a=p.parse_args()
    if min(a.episodes,a.validate_every,a.updates_per_episode)<1 or a.episode_rows<32 or a.pilot_rows<32:
        p.error('Use positive episodes/updates/intervals and at least 32 pilot/episode rows.')
    if a.episode_rows>a.pilot_rows:p.error('EpisodeRows cannot exceed PilotRows.')
    if not np.isfinite(a.hard_fraction) or not 0<=a.hard_fraction<=1:p.error('HardFraction must be between zero and one.')
    values=[a.policy_lr,a.dual_lr,a.ac_tolerance,a.f1_tolerance,a.forgetting_tolerance]
    if not np.isfinite(values).all() or min(values)<0 or a.policy_lr==0:p.error('Invalid learning rates/tolerances.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_tag):p.error('RunTag must use letters, numbers, underscore or hyphen.')
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.teacher=root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    a.output=root/'protocol_runs'/f'constrained_rl_seed{a.seed}_p{a.pilot_rows}_e{a.episodes}_n{a.episode_rows}_{a.run_tag}'
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.epochs=a.episodes;a.budget_fraction=.2
    print(f'Constrained RL {a.dataset}: 20% hard cap, optional spending | {a.episodes} TRAIN episodes | '
        f'first cycle up to {a.pilot_rows} rows, subsequent episodes up to {a.episode_rows}',flush=True)
    experiment=Experiment(a)
    pilot.base.legacy.core.set_seed(a.seed)
    agent=ConstrainedDDQN(experiment.width,pilot.base.legacy.core.device,a.seed,a.policy_lr)
    protocol=dict(version=1,dataset=a.dataset,seed=a.seed,pilot_rows=a.pilot_rows,episode_rows=a.episode_rows,
        episodes=a.episodes,validate_every=a.validate_every,updates_per_episode=a.updates_per_episode,
        policy_lr=a.policy_lr,dual_lr=a.dual_lr,tolerances=values[2:],budget_fraction=.2,
        TRAIN_curriculum=dict(hard_fraction=a.hard_fraction,
            length_schedule='first three episodes at pilot length; subsequent episodes at short EpisodeRows length',
            rule='class-covered higher healthy-Context-loss TRAIN examples plus uniform remainder; no synthetic features or TEST sampling'),
        context_lr=a.context_lr,replay_weight=a.replay_weight,teacher_checkpoint_sha256=experiment.teacher_hash,
        TRAIN_source_audit_ids=list(map(int,experiment.reference[2])),
        state_width=experiment.width,network='compact three-action DDQN, 32 hidden units',gamma=1.,
        actions={'0':'Frozen prediction','1':'Context prediction','2':'Context prediction, then paid feedback/update'},
        objective='min policy label fraction; max nonnegative AC/F1/source-forgetting constraint multipliers',
        theory='policy-versus-dual constrained min-max; not selector-versus-Context zero-sum training; no global convergence guarantee',
        reward='[-queries/N, correct/N, delta macro F1, terminal -source Context forgetting]; replay reweighted using current duals',
        reset='Context, Adam, memory, history and budget per episode/intensity; Frozen weights unchanged',
        audit=experiment.audit_info,threshold=experiment.threshold,
        data_scope='TRAIN/VAL 0/1/3; TEST only on explicit Evaluate/All; audit IDs/features excluded from streams',
        comparison='same healthy initial Context, paid replay update and rows for every deployment control',
        units='JSON quality/constraints are fractions; summary CSV quality/labels are percent, forgetting in points')
    a.output.mkdir(parents=True,exist_ok=True)
    config=a.output/'protocol.json'
    if config.exists() and json.loads(config.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Run settings differ from existing results. Use a new RunTag.')
    pilot.write_json(config,protocol)
    if a.action in ('Train','All'):train(a,experiment,agent,protocol)
    if a.action in ('Evaluate','All'):evaluate(a,experiment,agent,protocol)
    if pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=experiment.teacher_hash:raise RuntimeError('Healthy teacher changed.')
    print(f'Outputs: {a.output}',flush=True)


if __name__=='__main__':main()
