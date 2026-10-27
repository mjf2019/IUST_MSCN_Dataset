"""Actor-only budget-ranking control on a fixed healthy Context checkpoint.

The default signed TRAIN score and Context update function are unchanged.
UtilityEarly explicitly reweights competitive components using TRAIN scales
and confines feedback to the first 20 percent; it is a separate experiment.
TRAIN collection uses temporary uniform-feedback Context trajectories. Only the
query Actor is fitted/saved; selected Context/CNN weights are copied unchanged.
This is a fixed-quota local min-max surrogate, not an exact multistep game.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import math
import sys

import numpy as np
import torch

import amcal_minimax_paironly as pilot
from enhancements.budget_actor import quota_max_step
from enhancements.standardized_actor import StandardizedQueryMaximizer
from enhancements.config import EnhancementConfig
from enhancements.console_summary import print_level_result, print_saved_summaries
from enhancements.budget_actor_math import fit_signal_scales, scaled_competitive_score, query_window_count

VERSION = 1
LEVELS = [5, 7, 10, 12, 15, 17, 20]


def pilot_domain(a, domain, offset):
    return pilot.pilot_domain(a, domain, offset)


def setup(a):
    if a.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable.')
    device = torch.device('cuda' if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    pilot.base.legacy.core.device = device
    pilot.base.legacy.core.set_seed(a.seed)
    checkpoint = a.teacher/'Models'/'minimax_query.pth'
    healthy = torch.load(checkpoint, map_location='cpu', weights_only=False)
    completion = json.loads((a.teacher/'training_complete.json').read_text(encoding='utf-8'))
    if healthy['version'] != pilot.VERSION or healthy['dataset'] != a.dataset or healthy['settings']['seed'] != a.seed:
        raise ValueError('Healthy checkpoint protocol/dataset/seed mismatch.')
    if completion['checkpoint_sha256'] != pilot.digest(checkpoint): raise ValueError('Healthy checkpoint hash mismatch.')
    if healthy.get('pilot'): raise ValueError('Use the full healthy checkpoint, not a small-data pilot.')
    if set(healthy['training_levels']) != {0,1,3}: raise ValueError('Healthy TRAIN levels must be 0/1/3.')
    a.models = a.teacher/'Models'
    a.config = EnhancementConfig(**healthy['config']).validated()
    for key, value in healthy['settings'].items():
        if key not in ('budget_fraction', 'epochs', 'patience', 'seed'): setattr(a, key, value)
    for name, expected in healthy['model_hashes'].items():
        if pilot.digest(a.models/name) != expected: raise ValueError('Healthy model provenance changed: '+name)
    for level, expected in healthy['training_domain_sha256'].items():
        if pilot.digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') != expected:
            raise ValueError('TRAIN data changed since healthy training.')
    artifact = pilot.base.load_artifact(a.models, a.burst_dir)
    a.classes = list(artifact['label_encoder'].classes_)
    a.context_count, a.gamma = 1, .95
    a.eval_query_schedule, a.eval_routing, a.train_query_schedule = 'policy', 'policy', 'uniform'
    a.budget_actor_mode = True
    a.query_target, a.actor_logit_penalty = 'competitive', 0.
    source = pilot.base.legacy.read_domain(a.burst_dir, artifact, 0, 'train_ids')
    reserved = set(healthy['provenance']['reference_source_ids'])
    mask = np.array([int(i) in reserved for i in source[2]])
    reference, anchors = pilot.subset(source, mask), pilot.subset(source, ~mask)
    if not len(reference[1]): raise ValueError('Independent TRAIN retention probe is missing.')
    return healthy, artifact, anchors, reference, pilot.digest(checkpoint)


def runtime(a, healthy, artifact, query_state=None):
    return pilot.new_runtime(a, artifact, healthy, healthy['context_state_dict'], query_state)


class Recorder:
    def __init__(self): self.states, self.scores, self.rows = [], [], []
    def observe(self, state, row):
        self.rows.append(dict(row))
        if row.get('score') is not None:
            self.states.append(state.detach().cpu().reshape(-1).clone())
            self.scores.append(float(row['score']))


def collect(a, healthy, artifact, anchors, reference):
    a.actor_experiment = healthy.get('actor_experiment')
    a.collect_query_targets_only = True
    ref_keys = set(pilot.feature_keys(reference[0]))
    reserved = set(healthy['provenance']['reference_source_ids'])
    states, scores, all_rows = [], [], []
    for level in (0,1,3):
        domain = pilot.base.legacy.read_domain(a.burst_dir, artifact, level, 'train_ids')
        mask = np.array([int(i) not in reserved and key not in ref_keys
            for i,key in zip(domain[2],pilot.feature_keys(domain[0]))])
        domain = pilot_domain(a, pilot.subset(domain,mask), 702)
        recorder = Recorder()
        pilot.base.legacy.core.set_seed(a.seed)
        live = runtime(a,healthy,artifact,healthy['query_state_dict'])
        with torch.no_grad(): before_query = deepcopy(live[3].state_dict())
        print(f'Collect TRAIN level {level}: {len(domain[1])} rows; temporary Context only', flush=True)
        live, result, _ = pilot.stream(a,live,domain,anchors,reference,domain,a.seed+level*101,
            training=True,diagnostic_sink=recorder)
        if any(not torch.equal(before_query[k],live[3].state_dict()[k]) for k in before_query):
            raise RuntimeError('Actor changed during fixed-target collection.')
        states.extend(recorder.states); scores.extend(recorder.scores)
        for row in recorder.rows: row['level'] = level
        all_rows.extend(recorder.rows)
        pilot.write_json(a.output/f'collection_level{level}.json',dict(samples=result['samples'],labels=result['labels'],
            cap=result['cap'],source_ids=list(map(int,domain[2])),scope='TRAIN only; temporary states discarded'))
    a.collect_query_targets_only = False
    if len(scores)<32 or not np.isfinite(scores).all() or np.std(scores)<1e-8:
        raise ValueError('Insufficient finite variation in TRAIN ranking targets.')
    if getattr(a,'utility_early',False):
        measured=[row for row in all_rows if row.get('score') is not None]
        a.ranking_groups=[row['level'] for row in measured]
        a.signal_scales=fit_signal_scales(measured)
        scores=[scaled_competitive_score(row,a.signal_scales) for row in measured]
        for row,score in zip(measured,scores):row['ranking_target']=score
        pilot.write_json(a.output/'signal_scaling.json',dict(scope='measured TRAIN only',
            scales=a.signal_scales,raw_target='gap + weighted_gain - weighted_harm - request_cost',
            ranking_target='gap/sd_gap + weighted_gain/sd_gain - weighted_harm/sd_harm - request_cost'))
        print('TRAIN signal SD: '+', '.join(f'{key}={value:.6f}' for key,value in a.signal_scales.items()),flush=True)
        print(f'Ranking targets: {sum(score>0 for score in scores)}/{len(scores)} positive',flush=True)
    pilot.write_csv(a.output/'train_targets.csv',all_rows)
    return torch.stack(states), torch.tensor(scores,dtype=torch.float32)


def normalizer(states):
    mean = states.double().mean(0); scale = states.double().std(0,unbiased=False)
    scale = torch.where(scale<1e-6,torch.ones_like(scale),scale)
    return dict(mean=mean.tolist(),scale=scale.tolist(),dimensions=states.shape[1],
        fitting_scope='all collected TRAIN states only; no VAL/TEST fitting')


def score_reference(query, states):
    with torch.no_grad(): values = query(states)[0].detach().cpu().tolist()
    if not np.isfinite(values).all(): raise ValueError('Nonfinite TRAIN score calibration.')
    return values


def evaluate_levels(a,healthy,artifact,anchors,query_state,levels,split,calibration,save_root=None,schedule='policy'):
    a.budget_score_reference = calibration
    a.eval_query_schedule = schedule
    clean = pilot_domain(a,pilot.base.legacy.read_domain(a.burst_dir,artifact,0,split),703 if split=='val_ids' else 704)
    rows=[]
    for level in levels:
        pilot.base.legacy.core.set_seed(a.seed)
        live=runtime(a,healthy,artifact,query_state)
        domain=pilot_domain(a,pilot.base.legacy.read_domain(a.burst_dir,artifact,level,split),703 if split=='val_ids' else 704)
        _,result,trace=pilot.stream(a,live,domain,anchors,None,clean,a.seed,target_endpoints=save_root is not None)
        if schedule != 'uncertainty' and result['labels'] != result['cap']:
            raise RuntimeError('Fixed-quota policy failed to use the requested budget.')
        for key,value in healthy['policy_state_dict'].items():
            if not torch.equal(value.cpu(),live[2].policy.state_dict()[key].cpu()):
                raise RuntimeError('Prediction router changed in a Context-only evaluation.')
        result['level']=level
        if schedule == 'policy':
            result['budget_actor_deployment']=dict(
                forced_tail_labels=sum(int(row.get('budget_forced_tail',False) and row['queried']) for row in trace),
                rule='TRAIN-calibrated rank with remaining budget fraction; no online Actor updates')
        if getattr(a,'query_window_fraction',1.)<1:
            result['query_window_fraction']=a.query_window_fraction
            result['query_window_rows']=query_window_count(len(domain[1]),result['cap'],a.query_window_fraction)
            if any(row['queried'] and row['step']>result['query_window_rows'] for row in trace):
                raise RuntimeError('Query occurred after the early-window deadline.')
        rows.append(result)
        print_level_result(save_root.name if save_root is not None else 'VAL',result)
        if save_root is not None:
            pilot.write_csv(save_root/f'level_{level}_trace.csv',trace)
            pilot.write_json(save_root/f'level_{level}_metrics.json',result)
    return rows


def train(a,healthy,artifact,anchors,reference,teacher_hash):
    completed=a.output/'training_complete.json'
    if completed.exists():
        c=json.loads(completed.read_text(encoding='utf-8'))
        if c['protocol']!=protocol(a,teacher_hash) or c['checkpoint_sha256']!=pilot.digest(a.output/'Models'/'budget_actor.pth'):
            raise ValueError('Completed Actor settings/checkpoint differ; use a new output.')
        print('Using completed budget Actor training.',flush=True);return
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('Partial output preserved; use a fresh --output.')
    a.output.mkdir(parents=True,exist_ok=True)
    pilot.write_json(a.output/'protocol.json',protocol(a,teacher_hash))
    states,targets=collect(a,healthy,artifact,anchors,reference)
    spec=normalizer(states)
    a.actor_experiment=dict(version=1,control='natural',normalizer=spec,
        budget_actor='fixed-quota Actor-only training; Context checkpoint unchanged')
    pilot.base.legacy.core.set_seed(a.seed)
    device=pilot.base.legacy.core.device
    query=StandardizedQueryMaximizer(states.shape[1],spec).to(device)
    states,targets=states.to(device),targets.to(device)
    optimizer=torch.optim.Adam(query.actor.parameters(),lr=a.actor_lr)
    quota=a.budget_fraction
    if getattr(a,'utility_early',False):
        # Collection domains can be shorter than PilotRows; use actual row/cap counts.
        collections=[json.loads((a.output/f'collection_level{level}.json').read_text(encoding='utf-8')) for level in (0,1,3)]
        quota=float(np.mean([row['cap']/query_window_count(row['samples'],row['cap'],.2) for row in collections]))
    generator=torch.Generator(device='cpu').manual_seed(a.seed+901)
    rank_stages=[];rank_validation=None
    if getattr(a,'rank_warm_start',False):
        from enhancements.rank_warmstart import pairwise_rank_step
        rank_validation=collect_rank_validation(a,healthy,artifact,anchors)
        record_rank_stage(a,query,states,targets,rank_validation,quota,'initial',rank_stages)
        warm_optimizer=torch.optim.Adam(query.actor.parameters(),lr=a.actor_lr,weight_decay=1e-4)
        for warm_epoch in range(1,3):
            losses=[pairwise_rank_step(query,warm_optimizer,states,targets,a.ranking_groups)
                for _ in range(a.steps_per_epoch)]
            record_rank_stage(a,query,states,targets,rank_validation,quota,
                f'rank_warmup_{warm_epoch}',rank_stages,float(np.mean(losses)))
        # Warmup is initialization only. No pairwise loss is mixed into the max phase.
        optimizer=torch.optim.Adam(query.actor.parameters(),lr=a.actor_lr)
    best=-math.inf; log=[]
    for epoch in range(1,a.epochs+1):
        objectives=[]
        for _ in range(a.steps_per_epoch):
            at=torch.randperm(len(states),generator=generator)[:min(a.batch_size,len(states))].to(device)
            fit=quota_max_step(query,optimizer,states[at],targets[at],quota)
            objectives.append(fit['objective'])
        calibration=score_reference(query,states)
        validation=evaluate_levels(a,healthy,artifact,anchors,deepcopy(query.state_dict()),[0,1,3],'val_ids',calibration)
        ac=float(np.mean([r['online']['AC'] for r in validation]))
        drop=float(np.mean([r['original_domain_forgetting']['AC'] for r in validation]))
        selection=ac-max(0.,drop)
        log.append(dict(epoch=epoch,TRAIN_objective=float(np.mean(objectives)),validation_AC=ac,
            source_drop=drop,selection=selection,validation=validation))
        print(f'Actor epoch {epoch}/{a.epochs}: VAL AC {ac:.2f}% | source drop {drop:+.2f} pp',flush=True)
        if rank_validation is not None:
            record_rank_stage(a,query,states,targets,rank_validation,quota,f'quota_max_{epoch}',rank_stages)
        pilot.write_json(a.output/'training.json',dict(protocol=protocol(a,teacher_hash),epochs=log,
            signal_scales=getattr(a,'signal_scales',None),within_window_quota=quota,
            rank_stages=rank_stages,
            positive_targets=int((targets>0).sum()),targets=len(targets)))
        if selection>best:
            best=selection
            path=a.output/'Models';path.mkdir(exist_ok=True)
            torch.save(dict(version=VERSION,protocol=protocol(a,teacher_hash),query_state_dict=deepcopy(query.state_dict()),
                normalizer=spec,score_reference=calibration,selected_epoch=epoch,selected_validation=selection,
                teacher_checkpoint_sha256=teacher_hash,signal_scales=getattr(a,'signal_scales',None)),path/'budget_actor.pth')
    pilot.write_json(completed,dict(protocol=protocol(a,teacher_hash),checkpoint_sha256=pilot.digest(a.output/'Models'/'budget_actor.pth')))


def collect_rank_validation(a,healthy,artifact,anchors):
    """Fixed VAL ranking probes, never an Actor target or normalization fit."""
    source=pilot.base.legacy.read_domain(a.burst_dir,artifact,0,'val_ids')
    _,reference,reserved=pilot.reference_partition(source,a.seed)
    ref_keys=set(pilot.feature_keys(reference[0]))
    states,targets,groups,rows=[],[],[],[]
    a.collect_query_targets_only=True
    try:
        for level in (0,1,3):
            raw=pilot.base.legacy.read_domain(a.burst_dir,artifact,level,'val_ids')
            mask=np.asarray([int(i) not in reserved and key not in ref_keys
                for i,key in zip(raw[2],pilot.feature_keys(raw[0]))])
            domain=pilot_domain(a,pilot.subset(raw,mask),703)
            recorder=Recorder();pilot.base.legacy.core.set_seed(a.seed)
            live=runtime(a,healthy,artifact)
            before_query=deepcopy(live[3].state_dict());before_policy=deepcopy(live[2].policy.state_dict())
            print(f'Collect fixed offline VAL ranking probes: level {level}; no Actor fitting',flush=True)
            updated,_,_=pilot.stream(a,live,domain,anchors,reference,domain,a.seed+level*101,
                training=True,diagnostic_sink=recorder)
            for before,after in [(before_query,updated[3].state_dict()),(before_policy,updated[2].policy.state_dict())]:
                if any(not torch.equal(value,after[key]) for key,value in before.items()):
                    raise RuntimeError('VAL ranking observer changed a policy head.')
            measured=[row for row in recorder.rows if row.get('score') is not None]
            states.extend(recorder.states)
            for row in measured:
                row.update(level=level,ranking_target=scaled_competitive_score(row,a.signal_scales))
                targets.append(row['ranking_target']);groups.append(level);rows.append(row)
    finally:a.collect_query_targets_only=False
    pilot.write_csv(a.output/'VAL_ranking_observer.csv',rows)
    pilot.write_json(a.output/'VAL_ranking_protocol.json',dict(scope='offline full-information observer only',
        splits=['val_ids'],levels=[0,1,3],TRAIN_scales_frozen=True,
        Actor_fitting=False,normalizer_fitting=False,temporary_Context_updates='discarded',
        probe='independent source VAL reference; IDs and feature duplicates excluded from candidates',
        note='ranking oracles and probe labels are diagnostics, not budgeted deployment metrics'))
    device=pilot.base.legacy.core.device
    return torch.stack(states).to(device),torch.tensor(targets,dtype=torch.float32,device=device),groups


def record_rank_stage(a,query,states,targets,validation,quota,name,history,pairwise_loss=None):
    from enhancements.rank_warmstart import rank_metrics
    train_report=rank_metrics(query,states,targets,a.ranking_groups,quota)
    val_report=rank_metrics(query,*validation,quota)
    record=dict(stage=name,TRAIN=train_report,VAL=val_report)
    if pairwise_loss is not None:record['TRAIN_pairwise_loss']=pairwise_loss
    history.append(record)
    def show(report):return 'undefined' if report['spearman'] is None else f"{report['spearman']:+.3f}"
    print(f"RANK {name}: TRAIN Spearman {show(train_report)} | VAL Spearman {show(val_report)} | "
        f"TRAIN target regret {train_report['target_regret']:.4f} | VAL target regret {val_report['target_regret']:.4f}",flush=True)
    pilot.write_json(a.output/'ranking_stages.json',dict(stages=history,
        acceptance='inspect independent VAL ranking, then online VAL performance; TRAIN fit alone is insufficient'))
    return record


def protocol(a,teacher_hash):
    result=dict(version=VERSION,dataset=a.dataset,seed=a.seed,budget_fraction=a.budget_fraction,
        pilot_rows=a.pilot_rows,epochs=a.epochs,steps_per_epoch=a.steps_per_epoch,batch_size=a.batch_size,actor_lr=a.actor_lr,
        teacher_checkpoint_sha256=teacher_hash,context='healthy selected Context; unchanged initial weights and update function',
        prediction='Context -> frozen CNN; no prediction routing',
        objective='maximize mean(p*S), mean(p)=budget; unchanged signed S=gap+G-F-cost',
        interpretation='constrained local min-max surrogate on fixed TRAIN trajectories; not exact on-policy multistep equivalence',
        deployment='TRAIN-logit rank versus remaining budget fraction; randomized ties; forced tail if needed for exact cap',
        collection='uniform TRAIN feedback; copied Context/Adam states; no query/router training',
        selection='VAL online AC minus positive source drop; no TEST targets or selection',
        test_scope='small-data pilot' if a.pilot_rows else 'full TEST domains')
    if getattr(a,'utility_early',False):
        result.update(experiment='TRAIN-scaled competitive signals with an early query window',
            signal_scaling='positive standard deviation scales from measured TRAIN only; no centering or TEST fitting',
            query_window_fraction=.2,
            objective='maximize mean(p*(gap/sd_gap + weighted_gain/sd_gain - weighted_harm/sd_harm - cost)) under early-window quota',
            interpretation='reweighted constrained local min-max surrogate; Context descent unchanged up to positive gap scaling; not the original multistep objective',
            deployment='TRAIN-logit rank; exact label cap by first 20 percent of the stream; fixed Actor online')
    if getattr(a,'rank_warm_start',False):
        result.update(rank_warm_start=dict(epochs=2,steps_per_epoch=a.steps_per_epoch,
            loss='within-TRAIN-level pairwise softplus preference; initialization only',
            validation='fixed observer-only VAL targets; no fitting; TRAIN scales/normalizer frozen',
            continuation='reset Adam; original signed quota-max objective only; select within max phase on online VAL'))
    return result


def evaluate(a,healthy,artifact,anchors,teacher_hash):
    saved=torch.load(a.output/'Models'/'budget_actor.pth',map_location='cpu',weights_only=False)
    if saved['protocol']!=protocol(a,teacher_hash): raise ValueError('Actor checkpoint and evaluation settings differ.')
    a.actor_experiment=dict(version=1,control='natural',normalizer=saved['normalizer'])
    uniform_only=getattr(a,'uniform_control_only',False)
    methods=[('Uniform','uniform',a.budget_fraction)] if uniform_only else [
        ('Actor','policy',a.budget_fraction),('FrontLoaded','frontloaded',a.budget_fraction),
        ('Uncertainty','uncertainty',a.budget_fraction),('FrontLoaded20','frontloaded',.2)]
    if getattr(a,'utility_early',False):
        methods.extend([('UniformEarly','uniform',a.budget_fraction),('Uniform','uniform',a.budget_fraction)])
    root=a.output/('UniformResults' if uniform_only else 'Results');root.mkdir(exist_ok=True)
    execution=dict(protocol=saved['protocol'],checkpoint_sha256=pilot.digest(a.output/'Models'/'budget_actor.pth'),
        matched_controls=[item[0] for item in methods])
    if (root/'completed.json').exists():
        if json.loads((root/'execution.json').read_text(encoding='utf-8'))!=execution: raise ValueError('Evaluation differs from completed run.')
        completion=json.loads((root/'completed.json').read_text(encoding='utf-8'))
        for name,expected in completion['sha256'].items():
            if pilot.digest(root/name)!=expected: raise ValueError('Completed result changed: '+name)
        print('Using completed budget Actor TEST.',flush=True);return
    if (root/'execution.json').exists() and json.loads((root/'execution.json').read_text(encoding='utf-8'))!=execution:
        raise ValueError('Partial evaluation settings differ; output preserved.')
    pilot.write_json(root/'execution.json',execution)
    if uniform_only:
        return evaluate_uniform(a,healthy,artifact,anchors,saved,root,execution)
    a.uncertainty_percentile = 75.
    calibration_domains={level:pilot_domain(a,pilot.base.legacy.read_domain(a.burst_dir,artifact,level,'val_ids'),703)
        for level in (0,1,3)}
    pilot.base.legacy.core.set_seed(a.seed)
    calibration=pilot.calibrate_uncertainty(a,runtime(a,healthy,artifact,saved['query_state_dict']),calibration_domains)
    a.uncertainty_threshold=calibration['threshold']
    pilot.write_json(root/'uncertainty_calibration.json',calibration)
    summary=[]; files=['summary.csv','uncertainty_calibration.json']
    target_budget=a.budget_fraction
    for method,schedule,method_budget in methods:
        a.budget_fraction=method_budget
        a.query_window_fraction=.2 if getattr(a,'utility_early',False) and method in ('Actor','UniformEarly') else 1.
        directory=root/method;directory.mkdir(exist_ok=True)
        results=evaluate_levels(a,healthy,artifact,anchors,saved['query_state_dict'],LEVELS,'test_ids',
            saved['score_reference'],directory,schedule)
        for result in results:
            summary.append(dict(method=method,budget_fraction=method_budget,level=result['level'],**result['online'],labels=result['labels'],
                cap=result['cap'],labels_percent=result['labels_percent'],
                forced_tail_labels=result.get('budget_actor_deployment',{}).get('forced_tail_labels',0),
                target_final_AC=result['target_after']['AC'],
                source_initial_AC=result['source_before']['AC'],source_final_AC=result['source_after']['AC'],
                **{'forgetting_'+k:v for k,v in result['original_domain_forgetting'].items()}))
        files.extend(f'{method}/level_{level}_{suffix}' for level in LEVELS for suffix in ('metrics.json','trace.csv'))
    a.budget_fraction=target_budget
    a.query_window_fraction=.2 if getattr(a,'utility_early',False) else 1.
    pilot.write_csv(root/'summary.csv',summary)
    pilot.write_json(root/'completed.json',dict(levels=len(LEVELS),methods=len(methods),
        sha256={name:pilot.digest(root/name) for name in files}))
    print(f'Results: {root}',flush=True)


def evaluate_uniform(a,healthy,artifact,anchors,saved,root,execution):
    """Independent result files; same checkpoint, rows, reset and update as Actor."""
    directory=root/'Uniform';directory.mkdir(exist_ok=True)
    results=evaluate_levels(a,healthy,artifact,anchors,saved['query_state_dict'],LEVELS,'test_ids',
        saved['score_reference'],directory,'uniform')
    summary=[]
    for result in results:
        summary.append(dict(method='Uniform',budget_fraction=a.budget_fraction,level=result['level'],
            **result['online'],labels=result['labels'],cap=result['cap'],labels_percent=result['labels_percent'],
            forced_tail_labels=0,target_final_AC=result['target_after']['AC'],
            source_initial_AC=result['source_before']['AC'],source_final_AC=result['source_after']['AC'],
            **{'forgetting_'+k:v for k,v in result['original_domain_forgetting'].items()}))
    pilot.write_csv(root/'summary.csv',summary)
    files=['summary.csv']+[f'Uniform/level_{level}_{suffix}' for level in LEVELS
        for suffix in ('metrics.json','trace.csv')]
    pilot.write_json(root/'completed.json',dict(levels=len(LEVELS),methods=1,
        sha256={name:pilot.digest(root/name) for name in files}))
    print(f'Uniform control results: {root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--action',choices=['Train','Evaluate','All'],default='All')
    p.add_argument('--budget-percent',type=int,choices=[5,10],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--pilot-rows',type=int,default=300)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--steps-per-epoch',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--actor-lr',type=float,default=.0001)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--uniform-control-only',action='store_true',
        help='Evaluate evenly spaced feedback with an existing Actor checkpoint; no training.')
    p.add_argument('--utility-early',action='store_true',
        help='Experimental TRAIN-scaled reward components and first-20-percent query window.')
    p.add_argument('--rank-warm-start',action='store_true',
        help='TRAIN pairwise ranking initialization, followed by the unchanged UtilityEarly quota max.')
    a=p.parse_args()
    if a.rank_warm_start:a.utility_early=True
    if a.uniform_control_only and a.action!='Evaluate':
        p.error('Uniform control requires --action Evaluate; existing checkpoints are reused.')
    if a.uniform_control_only and a.utility_early:p.error('UtilityEarly already includes both uniform controls.')
    if a.pilot_rows<0 or (a.pilot_rows and a.pilot_rows<=16) or min(a.epochs,a.steps_per_epoch,a.batch_size)<2:
        p.error('Pilot rows must be zero or greater than 16; epochs/steps/batch-size must be at least two.')
    if not math.isfinite(a.actor_lr) or a.actor_lr<=0:p.error('Actor LR must be finite and positive.')
    root=Path(__file__).resolve().parent/f'ANT_Burst_{a.dataset}_TEST'
    a.teacher=a.teacher or root/'protocol_runs'/f'minimax_query_batch5_seed{a.seed}_v2_h16_c0.05_actorstd_natural'
    explicit_output=a.output is not None
    a.output=a.output or root/'protocol_runs'/f'budget_actor_q{a.budget_percent:02d}_seed{a.seed}_pilot{a.pilot_rows}_ep{a.epochs}'
    if a.utility_early and not explicit_output:
        a.output=a.output.with_name(a.output.name+'_scaled_early20')
    if a.rank_warm_start and not explicit_output:a.output=a.output.with_name(a.output.name+'_rankwarm')
    a.burst_dir=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    a.budget_fraction=a.budget_percent/100; a.progress_every=250
    a.query_window_fraction=.2 if a.utility_early else 1.
    if a.utility_early:
        print(f'UtilityEarly {a.dataset}: total label cap {a.budget_percent}% | query window first 20% | '
            'TRAIN-scaled gap/gain/harm | Context -> frozen CNN | batch5 | no CNN/initial Context retraining',flush=True)
    healthy,artifact,anchors,reference,teacher_hash=setup(a)
    if a.action in ('Train','All'):train(a,healthy,artifact,anchors,reference,teacher_hash)
    if a.action in ('Evaluate','All'):evaluate(a,healthy,artifact,anchors,teacher_hash)
    if a.action in ('Evaluate','All'):print_saved_summaries(a.output)
    if pilot.digest(a.teacher/'Models'/'minimax_query.pth')!=teacher_hash:raise RuntimeError('Healthy checkpoint was modified.')


if __name__=='__main__':main()
