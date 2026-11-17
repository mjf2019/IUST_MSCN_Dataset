"""Sequential retention of existing corrected Burst baselines; no offline refit.

Same final-AMCAL target/source observer populations and level order. Native
learners persist across domains. AF chains its corrected checkpoint batch
calibration with fresh per-domain optimizers and the author's feature reset.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace
import time

import amcal_final_multiseed as io
from final_sequential_metrics import LEVELS, METRICS, summarize
from final_sequential_baseline_budget import reset_domain_budget

METHODS = dict(RAL='ral',DFE='dfe',CALMID='calmid',MicFoal='micfoal',ARF='arf',AF=None)


def common_data(dataset, runs, output, device):
    import torch
    import amcal_two_action_paper_gain as base
    from enhancements.independent_target_holdout import split_targets
    plan = io.read(runs/'final_multiseed_fixedcnn_v1'/'plan.json')
    base.pilot.base.legacy.core.device = torch.device(device)
    a = SimpleNamespace(dataset=dataset,seed=42,device=device,
        pilot_rows=plan['train_rows_per_level'],beta=0.,budget_fraction=.2,
        burst_dir=io.ROOT/f'ANT_Burst_{dataset}_TEST'/'Dataset_official_v1'/f'AdvBurst_FTSC-{dataset}',
        teacher=runs/'minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_natural',
        actor_run=runs/'competitive_selector_seed42_p300_r2_v1_paper_reward',output=output)
    exp = base.PaperGainExperiment(a)
    exp.a.pilot_rows = 0
    domains,source = exp.split('test_ids',LEVELS)
    if exp.data_manifest('test_ids',LEVELS)!=plan['fixed_TEST_data']:
        raise ValueError('Canonical full TEST/source observers differ from final AMCAL.')
    streams,audits,split = split_targets(domains,42)
    def arrays(data):
        return data[0].squeeze(1).detach().cpu().numpy(),base.np.asarray(data[1]),base.np.asarray(data[2])
    data = dict(streams={k:arrays(v) for k,v in streams.items()},
                targets={k:arrays(v) for k,v in audits.items()},source=arrays(source))
    data.update(plan=plan,split=split,burst=a.burst_dir,artifact=exp.artifact,
                preprocessing_sha256=plan['source_sha256']['protocol_preprocessing.joblib'])
    return base,data


def native_source(runs,dataset,method):
    common = runs/('baselines_levels1_3_seed42' if dataset=='IAT' else 'baselines_levels1_3_rf10_seed42')
    if method=='RAL':
        return runs/'ral_paper_rf10_prefix_levels1_3_seed42' if dataset=='IAT' else common
    if method in ('CALMID','MicFoal'):
        return runs/'calmid_micfoal_levels1_3_seed42'
    if method=='ARF' and dataset=='IAT':
        return runs/'arf_levels1_3_shared_seed42'
    return common


class SequentialAF:
    def __init__(self,saved,sx,sy,cx,cy,classes,device,seed):
        self.saved = deepcopy(saved)
        self.sx,self.sy = sx,sy
        self.support_x,self.support_y = cx,cy
        self.classes,self.device,self.seed = classes,device,seed
        self.cumulative_paid = 0

    def predict(self,x):
        import amcal_af_clean_reference as af
        return af.predict(self.saved,self.support_x,self.support_y,x,self.classes,self.device)

    def calibrate(self,x,y,stage):
        from baselines.af_author_protocol import AFParameters,fit
        import amcal_af_clean_reference as af
        p = replace(AFParameters(**self.saved['parameters']),
                    target_pool_per_class=max(int((y==c).sum()) for c in self.classes))
        model,log = fit(self.sx,self.sy,x,y,self.classes,self.seed+stage*1009,
                        p,self.device,initial_state=self.saved['state'])
        self.saved = dict(state=af.cpu_state(model),parameters=af.asdict(p),
                          k=max(1,len(y)//len(self.classes)))
        self.support_x,self.support_y = x.copy(),y.copy()
        self.cumulative_paid += len(y)
        return log


def make_learner(args,data,seed):
    import numpy as np
    import torch
    import joblib
    import amcal_burst_baselines as b
    native = METHODS[args.method]
    if native is None:
        import amcal_af_clean_reference as af
        root = args.runs/f'AF_checkpoint20_online_seed{seed}_v2'
        checkpoint = root/'Models'/'af_clean_reference.pth'
        if not checkpoint.exists():
            raise FileNotFoundError('Restore/build the existing corrected AF clean checkpoint first: '+str(checkpoint))
        done = io.read(root/'training_complete.json'); recipe = io.read(root/'recipe.json')
        if (done['checkpoint_sha256']!=io.digest(checkpoint) or
                recipe['seed']!=seed or recipe['train_levels']!=[0] or
                recipe['clean_csv_sha256']!=io.digest(data['burst']/'AdvBurstTrainedBurst_size_0_Step_2000.csv')):
            raise ValueError('AF clean checkpoint provenance mismatch.')
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['preprocessing_sha256']!=data['preprocessing_sha256']:
            raise ValueError('AF preprocessing differs from common observers.')
        full = b.shared.read_domain(data['burst'],data['artifact'],0,'train_ids')
        x = full[0].squeeze(1).cpu().numpy(); y=np.asarray(full[1])
        index = {int(identity):i for i,identity in enumerate(full[2])}
        si = [index[int(i)] for i in recipe['source_ids']]
        ci = [index[int(i)] for i in recipe['calibration_ids']]
        if saved['support_ids']!=recipe['calibration_ids']:
            raise ValueError('AF clean support differs from saved recipe.')
        learner = SequentialAF(saved,x[si],y[si],x[ci],y[ci],
                               np.arange(len(data['artifact']['label_encoder'].classes_)),args.device,seed)
        provenance = dict(checkpoint=str(checkpoint),sha256=io.digest(checkpoint),
            recipe_sha256=io.digest(root/'recipe.json'),offline_training_levels=[0],
            extension='previous AF model state initializes next domain fit; source25 fixed; target support replaced by current paid prefix',
            preserved='corrected author DANN, feature-layer reset and supervised tuning; fresh optimizers per domain')
        return learner,provenance
    root = args.model_run or native_source(args.runs,args.dataset,args.method)
    checkpoint = root/'Models'/f'{native}.joblib'
    if not checkpoint.exists():
        raise FileNotFoundError('Restore existing baseline checkpoint; no automatic offline training: '+str(checkpoint))
    saved = joblib.load(checkpoint); p = saved['provenance']
    if (p['train_levels']!=[0,1,3] or
            p['source_preprocessing_sha256']!=data['preprocessing_sha256']):
        raise ValueError('Baseline training domains/preprocessing mismatch.')
    for level,sha in p['training_domains_sha256'].items():
        if io.digest(data['burst']/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')!=sha:
            raise ValueError('Baseline training files changed.')
    if native=='ral' and (saved.get('ral_protocol_version',0)<2 or saved.get('ral_trees')!=10):
        raise ValueError('Expected corrected full RAL RF10 checkpoint.')
    if native in ('arf','calmid','micfoal'):
        import river
        if river.__version__!=saved['river_version']:
            raise ValueError('River version differs from saved checkpoint; no automatic installation.')
        if native in ('calmid','micfoal') and saved.get('active_protocol_version')!=b.ACTIVE_VERSION:
            raise ValueError('Active baseline implementation version mismatch.')
    config = dict(p['config'])
    config.update(seed=seed,budget_fraction=.2,ral_update_window=0,ral_budget_mode='prefix',
                  device=args.device,progress_every=args.progress_every)
    learner = b.Learner(native,saved,SimpleNamespace(**config))
    return learner,dict(checkpoint=str(checkpoint),sha256=io.digest(checkpoint),
        offline_training_levels=p['train_levels'],source_model_seed=p['config']['seed'],
        version=saved.get('active_protocol_version',saved.get('ral_protocol_version')),
        extension='persistent fitted model, template/cache/feedback and detector state; reset only per-domain budget counters')


def observe(base,learner,data,stage,adapted_level):
    rows,predictions = [],[]
    for domain,part in [(0,data['source']),*sorted(data['targets'].items())]:
        x,y,ids = part
        predicted = learner.predict(x)
        rows.append(dict(stage=stage,adapted_level=adapted_level,domain=domain,samples=len(y),
                         **base.pilot.metrics(y,predicted,len(data['artifact']['label_encoder'].classes_))))
        predictions.extend(dict(stage=stage,domain=domain,source_row=int(i),truth=int(t),prediction=int(p))
                           for i,t,p in zip(ids,y,predicted))
    return rows,predictions


def stage_episode(args,base,learner,data,seed,stage,level):
    import numpy as np
    x,y,ids = data['streams'][level]
    order = np.random.default_rng(seed).permutation(len(y))
    x,y,ids = x[order],y[order],ids[order]
    n=len(y); cap=int(.2*n)
    if not 1<=cap<n:
        raise ValueError('Nonempty paid prefix and remaining stream required.')
    trace=[]; used=0
    started=time.perf_counter()
    if args.method=='AF':
        from amcal_af_clean_reference import prequential_batch_calibration
        log={}
        def adapt(selected):
            log.update(learner.calibrate(x[selected],y[selected],stage))
            return learner.predict,log
        prediction,log=prequential_batch_calibration(x,cap,learner.predict,adapt)
        used=cap
        for i,pred in enumerate(prediction):
            trace.append(dict(level=level,step=i+1,source_row=int(ids[i]),truth=int(y[i]),prediction=int(pred),
                queried=bool(i<cap),labels_used_before_prediction=min(i,cap),
                batch_update_after_prediction=bool(i==cap-1)))
    else:
        native=METHODS[args.method]
        reset_domain_budget(learner,native)
        predictions=[]
        schedule=set(np.random.default_rng(seed+819003).choice(n,cap,replace=False).tolist())
        for pos in range(n):
            prediction=int(learner.predict(x[pos:pos+1])[0])
            if native in ('ral','calmid','micfoal'):
                request,info=learner.request(x[pos],can_query=used<cap)
            else:
                request,info=(pos in schedule),None
            paid=bool(request and used<cap)
            if paid:
                learner.update(x[pos],int(y[pos]),info)
                used+=1
            predictions.append(prediction)
            trace.append(dict(level=level,step=pos+1,source_row=int(ids[pos]),truth=int(y[pos]),
                              prediction=prediction,queried=paid,labels_used=used))
            if args.progress_every and (pos+1)%args.progress_every==0:
                print(f'  {args.method} SEED={seed} LEVEL={level} {pos+1}/{n} LABELS={used}/{cap}',flush=True)
        prediction=np.asarray(predictions)
        log={}
    if used>cap or used!=sum(r['queried'] for r in trace):
        raise RuntimeError('Sequential baseline budget accounting failed.')
    result=dict(level=level,samples=n,labels=used,cap=cap,labels_percent=100*used/n,
        **base.pilot.metrics(y,prediction,len(data['artifact']['label_encoder'].classes_)),
        milliseconds_per_row=1000*(time.perf_counter()-started)/n)
    return result,trace,log


def evaluate_seed(args,base,data,seed):
    import joblib
    import numpy as np
    import torch
    learner,provenance=make_learner(args,data,seed)
    root=args.output/f'seed{seed}'
    execution=dict(version=1,dataset=args.dataset,method=args.method,seed=seed,
        seed_scope='AF seed-specific offline checkpoint plus evaluation; other baselines fixed source checkpoint plus evaluation seed',
        source=provenance,levels=LEVELS,budget_fraction_per_domain=.2,device=args.device,
        common_TEST=data['plan']['fixed_TEST_data'],target_split=data['split'],
        source_observer='same final-AMCAL original-domain TEST observer',
        observers='fixed across stages, methods and seeds; excluded by joint ID/feature grouping from all new target updates',
        historical_source='existing offline models and source support/cache retained; historical feature duplicates are not retrospectively removed',
        AF_scope='sequential extension of corrected checkpoint AF, not independent clean resets',
        evaluator_sha256=io.digest(Path(__file__)),
        metrics_sha256=io.digest(io.ROOT/'final_sequential_metrics.py'),
        budget_sha256=io.digest(io.ROOT/'final_sequential_baseline_budget.py'),
        baseline_sha256=io.digest(io.ROOT/'amcal_burst_baselines.py'),
        af_sha256=io.digest(io.ROOT/'baselines'/'af_author_protocol.py'))
    ef=root/'execution.json'
    if ef.exists() and io.read(ef)!=execution:
        raise ValueError('Sequential baseline recipe changed; preserve existing results.')
    io.write(ef,execution); fingerprint=io.digest(ef)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    initial=root/'initial_matrix.json'; initial_pred=root/'initial_observer_predictions.csv'
    initial_done=root/'initial_complete.json'
    if initial_done.exists():
        done=io.read(initial_done)
        if done['execution_sha256']!=fingerprint or any(io.digest(f)!=done['sha256'][f.name] for f in (initial,initial_pred)):
            raise ValueError('Initial observers changed.')
        matrix=io.read(initial)
    else:
        matrix,predictions=observe(base,learner,data,0,0)
        io.write(initial,matrix); io.csv_write(initial_pred,predictions)
        io.write(initial_done,dict(execution_sha256=fingerprint,
            sha256={f.name:io.digest(f) for f in (initial,initial_pred)}))
    online=[]
    for stage,level in enumerate(LEVELS,1):
        location=root/f'stage{stage}_level{level}'
        mf,tf,af,pf,sf,cf=[location/name for name in
            ('online_metrics.json','online_trace.csv','observer_matrix.json',
             'observer_predictions.csv','adaptive_state.joblib','completed.json')]
        files=(mf,tf,af,pf,sf)
        if cf.exists():
            done=io.read(cf)
            if done['execution_sha256']!=fingerprint or any(io.digest(f)!=done['sha256'][f.name] for f in files):
                raise ValueError('Completed sequential baseline stage changed.')
            state=joblib.load(sf)
            learner=state['learner']
            random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if state['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
            row,observed=io.read(mf),io.read(af)
            print(f'RESUMED {args.method} {args.dataset} SEED={seed} LEVEL={level}',flush=True)
        else:
            print(f'START SEQUENTIAL METHOD={args.method} DATASET={args.dataset} SEED={seed} '
                  f'STAGE={stage}/7 LEVEL={level} | retain previous adaptive model',flush=True)
            row,trace,log=stage_episode(args,base,learner,data,seed,stage,level)
            observed,predictions=observe(base,learner,data,stage,level)
            io.write(mf,row); io.csv_write(tf,trace); io.write(af,observed); io.csv_write(pf,predictions)
            if log: io.write(location/'calibration_log.json',log)
            state=dict(learner=learner,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            tmp=sf.with_suffix('.joblib.tmp')
            joblib.dump(state,tmp); tmp.replace(sf)
            io.write(cf,dict(execution_sha256=fingerprint,sha256={f.name:io.digest(f) for f in files}))
        online.append(row); matrix.extend(observed)
        print(f'STAGE COMPLETE METHOD={args.method} DATASET={args.dataset} SEED={seed} LEVEL={level} '
              f'ONLINE_AC={row["AC"]:.2f}% F1={row["F1"]:.2f}% LABELS={row["labels"]}/{row["cap"]}',flush=True)
        for r in observed:
            if r['domain']==0 or r['domain'] in LEVELS[:stage]:
                print(f'  RETEST AFTER_LEVEL={level} OBSERVER_DOMAIN={r["domain"]} '
                      f'AC={r["AC"]:.2f}% F1={r["F1"]:.2f}%',flush=True)
    if io.digest(Path(provenance['checkpoint']))!=provenance['sha256']:
        raise RuntimeError('Original baseline checkpoint changed.')
    summary=summarize(matrix,online)
    summary.update(dataset=args.dataset,method=args.method,seed=seed)
    io.write(root/'summary.json',summary); io.csv_write(root/'observer_matrix.csv',matrix)
    io.csv_write(root/'online_by_level.csv',online)
    io.write(root/'completed.json',dict(seed=seed,execution_sha256=fingerprint,
        summary_sha256=io.digest(root/'summary.json'),matrix_sha256=io.digest(root/'observer_matrix.csv')))
    print('SEED SUMMARY '+json.dumps(dict(method=args.method,dataset=args.dataset,seed=seed,**summary['AC'])),flush=True)


def collect(args):
    rows=[]
    for seed in args.seeds:
        root=args.output/f'seed{seed}'; done=io.read(root/'completed.json')
        if (done['summary_sha256']!=io.digest(root/'summary.json') or
                done['matrix_sha256']!=io.digest(root/'observer_matrix.csv') or
                done['execution_sha256']!=io.digest(root/'execution.json')):
            raise ValueError('Incomplete sequential baseline results.')
        summary=io.read(root/'summary.json')
        row=dict(dataset=args.dataset,method=args.method,seed=seed)
        for metric in METRICS: row.update({metric+'_'+k:v for k,v in summary[metric].items()})
        row['labels_percent']=summary['feedback']['labels_percent']
        rows.append(row)
    overall=dict(dataset=args.dataset,method=args.method,seeds=len(rows))
    for key in rows[0]:
        if key in ('dataset','method','seed'): continue
        values=[r[key] for r in rows]
        overall[key+'_mean']=sum(values)/len(values)
        overall[key+'_sd']=io.moments(values)[1] if len(values)>1 else None
    root=args.output/'Results'/('seeds_'+'_'.join(map(str,args.seeds)))
    io.csv_write(root/'summary_per_seed.csv',rows); io.csv_write(root/'summary_overall.csv',[overall])
    io.write(root/'completed.json',dict(seeds=args.seeds,method=args.method,overall=overall))
    print('ALL SEQUENTIAL SEEDS COMPLETE '+json.dumps(overall,sort_keys=True),flush=True)
    print('OUTPUT: '+str(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--method',choices=list(METHODS),required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,48,54,62,72])
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--model-run',type=Path)
    p.add_argument('--action',choices=['Evaluate','Collect'],default='Evaluate')
    p.add_argument('--progress-every',type=int,default=200)
    args=p.parse_args()
    if (not args.seeds or len(args.seeds)!=len(set(args.seeds)) or min(args.seeds)<0 or args.progress_every<0):
        p.error('Use distinct nonnegative seeds and a nonnegative progress interval.')
    if args.method=='AF' and args.model_run is not None:
        p.error('AF uses the existing per-seed AF_checkpoint20_online checkpoints.')
    args.runs=io.ROOT/f'ANT_Burst_{args.dataset}_TEST'/'protocol_runs'
    args.output=args.runs/'final_baseline_sequential_observers_v1'/args.method
    if args.action=='Evaluate':
        import torch
        import amcal_burst_baselines as b
        args.device=('cpu' if args.method in ('RAL','CALMID','MicFoal','ARF') else
                     ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
        if args.device=='cuda' and not torch.cuda.is_available():
            p.error('CUDA unavailable; use --device cpu.')
        b.core.device=torch.device(args.device)
        base,data=common_data(args.dataset,args.runs,args.output,args.device)
        for seed in args.seeds: evaluate_seed(args,base,data,seed)
    collect(args)


if __name__=='__main__':
    main()
