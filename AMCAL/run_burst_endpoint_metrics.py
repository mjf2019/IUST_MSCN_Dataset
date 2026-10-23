"""Per-method five-seed endpoint audits, preserving existing checkpoints/results."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
LEVELS = [5, 7, 10, 12, 15, 17, 20]
SEEDS = [42, 48, 54, 62, 72]
METHODS = {'AMCAL': 'amcal', 'CNN': 'cnn', 'CALMID': 'calmid', 'MICFOAL': 'micfoal',
           'RAL': 'ral5', 'AF': 'af_cal', 'DFE': 'dfe', 'ARF': 'arf'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def invoke(args):
    print('Running:', ' '.join(map(str,args)), flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=REPO,check=True)


def audited(m):
    same=m.get('same_level',{}).get('unqueried',{}).get('metric_pairs',{})
    ref=m.get('reference',{}).get('metric_pairs',{})
    metrics=('accuracy','macro_precision','macro_recall','macro_f1')
    return all(block.get('version')==1 and all(k in block.get('gain_pp',{}) and k in block.get('drop_pp',{})
               for k in metrics) for block in (same,ref))


def valid_online_audit(m,seed,level,n,method,history):
    if method!='arf':return history.valid_summary(m,seed,level,n,method=='ral5')
    e=m.get('execution_config',{})
    return (e.get('seed')==seed and e.get('budget')==1.0
            and not e.get('test_scaler') and not e.get('unseen_rows')
            and m.get('level')==level and m.get('samples')==n
            and m.get('label_cap')==n and m.get('labels')==n
            and m.get('training_provenance',{}).get('train_levels')==[0,1,3]
            and not e.get('ral_online_update_window'))


def find_audit(dataset,method,seed,level,history):
    root,runs,main,sources=history.locations(dataset)
    if method=='amcal':
        candidates=[]
        for ep in (main/'Results').glob('evaluation_*/execution.json'):
            e=read(ep)
            path=ep.with_name(f'learned_level{level}_metrics.json')
            if history.amcal_execution(e,seed) and path.exists() and audited(m:=read(path)):
                candidates.append((path,path.with_name(path.name.replace('_metrics.json','_trace.csv')),m))
    elif method=='af_cal':
        candidates=[]
        for path in runs.glob(f'af_endpoint_seed{seed}_*/af_fewshot/level{level}_shot5/metrics.json'):
            m=read(path)
            if (m.get('execution',{}).get('seed')==seed and m.get('shots')==5 and audited(m)
                and m.get('execution',{}).get('source_levels')==[0,1,3]):
                candidates.append((path,path.with_name('predictions.csv'),m))
    else:
        native='ral' if method=='ral5' else method
        folder=runs/f'ral_recent5_seed{seed}' if method=='ral5' else sources[method]
        candidates=[]
        for path in (folder/'Results').glob(f'{native}_per{level}_seed{seed}_*_metrics.json'):
            m=read(path)
            if (m.get('execution_config',{}).get('endpoint_metrics_version')==1 and audited(m)
                and valid_online_audit(m,seed,level,1073 if dataset=='IAT' else 886,method,history)):
                candidates.append((path,path.with_name(path.name.replace('_metrics.json','_trace.csv')),m))
    return max(candidates,key=lambda v:v[0].stat().st_mtime) if candidates else None


def run_seed(a,method,seed,history):
    root,runs,main,sources=history.locations(a.dataset)
    burst=root/'Dataset_official_v1'/f'AdvBurst_FTSC-{a.dataset}'
    pending=[level for level in a.levels if not find_audit(a.dataset,method,seed,level,history)]
    if not pending:
        print('Reusing complete endpoint audit:',a.dataset,a.method,seed,flush=True)
        return
    if method=='amcal':
        if not (main/'Models/final_enhancement.pth').exists():raise FileNotFoundError(main/'Models/final_enhancement.pth')
        invoke(['AMCAL/amcal_final_enhancement.py','evaluate','--output',main,'--burst-dir',burst,
            '--levels',*pending,'--seed',seed,'--budget-fraction','.2','--context-lr','.00025',
            '--context-count','1','--modes','learned','--query-schedule','policy','--update-policy','learned',
            '--paid-replay-mode','batch5','--query-budget-bonus','1','--online-memory-ce-weight','.1',
            '--evaluation-tag','final_five_seeds_bonus1_batch5_lr0p00025','--endpoint-metrics','--device',a.device])
    elif method=='af_cal':
        source=runs/'structured_bank3_policy_lr0p0005_seed42'
        out=runs/f'af_endpoint_seed{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'/'af_fewshot'
        invoke(['AMCAL/amcal_af_paper_burst.py','--source-run',source,'--burst-dir',burst,'--output',out,
            '--levels',*pending,'--source-levels','0','1','3','--shots','5','--seed',seed,'--endpoint-metrics','--device',a.device])
    else:
        if method=='ral5':
            # Same fixed offline RAL checkpoint as the existing recent5 experiments.
            # Reuse the existing seed42 model directory for all evaluation seeds;
            # evaluate_level deep-copies it and resets feedback for every level.
            folder=runs/'ral_recent5_seed42';native='ral'
            # Put new metrics beside each seed's original models where available.
            if (runs/f'ral_recent5_seed{seed}'/'Models/ral.joblib').exists():
                folder=runs/f'ral_recent5_seed{seed}'
        else:folder,native=sources[method],method
        if not (folder/'Models'/f'{native}.joblib').exists():raise FileNotFoundError(folder/'Models'/f'{native}.joblib')
        # If a fallback folder is used, find_audit must know where to read it.
        if method=='ral5' and folder.name!=f'ral_recent5_seed{seed}':
            raise FileNotFoundError(f'Missing original recent5 seed{seed} Models; restore/copy the original checkpoint first.')
        invoke(['AMCAL/amcal_burst_baselines.py','evaluate','--methods',native,'--output',folder,
            '--burst-dir',burst,'--levels',*pending,'--seed',seed,'--budget-fraction','1.0' if method=='arf' else '.2',
            '--endpoint-metrics','--device',a.device if method in ('cnn','dfe') else 'cpu',
            *(['--ral-budget-mode','prefix','--ral-update-window','5'] if method=='ral5' else [])])


def collect(a,method,history,before):
    from baselines.endpoint_metrics import METRICS
    _,runs,_,_=history.locations(a.dataset)
    bundle=runs/'paper_endpoint_metrics'/a.method
    entries=[]
    for seed in a.seeds:
        for level in a.levels:
            found=find_audit(a.dataset,method,seed,level,history)
            if not found:raise ValueError(f'Missing audit: {a.dataset}/{a.method}/{seed}/{level}')
            path,trace,m=found
            if not trace.exists():raise FileNotFoundError(trace)
            online=history.derived(trace,m)
            target=m['same_level']['unqueried']['metric_pairs'];reference=m['reference']['metric_pairs']
            if not reference['initial']['samples']:
                raise ValueError('Empty reference endpoint population.')
            if not target['initial']['samples'] and method!='arf':
                raise ValueError('Empty unqueried target endpoint population.')
            all_target=m['same_level']['all_rows']['metric_pairs']
            for pair in (target,reference):
                for k in METRICS:
                    if not pair['initial']['samples']:
                        if pair['gain_pp'][k] is not None or pair['drop_pp'][k] is not None:
                            raise ValueError('Empty endpoint population must have undefined changes.')
                        continue
                    value=100*(pair['final'][k]-pair['initial'][k])
                    if abs(pair['gain_pp'][k]-value)>1e-9 or abs(pair['drop_pp'][k]+value)>1e-9:
                        raise ValueError('Endpoint metric arithmetic mismatch.')
            previous=before.get((seed,level))
            comparison=None
            if previous:
                old=previous[2]
                comparison=dict(accuracy_delta=online['accuracy']-old['accuracy'],macro_f1_delta=online['macro_f1']-old['macro_f1'],
                    labels_match=m.get('labels')==old.get('labels'),prior_source=str(previous[0].relative_to(REPO)))
                if abs(comparison['accuracy_delta'])>1e-10 or abs(comparison['macro_f1_delta'])>1e-10 or not comparison['labels_match']:
                    print('NOTICE: repeated trajectory differs from previous result:',a.dataset,a.method,seed,level,comparison,flush=True)
            entry=dict(version=1,dataset=a.dataset,method=a.method,seed=seed,level=level,
                online=online,target_gain_pp=target['gain_pp'],reference_drop_pp=reference['drop_pp'],
                target_endpoint=target,reference_endpoint=reference,
                target_endpoint_all_rows=all_target,target_gain_all_rows_pp=all_target['gain_pp'],
                target_gain_scope='unqueried TEST subset; undefined when ARF acquires all stream labels',
                all_rows_gain_scope='retrospective full TEST stream, including acquired labels; not heldout generalization',
                endpoint_phase=m.get('endpoint_phase','whole online adaptation; initial checkpoint to end of stream'),
                source=str(path.relative_to(REPO)).replace('\\','/'),source_sha256=sha(path),
                trace=str(trace.relative_to(REPO)).replace('\\','/'),trace_sha256=sha(trace),
                prior_comparison=comparison,original=m)
            out=bundle/f'seed{seed}'/f'level{level}.json';out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(json.dumps(entry,indent=2,allow_nan=False),encoding='utf-8')
            if method=='af_cal':
                ep=path.with_name('endpoint_predictions.csv')
                if not ep.exists():raise FileNotFoundError(ep)
                import shutil
                shutil.copy2(ep,out.with_name(f'level{level}_endpoint_predictions.csv'))
            entries.append(dict(seed=seed,level=level,path=str(out.relative_to(REPO)).replace('\\','/')))
    manifest=dict(version=1,dataset=a.dataset,method=a.method,seeds=a.seeds,levels=a.levels,
        budget_fraction=None if method=='af_cal' else 1.0 if method=='arf' else .2,train_levels=[0,1,3],metrics=list(METRICS),entries=entries,
        protocol='fresh initial model per level; paired endpoint observers only; no transfer across levels',
        af_caveat='AF measures the supervised tuning stage, not whole-method source-only forgetting' if method=='af_cal' else None)
    bundle.mkdir(parents=True,exist_ok=True)
    (bundle/'coverage.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Push endpoint bundle:',bundle,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--method',choices=list(METHODS),required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--levels',type=int,nargs='+',default=LEVELS)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--action',choices=['run','collect'],default='run')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or not a.seeds or len(set(a.levels))!=len(a.levels) or any(l not in LEVELS for l in a.levels):
        p.error('Use distinct seeds and heldout levels from 5,7,10,12,15,17,20.')
    import run_burst_final_five_seeds as history
    method=METHODS[a.method]
    before={(seed,level):history.find(a.dataset,method,seed,level) for seed in a.seeds for level in a.levels}
    if method=='af_cal':
        before={key:(value if value and value[2].get('execution',{}).get('source_levels')==[0,1,3] else None)
                for key,value in before.items()}
    if method=='arf':
        before={key:(value if value and value[2].get('execution_config',{}).get('budget')==1.0 else None)
                for key,value in before.items()}
    if a.action=='run':
        for seed in a.seeds:run_seed(a,method,seed,history)
    collect(a,method,history,before)


if __name__=='__main__':main()
