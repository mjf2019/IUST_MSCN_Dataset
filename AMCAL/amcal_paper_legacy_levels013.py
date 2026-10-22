"""Data-only adapter to the existing approved binary AMCAL implementation."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


LEVELS=(0,1,3)
TEST_LEVELS=(5,7,10,12,15,17,20)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_summary(path, level):
    trace=pd.read_csv(path)
    if trace.empty: raise ValueError('Empty legacy evaluation trace.')
    truth=trace['true_label'].to_numpy()
    return dict(level=level,samples=len(trace),
        accuracy_before_feedback_pct=100*float(np.mean(truth==trace['prediction_before_update'].to_numpy())),
        accuracy_after_feedback_pct=100*float(np.mean(truth==trace['prediction_after_update'].to_numpy())),
        base_accuracy_pct=100*float(np.mean(truth==trace['base_prediction'].to_numpy())),
        labels=int(trace['labels_used'].max()))


def prepare(burst, source, output):
    artifact=joblib.load(source/'Models'/'protocol_preprocessing.joblib')
    if sha(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv')!=artifact['source_sha256']:
        raise ValueError('Source data differ from the saved partition provenance.')
    pools=[set(map(int,artifact[k])) for k in ('train_ids','val_ids','test_ids')]
    if any(not p for p in pools) or any(pools[i]&pools[j] for i in range(3) for j in range(i)):
        raise ValueError('Source TRAIN/VAL/TEST partitions must be nonempty and disjoint.')
    frames={}
    for level in (*LEVELS,*TEST_LEVELS):
        path=burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
        frame=pd.read_csv(path); mapping=pd.read_csv(path.with_name(path.stem+'_rows.csv'))
        raw=pd.to_numeric(mapping['source_row'],errors='raise').to_numpy()
        if len(raw)!=len(frame) or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
            raise ValueError('Invalid source-row map.')
        ids=raw.astype(np.int64)
        if len(set(ids))!=len(ids) or (ids<0).any() or (ids>=artifact['source_rows']).any():
            raise ValueError('Duplicate or out-of-range source IDs.')
        if list(frame.columns)!=artifact['feature_names']+[artifact['label_name']]:
            raise ValueError('Feature schema mismatch.')
        if not np.array_equal(frame.iloc[:,-1].to_numpy(),artifact['source_labels'][ids]):
            raise ValueError('Source map contradicts saved labels.')
        frames[level]=(frame,ids)
    def split(level,key):
        frame,ids=frames[level]
        wanted=set(map(int,artifact[key])); mask=np.array([int(i) in wanted for i in ids])
        if set(map(int,ids[mask]))!=wanted: raise ValueError('A source partition is incomplete.')
        return frame.loc[mask].reset_index(drop=True)
    data=output/'PreparedData'; data.mkdir(parents=True,exist_ok=False)
    split(0,'train_ids').to_csv(data/'oversampled_train_dataset.csv',index=False)
    split(0,'val_ids').to_csv(data/'oversampled_validation_dataset.csv',index=False)
    # Legacy training prints a test observer. Supply VAL here; actual TEST stays unread by training.
    split(0,'val_ids').to_csv(data/'oversampled_test_dataset.csv',index=False)
    joint_train=pd.concat([split(k,'train_ids') for k in LEVELS],ignore_index=True)
    joint_val=pd.concat([split(k,'val_ids') for k in LEVELS],ignore_index=True)
    joint_train.to_csv(data/'context_train_levels013.csv',index=False)
    joint_val.to_csv(data/'context_validation_levels013.csv',index=False)
    attacks=data/'AdvBurst_FTSC-IAT'; attacks.mkdir()
    for k in TEST_LEVELS:
        split(k,'test_ids').to_csv(attacks/f'AdvBurstTrainedBurst_size_{k}_Step_2000.csv',index=False)
    manifest=dict(reference='mjf2019/AMCAL-paper main/elsarticle/AMCAL.tex',
        implementation='AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py',
        base_training_levels=[0],context_selector_training_levels=list(LEVELS),
        train_ids=sorted(pools[0]),val_ids=sorted(pools[1]),test_ids=sorted(pools[2]),
        training_test_observer='VAL placeholder; actual TEST evaluated separately',
        extension='three-domain Context/selector training; original paper trains on level 0',
        source_partition_sha256=sha(source/'Models'/'protocol_preprocessing.joblib'),
        source_data_sha256={str(k):sha(burst/f'AdvBurstTrainedBurst_size_{k}_Step_2000.csv') for k in frames})
    (output/'legacy_data_protocol.json').write_text(json.dumps(manifest,indent=2))
    return data


@contextmanager
def joint_adaptation_data(legacy, data):
    """Change only loaders passed to the already implemented training routine."""
    original=legacy.train_adversarial_system
    def invoke(cnn,agent,context,full_loader,val_loader,counts,width,classes,*args,**kwargs):
        _,encoder,scaler,_,_=legacy.load_metadata()
        def dataset(name):
            frame=pd.read_csv(data/name)
            x=torch.tensor(scaler.transform(frame.iloc[:,:-1].to_numpy()),dtype=torch.float32,
                           device=legacy.device).unsqueeze(1)
            y=torch.tensor(encoder.transform(frame.iloc[:,-1].to_numpy()),device=legacy.device)
            return TensorDataset(x,y)
        train=dataset('context_train_levels013.csv'); val=dataset('context_validation_levels013.csv')
        y=train.tensors[1].cpu().numpy(); group=np.bincount(y,minlength=classes)
        sampler=WeightedRandomSampler(1./group[y],len(y),replacement=True)
        return original(cnn,agent,context,DataLoader(train,batch_size=128,sampler=sampler),
            DataLoader(val,batch_size=128,shuffle=False),counts,width,classes,*args,**kwargs)
    legacy.train_adversarial_system=invoke
    try: yield
    finally: legacy.train_adversarial_system=original


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--burst-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--base-epochs',type=int,default=200)
    p.add_argument('--context-epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=30)
    a=p.parse_args(argv)
    output=a.output.resolve(); burst=a.burst_dir.resolve(); source=a.source_run.resolve()
    if a.command=='train' and output.exists(): raise ValueError('Use a fresh output directory.')
    sys.path.insert(0,str(Path(__file__).resolve().parent/'ANT_Burst_TEST'))
    import amcal_burst_legacy as legacy
    data=prepare(burst,source,output) if a.command=='train' else output/'PreparedData'
    if not data.is_dir(): raise ValueError('Prepared data not found; train this run first.')
    settings=['--data-dir',str(data),'--output',str(output),'--seed',str(a.seed),
        '--reward-mode','paper','--transition-mode','local','--gamma','0.95',
        '--context-loss-mode','aligned','--training-selector-mode','original','--selection-cost','0']
    before=sha(Path(legacy.__file__)); old_argv=sys.argv[:]; old_cwd=Path.cwd()
    if a.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable.')
    legacy.device=torch.device('cuda' if a.device=='cuda' or a.device=='auto' and torch.cuda.is_available() else 'cpu')
    try:
        if a.command=='train':
            sys.argv=[legacy.__file__,'train',*settings,'--base-epochs',str(a.base_epochs),
                '--adversarial-epochs',str(a.context_epochs),'--train-patience',str(a.patience)]
            with joint_adaptation_data(legacy,data): legacy.cli()
        else:
            rows=len(pd.read_csv(data/'AdvBurst_FTSC-IAT'/'AdvBurstTrainedBurst_size_5_Step_2000.csv'))
            if rows>1073: raise ValueError('Legacy evaluator supports at most 1073 rows; do not silently truncate.')
            summaries=[]
            for level in TEST_LEVELS:
                if len(pd.read_csv(data/'AdvBurst_FTSC-IAT'/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'))!=rows:
                    raise ValueError('Legacy evaluation domains must share the same source partition.')
                sys.argv=[legacy.__file__,'evaluate',*settings,'--per',str(level),'--selector-mode','learned',
                    '--query-gate','none','--budget',str(int(.2*rows)),
                    '--epsilon-start','0.05','--epsilon-end','0.05','--epsilon-decay','1',
                    '--online-context-lr','0.0005','--stream-order','shuffle','--stream-seed',str(a.seed),
                    '--forgetting-diagnostics']
                legacy.cli()
                candidates=list((output/'Results').rglob(f'legacy_PER_{level}_trace.csv'))
                if len(candidates)!=1: raise RuntimeError('Missing or ambiguous legacy trace after evaluation.')
                summaries.append(trace_summary(candidates[0],level))
            pd.DataFrame(summaries).to_csv(output/'Results'/'legacy_levels_summary.csv',index=False)
    finally:
        sys.argv=old_argv; os.chdir(old_cwd)
        if sha(Path(legacy.__file__))!=before: raise RuntimeError('Legacy implementation was modified.')
    expected=('initial_cnn_model.pth','adversarial_best_context.pth','adversarial_best_dqn.pth')
    if any(not (output/'Models'/name).is_file() for name in expected):
        raise RuntimeError('Legacy routine did not produce all three required model files.')
    if a.command=='train':
        summaries={}
        for name in expected:
            saved=torch.load(output/'Models'/name,map_location='cpu',weights_only=False)
            summaries[name]={k:v for k,v in saved.items() if v is None or isinstance(v,(str,int,float,bool))}
        (output/'legacy_training_summary.json').write_text(json.dumps(summaries,indent=2))
    report=output/'legacy_execution.json'
    existing=json.loads(report.read_text()) if report.exists() else {}
    existing[a.command]=dict(implementation_sha256=before,seed=a.seed,source_main='elsarticle/AMCAL.tex',
        reward='paper',transition='local',gamma=.95,selection_cost=0,query_gate='none',
        training='base on 0; Context and selector on combined 0,1,3; no episodic redesign',
        evaluation='independent levels; same-level forgetting diagnostics; no persistent sequential evaluation')
    report.write_text(json.dumps(existing,indent=2))


if __name__=='__main__': main()
