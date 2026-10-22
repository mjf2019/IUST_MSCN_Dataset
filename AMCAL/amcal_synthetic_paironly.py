"""Fresh Burst PairOnly trained on clean level 0 and a label-blind synthetic drift."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import amcal_batch5_pipeline as pipeline
import amcal_final_enhancement as runner
from amcal_external_data_audit import excluded
from amcal_light_cnn import LightCNN1D, ARCHITECTURE

TEST=[3,5,7,10,12,15,17,20]

def filename(level):return f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'


def prepare(a):
    """Read level 0 only; retain source identities in synthetic TRAIN/VAL."""
    manifest=a.output_root/'synthetic_input.json'
    source=a.original_dir/filename(0)
    template=a.source_run/'Models/protocol_preprocessing.joblib'
    spec=dict(version=1,seed=a.seed,dataset=a.actual_dataset,source_sha256=pipeline.digest(source),
              template_sha256=pipeline.digest(template),gain=.25,bias=.3,noise=.1,
              train=[0,'synthetic drift from 0'],test=TEST)
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['spec']!=spec:raise ValueError('Synthetic inputs changed; use a new OutputRoot.')
        if not all((a.output_root/k).is_file() and pipeline.digest(a.output_root/k)==v for k,v in record['files'].items()):
            raise ValueError('Prepared synthetic inputs changed.')
        return
    if a.output_root.exists() and any(a.output_root.iterdir()):raise ValueError('Use an empty OutputRoot.')
    artifact=deepcopy(joblib.load(template))
    if any(excluded(n) for n in artifact['feature_names']):raise ValueError('Excluded feature in source schema.')
    frame,ids,_=runner.legacy.base.official.mapped_domain(source,artifact,a.original_dir,False)
    lookup={int(v):i for i,v in enumerate(ids)}
    train=np.array([lookup[int(i)] for i in artifact['train_ids']])
    columns=artifact['feature_names']
    scaler=StandardScaler().fit(frame.iloc[train][columns])
    artifact['scaler']=scaler
    a.burst_dir.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,a.burst_dir/filename(0))
    pd.DataFrame({'source_row':ids}).to_csv(a.burst_dir/(Path(filename(0)).stem+'_rows.csv'),index=False)
    chosen=np.concatenate([artifact['train_ids'],artifact['val_ids']]).astype(np.int64)
    if set(chosen)&set(artifact['test_ids']):raise ValueError('TRAIN/VAL overlaps TEST.')
    rows=np.array([lookup[int(i)] for i in chosen])
    raw=frame.iloc[rows][columns].to_numpy(dtype=float)
    z=scaler.transform(pd.DataFrame(raw,columns=columns))
    rng=np.random.default_rng(a.seed+71003)
    gains=rng.uniform(-.25,.25,len(columns));bias=rng.uniform(-.3,.3,len(columns))
    # Fixed feature-wise affine drift, increasing in four row-ID blocks, plus noise.
    severity=.5+(chosen % 4)/3
    shifted=z*(1+severity[:,None]*gains)+severity[:,None]*bias+rng.normal(0,.1,z.shape)
    shifted=scaler.inverse_transform(shifted)
    nonnegative=(frame.iloc[train][columns].min().to_numpy()>=0)
    shifted[:,nonnegative]=np.maximum(shifted[:,nonnegative],0)
    drift=pd.DataFrame(shifted,columns=columns)
    drift[artifact['label_name']]=frame.iloc[rows,-1].to_numpy()
    drift.to_csv(a.burst_dir/filename(1),index=False)
    pd.DataFrame({'source_row':chosen}).to_csv(a.burst_dir/(Path(filename(1)).stem+'_rows.csv'),index=False)
    joblib.dump(artifact,a.output_root/'input_preprocessing.joblib')
    paths=[*a.burst_dir.iterdir(),a.output_root/'input_preprocessing.joblib']
    runner.write_json(manifest,dict(spec=spec,
        drift='label-blind affine scale/shift + Gaussian noise in clean TRAIN scaler units; nonnegative features clipped at zero',
        synthetic_level_1='file-format placeholder, NOT official Burst level 1',
        inherited='feature schema and source partitions only; scaler refitted on clean TRAIN; no weights inherited',
        files={p.relative_to(a.output_root).as_posix():pipeline.digest(p) for p in paths}))


def fresh_source(a,output):
    core=runner.legacy.core
    artifact=joblib.load(a.output_root/'input_preprocessing.joblib')
    models=output/'Models';models.mkdir(parents=True)
    joblib.dump(artifact,models/'protocol_preprocessing.joblib')
    a.classes=list(artifact['label_encoder'].classes_)
    def data(split):
        x,y,_=runner.legacy.read_domain(a.burst_dir,artifact,0,split)
        return x,torch.tensor(y,dtype=torch.long,device=core.device)
    train,val=data('train_ids'),data('val_ids')
    width=len(artifact['feature_names']);classes=len(a.classes)
    cnn=(LightCNN1D(width,classes) if a.actual_dataset=='PS' else core.CNN1D(width,classes)).to(core.device)
    cnn_log=pipeline.fit(a,cnn,None,train,val)
    cnn.eval();cnn.requires_grad_(False)
    saved=dict(model_state_dict=cnn.state_dict(),preprocessing_sha256=pipeline.digest(models/'protocol_preprocessing.joblib'))
    if a.actual_dataset=='PS':saved.update(architecture=ARCHITECTURE,input_length=width,num_classes=classes)
    torch.save(saved,models/'initial_cnn_model.pth')
    context=core.ContextAwareNetwork(width).to(core.device)
    context_log=pipeline.fit(a,context,cnn,train,val,context=True)
    torch.save(dict(context_model_state_dict=context.state_dict(),preprocessing_sha256=saved['preprocessing_sha256']),models/'adversarial_best_context.pth')
    torch.save(dict(training_perturbations=[0,1],
        scope='fresh clean-0 source; synthetic-0 selector development; no inherited selector'),
        models/'adversarial_best_dqn.pth')
    runner.write_json(output/'source_training.json',dict(cnn=cnn_log,context=context_log,
        training_levels=[0],selection='clean level-0 VAL loss only',inherited_weights=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,choices=[1,3,5,7],default=5)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--source-run',type=Path);p.add_argument('--burst-dir',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if a.epochs<1:p.error('Epochs must be positive.')
    a.actual_dataset=a.dataset
    root=Path(f'AMCAL/ANT_Burst_{a.dataset}_TEST').resolve()
    a.source_run=(a.source_run or root/'protocol_runs'/('final_enhancement_accuracy_counterfactual_trainseed42' if a.dataset=='IAT' else 'structured_bank3_balancedref_lr0p0005_seed42')).resolve()
    a.original_dir=(a.burst_dir or root/f'Dataset_official_v1/AdvBurst_FTSC-{a.dataset}').resolve()
    a.output_root=(a.output_root or root/'protocol_runs'/f'paironly_synthetic0_batch{a.batch_size}_seed{a.seed}').resolve()
    a.burst_dir=a.output_root/'SyntheticDomains'
    core=runner.legacy.core;core.device=torch.device(a.device);core.set_seed(a.seed)
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable.')
    for level in TEST:
        test_path=a.original_dir/filename(level)
        if not test_path.is_file() or not test_path.with_name(test_path.stem+'_rows.csv').is_file():
            p.error('Missing TEST file or explicit row map: '+str(test_path))
    prepare(a)
    pipeline.TRAIN=[0,1];pipeline.TEST=TEST
    pipeline.fresh_ps_source=fresh_source
    # Reuse the complete existing structured -> accuracy -> utility -> PairOnly recipe.
    a.dataset='PS';a.patience=3;a.utility_epochs=2;a.cnn_epochs=20;a.context_epochs=5;a.evaluation_tag='synthetic0'
    stages,evaluation=pipeline.make_plan(a)
    if a.batch_size==1:
        for stage in stages:
            if '--paid-replay-mode' in stage['args']:
                stage['args'][stage['args'].index('--paid-replay-mode')+1]='off'
        evaluation[evaluation.index('--paid-replay-mode')+1]='off'
    plan=dict(seed=a.seed,batch_size=a.batch_size,epochs=a.epochs,stages=stages,evaluation=evaluation)
    plan_path=a.output_root/'synthetic_pipeline_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Recipe changed; choose a new OutputRoot.')
    else:runner.write_json(plan_path,plan)
    if a.action in ('All','Train'):
        for stage in stages:pipeline.execute(a,stage)
    elif not all(pipeline.stage_complete(stage) for stage in stages):raise ValueError('Training incomplete.')
    if a.action in ('All','Evaluate'):
        # Real attacks are opened only after all training/checkpoint selection completes.
        for level in TEST:
            src=a.original_dir/filename(level)
            mapping=src.with_name(src.stem+'_rows.csv')
            for path in (src,mapping):
                if not path.is_file():raise ValueError('Missing TEST domain/map: '+str(path))
                destination=a.burst_dir/path.name
                if destination.exists() and pipeline.digest(destination)!=pipeline.digest(path):raise ValueError('TEST input changed.')
                if not destination.exists():shutil.copy2(path,destination)
        runner.main(evaluation)
        runner.main([*evaluation,'--continual'])
    print('Synthetic PairOnly outputs:',a.output_root,flush=True)


if __name__=='__main__':main()
