"""Use existing Dense-generated attack CSVs; no attack generation or downloads.

The historical seed-42 stratified 80/20 attack-pool row order is ASSUMED,
not established by matching label sequences. This assumption is recorded.
Clean and shifted rows share source partitions; only TRAIN IDs enter training.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baselines.adversarial_domains import LEVELS, digest, domain_name
from amcal_adversarial_protocol import arguments, write_csv, summary_rows, display

VERSION = 'supplied_dense_attacks_v1'


def source_pools(frame):
    """Historical attack source membership is an explicit reproducibility assumption."""
    rows=np.arange(len(frame))
    development,pool=train_test_split(rows,test_size=.2,random_state=42,stratify=frame.iloc[:,-1])
    train,val=train_test_split(development,test_size=.25,random_state=42,stratify=frame.iloc[development,-1])
    attack_train,remaining=train_test_split(pool,test_size=.4,random_state=42,stratify=frame.iloc[pool,-1])
    attack_val,test=train_test_split(remaining,test_size=.5,random_state=42,stratify=frame.iloc[remaining,-1])
    return pool,dict(train=np.concatenate([train,attack_train]).tolist(),
                     validation=np.concatenate([val,attack_val]).tolist(),test=test.tolist())


def supplied_path(root,attack,level):
    radius=LEVELS[attack][level-1]
    matches=[p for p in (root/attack).glob(f'{attack}_eps_*.csv')
             if float(p.stem.split('_eps_')[-1])==radius]
    if len(matches)!=1:raise ValueError(f'Expected one supplied {attack} level {level} CSV.')
    return matches[0]


def prepare(a):
    if a.output.exists():raise ValueError('Use a fresh output directory: '+str(a.output))
    source=a.data_dir/a.source_file
    frame=pd.read_csv(source)
    pool,partitions=source_pools(frame)
    roots=a.output/'Domains'
    roots.mkdir(parents=True)
    shutil.copy2(source,roots/domain_name(0))
    pd.DataFrame({'source_row':np.arange(len(frame))}).to_csv(roots/(Path(domain_name(0)).stem+'_rows.csv'),index=False)
    records={}
    # Preserve the supplied files byte-for-byte; maps document the historical assumption.
    for level in range(1,6):
        path=supplied_path(a.data_dir,a.attack,level)
        target=roots/domain_name(level)
        shutil.copy2(path,target)
        pd.DataFrame({'source_row':pool}).to_csv(target.with_name(target.stem+'_rows.csv'),index=False)
        records[str(level)]=dict(input=str(path),sha256=digest(target))
    (roots/'official_partitions.json').write_text(json.dumps({**partitions,'allow_partial_domains':True,
        'alignment_assumption':'historical seed-42 stratified 80/20 attack-pool row order; unverified'},indent=2))
    manifest=dict(version=VERSION,attack=a.attack,source_sha256=digest(source),domains=records,
                  training_levels=[0,2],status='training',seed=a.seed,
                  alignment='ASSUMED historical stratified 80/20 split, random_state=42; not confirmed attack-generation provenance',
                  source_partitions=partitions,attack_surrogate='generic Dense model, as specified by dataset owner',
                  attack_generation=False)
    (a.output/'supplied_attack_training.json').write_text(json.dumps(manifest,indent=2))
    return manifest


@contextmanager
def working_directory(root):
    old=Path.cwd()
    try:
        os.chdir(root)
        yield
    finally:os.chdir(old)


def train(a):
    manifest=prepare(a)
    roots=a.output/'Domains'
    import amcal_burst_two_stage_protocol as two
    # Reuse the exact two-stage training engine. Supplied domains may cover
    # only a subset of clean source IDs; each episode keeps its own labels.
    with arguments(['train','--burst-dir',roots,'--burst-mapping-dir',roots,
                    '--official-split-dir',roots,'--output',a.output/'Source',
                    '--multishift-levels',0,2,'--stage1-rl-epochs',a.stage1_epochs,
                    '--cnn-multishift-epochs',a.cnn_epochs,'--cnn-multishift-patience',10,
                    '--stage2-pretrain-epochs',20,'--base-epochs',a.base_epochs,
                    '--base-patience',15,'--pretrain-epochs',20,
                    '--context-epochs',a.source_epochs,'--context-patience',10,
                    '--train-online-lr',.0001,'--budget-fraction',a.budget_fraction,
                    '--selection-cost',.0001,'--routing-reward-mode','absolute',
                    '--gain-weight',1,'--retention-weight',1,'--seed',a.seed,'--device',a.device]):
        two.main()
    import amcal_burst_structured_bank_protocol as bank
    with arguments(['train','--source-run',a.output/'Source','--burst-dir',roots,
                    '--output',a.output/'Bank','--levels',0,2,'--balanced-reward-reference',
                    '--epochs',a.bank_epochs,'--patience',5,'--router-warmup-episodes',3,
                    '--context-lr',a.context_lr,'--budget-fraction',a.budget_fraction,
                    '--query-schedule','policy','--update-policy','learned',
                    '--gamma',.95,'--selection-cost',.0001,'--gain-weight',1,'--retention-weight',1,
                    '--train-epsilon',.2,'--eval-epsilon',0,'--seed',a.seed,'--device',a.device]):
        bank.main()
    manifest['status']='complete'
    (a.output/'supplied_attack_training.json').write_text(json.dumps(manifest,indent=2))
    print('Supplied-data training complete. No heldout test scored.',flush=True)


def check(a):
    manifest=json.loads((a.output/'supplied_attack_training.json').read_text())
    if manifest['version']!=VERSION or manifest['attack']!=a.attack or manifest['status']!='complete':
        raise ValueError('Complete matching supplied-data AMCAL training required.')
    roots=a.output/'Domains'
    if digest(roots/domain_name(0))!=manifest['source_sha256']:raise ValueError('Clean CSV changed.')
    for level,record in manifest['domains'].items():
        if digest(roots/domain_name(int(level)))!=record['sha256']:raise ValueError('Supplied CSV changed.')
    return roots


def evaluate(a):
    roots=check(a)
    import amcal_burst_structured_bank_protocol as bank
    original=bank.evaluate
    metrics=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        metrics.extend(result)
        return result
    bank.evaluate=capture
    try:
        with arguments(['evaluate','--burst-dir',roots,'--output',a.output/'Bank',
                        '--levels',*a.levels,'--include-seen-levels','--full-unseen-levels','--balanced-reward-reference',
                        '--context-lr',a.context_lr,'--budget-fraction',a.budget_fraction,
                        '--query-schedule','policy','--update-policy','learned',
                        '--gamma',.95,'--selection-cost',.0001,'--gain-weight',1,'--retention-weight',1,
                        '--eval-epsilon',a.eval_epsilon,'--modes','learned','random',
                        '--seed',a.seed,'--device',a.device,'--test-scaler',a.test_scaler]):bank.main()
    finally:bank.evaluate=original
    rows=summary_rows(a.attack,metrics)
    target=a.output/'Results'/(f'supplied_summary_seed{a.seed}'+('_scalernotebook' if a.test_scaler=='notebook' else '')+'.csv')
    write_csv(target,rows)
    display(rows)
    print('Saved:',target,flush=True)


def independent_baseline_source(a):
    """Same clean/multishift CNN stages and split, without Context or RL training."""
    manifest=prepare(a)
    roots=a.output/'Domains'
    source=a.output/'BaselineSource'
    source.mkdir()
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from sklearn.preprocessing import StandardScaler,LabelEncoder
    import amcal_burst_structured_bank_protocol as bank
    core=bank.core
    core.device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    core.RUN_SEED=a.seed
    core.TRAIN_PERTURBATIONS=(0.,2.)
    core.set_seed(a.seed)
    frame=pd.read_csv(roots/domain_name(0))
    groups=core.feature_groups(frame);seen=set();filtered=[]
    for key in ('train','validation','test'):
        ids=np.asarray(manifest['source_partitions'][key],dtype=np.int64)
        keep=np.array([i for i in ids if groups[i] not in seen],dtype=np.int64)
        filtered.append(keep);seen.update(groups[i] for i in ids)
    if min(map(len,filtered))==0:raise ValueError('Empty partition after exact-feature overlap removal.')
    train_ids,val_ids,test_ids=filtered
    train=frame.iloc[train_ids]
    encoder=LabelEncoder().fit(train.iloc[:,-1])
    scaler=StandardScaler().fit(train.iloc[:,:-1])
    counts=np.bincount(encoder.transform(train.iloc[:,-1]),minlength=len(encoder.classes_))
    with working_directory(source):
        core.save_preprocessing(frame,str(roots),domain_name(0),scaler,encoder,counts,train_ids,val_ids,test_ids)
        artifact_path=source/'Models/protocol_preprocessing.joblib'
        artifact=joblib.load(artifact_path)
        artifact['allow_partial_domains']=True
        artifact['supplied_domain_alignment']=manifest['alignment']
        joblib.dump(artifact,artifact_path)
        def loader(levels,split,shuffle):
            values=[bank.read_domain(roots,artifact,level,split) for level in levels]
            x=torch.cat([value[0] for value in values])
            y=torch.as_tensor(np.concatenate([value[1] for value in values]),dtype=torch.long,device=core.device)
            return DataLoader(TensorDataset(x,y),batch_size=64,shuffle=shuffle)
        if 'cnn' in a.methods:
            cnn=core.CNN1D(len(artifact['feature_names']),len(encoder.classes_)).to(core.device)
            print('Independent CNN stage 1: clean TRAIN; validation-only selection.',flush=True)
            cnn=core.train_initial_cnn(cnn,loader([0],'train_ids',True),loader([0],'val_ids',False),counts,a.base_epochs,15)
            print('Independent CNN stage 2: clean + level-2 TRAIN; no Context/RL dependency.',flush=True)
            cnn=core.train_initial_cnn(cnn,loader([0,2],'train_ids',True),loader([0,2],'val_ids',False),counts,a.cnn_epochs,10)
            torch.save(dict(model_state_dict=cnn.state_dict(),input_length=len(artifact['feature_names']),
                            num_classes=len(encoder.classes_),training_levels=[0,2],
                            preprocessing_sha256=digest(artifact_path)),source/'Models/initial_cnn_model.pth')
        else:print('CNN benchmark omitted: prepare shared preprocessing only; use AMCAL Frozen results.',flush=True)
    names=['protocol_preprocessing.joblib']+(['initial_cnn_model.pth'] if 'cnn' in a.methods else [])
    record=dict(training_levels=[0,2],independent_cnn='cnn' in a.methods,seed=a.seed,
                model_hashes={name:digest(source/'Models'/name) for name in names},
                cnn_stages=['clean','clean + level 2'] if 'cnn' in a.methods else [],test_used=False)
    (source/'baseline_source.json').write_text(json.dumps(record,indent=2))
    manifest.update(status='complete',purpose='independent_baselines',baseline_source='BaselineSource')
    (a.output/'supplied_attack_training.json').write_text(json.dumps(manifest,indent=2))
    return roots


def baseline(a):
    if a.command=='train-baselines' and a.independent_baselines:
        roots=independent_baseline_source(a)
    else:roots=check(a)
    manifest=json.loads((a.output/'supplied_attack_training.json').read_text())
    independent=manifest.get('purpose')=='independent_baselines'
    source=a.output/('BaselineSource' if independent else 'Bank')
    command='train' if a.command=='train-baselines' else 'evaluate'
    values=[sys.executable,str(Path(__file__).with_name('amcal_burst_baselines.py')),
            command,'--source-run',str(source),'--burst-dir',str(roots),
            '--output',str(a.output/'Baselines'),'--train-levels','0','2',
            '--methods',*a.methods,'--budget-fraction',str(a.budget_fraction),
            '--seed',str(a.seed),'--device',a.device]
    if independent and command=='train':values+=['--independent-source']
    if command=='evaluate':values+=['--include-seen-levels','--full-unseen-levels','--levels',*map(str,a.levels),'--test-scaler',a.test_scaler]
    subprocess.run(values,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('train','evaluate','train-baselines','evaluate-baselines'))
    p.add_argument('--independent-baselines',action='store_true',help='Train baselines in their own fresh output, with an independent CNN and no dependency on AMCAL completion.')
    p.add_argument('--data-dir',type=Path,default=Path('AMCAL/SDNCampus_TEST/Dataset'))
    p.add_argument('--source-file',default='SDNCampus_original.csv')
    p.add_argument('--attack',choices=('all',*LEVELS),default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--levels',nargs='+',type=int,default=list(range(6)))
    p.add_argument('--methods',nargs='+',choices=('cnn','ral','af','dfe','arf','calmid','micfoal'),default=['cnn','ral','af','dfe','arf','calmid','micfoal'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--context-lr',type=float,default=.0005)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--eval-epsilon',type=float,default=.05)
    p.add_argument('--test-scaler',choices=('training','notebook'),default='training')
    p.add_argument('--base-epochs',type=int,default=100)
    p.add_argument('--stage1-epochs',type=int,default=12)
    p.add_argument('--cnn-epochs',type=int,default=30)
    p.add_argument('--source-epochs',type=int,default=30)
    p.add_argument('--bank-epochs',type=int,default=12)
    a=p.parse_args()
    a.output,a.data_dir=a.output.resolve(),a.data_dir.resolve()
    if any(v not in range(6) for v in a.levels) or len(set(a.levels))!=len(a.levels):p.error('Distinct levels 0..5 required.')
    if not 0<=a.budget_fraction<=1 or not 0<=a.eval_epsilon<=1 or not np.isfinite(a.context_lr) or a.context_lr<=0:p.error('Invalid budget, epsilon or LR.')
    if min(a.base_epochs,a.stage1_epochs,a.cnn_epochs,a.source_epochs,a.bank_epochs)<1:p.error('Epochs must be positive.')
    if a.test_scaler!='training' and a.command not in ('evaluate','evaluate-baselines'):p.error('--test-scaler notebook is evaluation only.')
    if a.independent_baselines and a.command not in ('train-baselines','evaluate-baselines'):p.error('--independent-baselines is baseline-only.')
    if a.attack=='all':
        for attack in LEVELS:
            values=[sys.executable,str(Path(__file__).resolve()),a.command,
                    '--attack',attack,'--data-dir',str(a.data_dir),'--source-file',a.source_file,
                    '--output',str(a.output/attack),'--seed',str(a.seed),'--device',a.device,
                    '--levels',*map(str,a.levels),'--methods',*a.methods]
            for key in ('context_lr','budget_fraction','eval_epsilon','base_epochs','stage1_epochs','cnn_epochs','source_epochs','bank_epochs'):
                values+=['--'+key.replace('_','-'),str(getattr(a,key))]
            values+=['--test-scaler',a.test_scaler]
            if a.independent_baselines:values+=['--independent-baselines']
            subprocess.run(values,check=True)
        return
    if a.command=='train':train(a)
    elif a.command=='evaluate':evaluate(a)
    else:baseline(a)


if __name__=='__main__':main()
