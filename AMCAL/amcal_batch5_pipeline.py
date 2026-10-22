"""IAT PairOnly batch5 specialization, or fresh light PS source + PairOnly batch5."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import amcal_final_enhancement as runner
from amcal_light_cnn import LightCNN1D, ARCHITECTURE

TRAIN = [0,1,3]
TEST = [5,7,10,12,15,17,20]


def digest(path): return runner.legacy.core.file_sha256(path)


def domain(a, artifact, level, split):
    return runner.legacy.read_domain(a.burst_dir, artifact, level, split)


def combined(a, artifact, split):
    values=[domain(a,artifact,level,split) for level in TRAIN]
    return torch.cat([v[0] for v in values]), torch.tensor(np.concatenate([v[1] for v in values]),
        dtype=torch.long,device=runner.legacy.core.device)


def fit(a, model, cnn, train, validation, context=False):
    x,y=train; vx,vy=validation
    count=torch.bincount(y,minlength=len(a.classes)).float()
    if (count==0).any(): raise ValueError('A source TRAIN class is missing.')
    weights=(1/count); weights=weights/weights.mean()
    external_balance=getattr(a,'external_balanced_loss',False)
    if external_balance:
        # Global normalization avoids batch-dependent class-weight denominators.
        weights=weights/(weights*count).sum()*count.sum()
    criterion=nn.CrossEntropyLoss(weight=weights,
        label_smoothing=0.0 if context or external_balance else .05,
        reduction='none' if external_balance else 'mean')
    if external_balance:
        print('External source loss: weighted CE, no smoothing, global normalization; class counts:',
              count.detach().cpu().tolist(),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003 if context else .001,
                               weight_decay=.0001 if context else .01)
    batch_size=min(32,len(y))
    while batch_size>2 and len(y)%batch_size==1: batch_size-=1
    loader=DataLoader(TensorDataset(x,y),batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(TensorDataset(vx,vy),batch_size=32)
    best=float('inf'); stale=0; saved=None; log=[]
    max_epochs=a.context_epochs if context else a.cnn_epochs
    for epoch in range(max_epochs):
        # Context online paths run in eval mode; retain that convention here.
        model.eval() if context else model.train()
        total=0.; correct=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            out=cnn(model(bx)) if context else model(bx)
            loss=criterion(out,by).mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            total+=float(loss.detach())*len(by);correct+=int((out.argmax(1)==by).sum())
        model.eval(); losses=0.; hits=0
        with torch.no_grad():
            for bx,by in val_loader:
                out=cnn(model(bx)) if context else model(bx)
                losses+=float(criterion(out,by).mean())*len(by);hits+=int((out.argmax(1)==by).sum())
        value=losses/len(vy)
        log.append(dict(epoch=epoch+1,train_loss=total/len(y),train_accuracy=correct/len(y),
                        validation_loss=value,validation_accuracy=hits/len(vy)))
        print(('Context' if context else 'Light CNN'),log[-1],flush=True)
        if value < best:
            best=value; saved=deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=a.patience:break
    if saved is None:raise RuntimeError('No finite validation checkpoint.')
    model.load_state_dict(saved);model.eval()
    return log


def fresh_ps_source(a, output):
    core=runner.legacy.core; source=a.source_run/'Models'
    artifact=runner.load_artifact(source,a.burst_dir)
    models=output/'Models';models.mkdir(parents=True)
    shutil.copy2(source/'protocol_preprocessing.joblib',models/'protocol_preprocessing.joblib')
    a.classes=list(artifact['label_encoder'].classes_)
    train=combined(a,artifact,'train_ids');val=combined(a,artifact,'val_ids')
    width=len(artifact['feature_names']);classes=len(a.classes)
    cnn=LightCNN1D(width,classes).to(core.device)
    cnn_log=fit(a,cnn,None,train,val)
    cnn.eval();cnn.requires_grad_(False)
    preprocessing=digest(models/'protocol_preprocessing.joblib')
    original=core.CNN1D(width,classes)
    old_count=sum(p.numel() for p in original.parameters())
    new_count=sum(p.numel() for p in cnn.parameters())
    if new_count>=old_count:raise RuntimeError('Light CNN is not smaller than original.')
    torch.save(dict(architecture=ARCHITECTURE,preprocessing_sha256=preprocessing,
        model_state_dict=cnn.state_dict(),input_length=width,num_classes=classes),models/'initial_cnn_model.pth')
    context=core.ContextAwareNetwork(width).to(core.device)
    context_log=fit(a,context,cnn,train,val,context=True)
    torch.save(dict(preprocessing_sha256=preprocessing,context_model_state_dict=context.state_dict()),
               models/'adversarial_best_context.pth')
    # Domain provenance required by the unchanged structured-selector trainer.
    torch.save(dict(training_perturbations=TRAIN,scope='fresh light PS source; no imported selector weights'),
               models/'adversarial_best_dqn.pth')
    runner.write_json(output/'source_training.json',dict(architecture=ARCHITECTURE,
        original_parameters=old_count,light_parameters=new_count,
        source_partition_sha256=preprocessing,cnn=cnn_log,context=context_log,
        training_levels=TRAIN,selection='validation loss only; no TEST scoring',
        optimizer='AdamW; CNN weight decay .01, dropout, label smoothing .05; Context weight decay .0001',
        original_cnn_sha256=digest(source/'initial_cnn_model.pth')))


def make_plan(a):
    common=['--burst-dir',str(a.burst_dir),'--profile','accuracy','--seed',str(a.seed),
        '--device',a.device,'--context-lr','0.0005','--context-alpha','0.8','--gamma','0.95',
        '--budget-fraction','0.2','--eval-epsilon','0.05','--update-policy','learned',
        '--modes','learned']
    stages=[]
    if a.dataset=='PS':
        source=a.output_root/'source';structured=a.output_root/'structured'
        stages.append(dict(name='source',kind='source',source=str(a.source_run),output=str(source),args=[]))
        stages.append(dict(name='structured',kind='structured',source=str(source),output=str(structured),args=[
            'train','--source-run',str(source),'--output',str(structured),'--burst-dir',str(a.burst_dir),
            '--levels',*map(str,TRAIN),'--balanced-reward-reference','--query-schedule','policy',
            '--context-lr','.0005','--epochs',str(a.epochs),'--patience',str(a.patience),
            '--seed',str(a.seed),'--device',a.device]))
        accuracy=a.output_root/'accuracy';utility=a.output_root/'counterfactual'
        stages.append(dict(name='accuracy',kind='enhancement',source=str(structured),output=str(accuracy),args=[
            'train','--source-run',str(structured),'--output',str(accuracy),*common,
            '--warm-start-selector','--query-schedule','policy','--levels',*map(str,TRAIN),
            '--epochs',str(a.epochs),'--patience',str(a.patience)]))
        stages.append(dict(name='counterfactual',kind='enhancement',source=str(accuracy),output=str(utility),args=[
            'train','--source-run',str(accuracy),'--output',str(utility),*common,
            '--warm-start-enhancement','--aligned-sequence','--counterfactual-utility',
            '--query-schedule','uniform','--levels',*map(str,TRAIN),
            '--epochs',str(a.utility_epochs),'--patience',str(a.utility_epochs)]))
    else:utility=a.source_run
    final=a.output_root/'paironly_batch5'
    stages.append(dict(name='paironly_batch5',kind='enhancement',source=str(utility),output=str(final),args=[
        'train','--source-run',str(utility),'--output',str(final),*common,
        '--warm-start-enhancement','--aligned-sequence','--one-context-rl-training',
        '--query-value-only-training','--paired-query-only-training','--context-count','1',
        '--query-schedule','policy','--paid-replay-mode','batch5','--levels',*map(str,TRAIN),
        '--epochs',str(a.epochs),'--patience',str(a.patience)]))
    evaluation=['evaluate','--output',str(final),'--burst-dir',str(a.burst_dir),
        '--context-count','1','--query-schedule','policy','--update-policy','learned',
        '--modes','learned','--paid-replay-mode','batch5','--context-lr','.0005',
        '--levels',*map(str,TEST),'--seed',str(a.seed),'--device',a.device,
        '--budget-fraction','.2','--eval-epsilon','.05','--routing-oracle-audit','--skip-existing-completed']
    if a.evaluation_tag:evaluation+=['--evaluation-tag',a.evaluation_tag]
    return stages,evaluation


def fingerprint(stage):
    source=Path(stage['source'])/'Models'
    names=list(runner.SOURCE_FILES)
    if stage['kind']=='enhancement':
        names+=['structured_bank_ddqn.pth' if stage['name']=='accuracy' else 'final_enhancement.pth']
    return dict(stage=stage,inputs={name:digest(source/name) for name in names})


def stage_complete(stage):
    out=Path(stage['output']);marker=out/'batch5_stage_complete.json'
    expected=fingerprint(stage)
    if not marker.exists():return False
    old=json.loads(marker.read_text())
    if old['fingerprint']!=expected:return False
    return all((out/path).is_file() and digest(out/path)==value for path,value in old['outputs'].items())


def execute(a,stage):
    out=Path(stage['output'])
    if stage_complete(stage):print('Skipping verified complete stage',stage['name'],flush=True);return
    if out.exists() and any(out.iterdir()):raise ValueError('Partial/mismatched stage preserved; use a fresh --output-root: '+str(out))
    expected=fingerprint(stage)
    if stage['kind']=='source':fresh_ps_source(a,out)
    elif stage['kind']=='structured':
        old=sys.argv
        try:sys.argv=['structured',*stage['args']];runner.legacy.main()
        finally:sys.argv=old
    else:runner.main(stage['args'])
    outputs={p.relative_to(out).as_posix():digest(p) for p in (out/'Models').iterdir() if p.is_file()}
    for name in ('source_training.json','structured_training.json','enhancement_training.json'):
        if (out/name).exists():outputs[name]=digest(out/name)
    runner.write_json(out/'batch5_stage_complete.json',dict(fingerprint=expected,outputs=outputs))


def frozen_comparison(a, final):
    """Post-training observer only; never used for selection or adaptation."""
    core=runner.legacy.core
    models=[('original',a.source_run/'Models'),('light',final/'Models')]
    rows=[]
    from sklearn.metrics import f1_score
    for name,path in models:
        artifact=runner.load_artifact(path,a.burst_dir);cnn=runner.load_cnn(path,artifact)
        for split in ('train_ids','val_ids','test_ids'):
            for level in ([0] if split!='test_ids' else [0,*TEST]):
                x,y,_=domain(a,artifact,level,split)
                z,_=runner.legacy.base_cache(cnn,x);p=z.argmax(1).cpu().numpy()
                rows.append(dict(model=name,split=split,level=level,samples=len(y),
                    accuracy=float((p==y).mean()),macro_f1=float(f1_score(y,p,average='macro',zero_division=0))))
    runner.write_csv(a.output_root/'Results/frozen_generalization.csv',rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate','Audit'],default='All')
    p.add_argument('--source-run',type=Path)
    p.add_argument('--burst-dir',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--utility-epochs',type=int,default=3)
    p.add_argument('--cnn-epochs',type=int,default=60);p.add_argument('--context-epochs',type=int,default=20)
    p.add_argument('--evaluation-tag',default='')
    a=p.parse_args(argv)
    if min(a.epochs,a.patience,a.utility_epochs,a.cnn_epochs,a.context_epochs)<1:p.error('Epochs and patience must be positive.')
    root=Path(f'AMCAL/ANT_Burst_{a.dataset}_TEST')
    a.burst_dir=(a.burst_dir or root/f'Dataset_official_v1/AdvBurst_FTSC-{a.dataset}').resolve()
    source=('final_enhancement_accuracy_counterfactual_trainseed42' if a.dataset=='IAT'
            else 'structured_bank3_balancedref_lr0p0005_seed42')
    a.source_run=(a.source_run or root/'protocol_runs'/source).resolve()
    suffix='light_ps_paironly_batch5' if a.dataset=='PS' else 'paironly_batch5'
    a.output_root=(a.output_root or root/'protocol_runs'/f'{suffix}_trainseed{a.seed}').resolve()
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable.')
    runner.legacy.core.device=torch.device(device);runner.legacy.core.set_seed(a.seed)
    artifact=runner.load_artifact(a.source_run/'Models',a.burst_dir)
    # Source-only data/provenance preflight. No TEST tensors or scores during Train.
    for level in TRAIN:
        for split in ('train_ids','val_ids'):domain(a,artifact,level,split)
    stages,evaluation=make_plan(a)
    plan=dict(version=1,dataset=a.dataset,stages=stages,
        training_knobs={k:getattr(a,k) for k in ('seed','epochs','patience','utility_epochs','cnn_epochs','context_epochs')},
        source_hashes={name:digest(a.source_run/'Models'/name) for name in runner.SOURCE_FILES},
        domain_hashes={str(level):digest(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') for level in [*TRAIN,*TEST]},
        method='PairOnly batch5 TRAIN/VAL/TEST; protected replay retained; no protected routing replay')
    plan_path=a.output_root/'batch5_pipeline_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Pipeline inputs/recipe changed; use a fresh output root.')
    else:
        if a.action not in ('All','Train'):raise ValueError('Train this pipeline first.')
        if a.output_root.exists() and any(a.output_root.iterdir()):raise ValueError('Output root must be empty.')
        runner.write_json(plan_path,plan)
    if a.action in ('All','Train'):
        for stage in stages:execute(a,stage)
    elif not all(stage_complete(stage) for stage in stages):raise ValueError('Training stages are incomplete or changed.')
    if a.action in ('All','Evaluate'):runner.main(evaluation)
    if a.action in ('All','Audit'):runner.main([*evaluation,'--continual'])
    if a.dataset=='PS' and a.action in ('All','Evaluate','Audit'):frozen_comparison(a,Path(stages[-1]['output']))
    print('Pipeline outputs:',a.output_root,flush=True)


if __name__=='__main__':main()
