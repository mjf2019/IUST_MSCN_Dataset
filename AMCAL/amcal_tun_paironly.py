"""Small-data ASNM-TUN PairOnly pilot using the existing three-action agent/stream."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import amcal_final_enhancement as runner
from amcal_batch5_pipeline import fit, digest
from amcal_external_data_audit import excluded, separator, roles_and_targets
from enhancements.config import EnhancementConfig
from enhancements.agent import DeviceReliability
from enhancements.stream import episode, observer


def partition_ids(features, roles, seed):
    hashes=pd.util.hash_pandas_object(features,index=False).to_numpy()
    groups=pd.factorize(hashes,sort=False)[0]
    ids=np.arange(len(features))
    train,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=seed).split(ids,groups=groups))
    v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(rest,groups=groups[rest]))
    splits={'train':train,'val':rest[v],'test':rest[t]}
    for key,indices in splits.items():
        if set(roles[indices]) != {'benign','direct','adversarial'}:
            raise ValueError(f'{key} lacks a role after grouped split; choose a predeclared different --seed.')
    for left,right in [('train','val'),('train','test'),('val','test')]:
        if set(groups[splits[left]]) & set(groups[splits[right]]):raise RuntimeError('Duplicate features cross partitions.')
    return splits,groups


def domain_splits(splits, roles, groups, seed):
    result={}
    for offset,(name,ids) in enumerate(splits.items()):
        if name=='train':
            source=ids[roles[ids]!='adversarial'];target=ids[roles[ids]!='direct']
            result.update(source_train=source,target_train=target)
            continue
        # Reserve entire benign groups for the clean audit. No audit identity or
        # feature-identical counterpart can enter target adaptation.
        benign=ids[roles[ids]=='benign']; unique=np.unique(groups[benign])
        chosen=np.random.default_rng(seed+500+offset).permutation(unique)[:max(1,len(unique)//2)]
        clean_benign=benign[np.isin(groups[benign],chosen)]
        clean=np.concatenate([clean_benign,ids[roles[ids]=='direct']])
        target=ids[(roles[ids]=='adversarial') | ((roles[ids]=='benign') & ~np.isin(groups[ids],chosen))]
        # Grouped target audit, with both classes required. Exact duplicates
        # remain together; no claim of time/session independence is made.
        ti,ai=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+900+offset).split(target,groups=groups[target]))
        stream,audit=target[ti],target[ai]
        for key,part in [('stream',stream),('audit',audit)]:
            if set(roles[part])!={'benign','adversarial'}:
                raise ValueError(f'{name} {key} lacks benign/adversarial rows; grouped split too small for this seed.')
        if set(groups[stream]) & (set(groups[clean])|set(groups[audit])):
            raise RuntimeError('Observer features overlap adaptation.')
        result.update({f'source_{name}':clean,f'target_{name}':stream,f'audit_{name}':audit})
    return result


def prepare(csv_path, output, seed, max_features):
    frame=pd.read_csv(csv_path,sep=separator(csv_path),encoding='utf-8-sig')
    roles,y=roles_and_targets(frame,'ASNM_TUN_TEST',csv_path)
    roles=roles.to_numpy();y=y.to_numpy(dtype=np.int64)
    names=[c for c in frame if not excluded(c)]
    features=frame[names].apply(pd.to_numeric,errors='raise')
    splits,groups=partition_ids(features,roles,seed)
    pools=domain_splits(splits,roles,groups,seed)
    source=pools['source_train']
    raw=features.to_numpy(dtype=np.float64)
    raw[~np.isfinite(raw)]=np.nan
    # ASNM polynomials have very different scales; a fixed signed-log transform
    # avoids overflow before TRAIN-fitted imputation/selection/scaling.
    logged=np.sign(raw)*np.log1p(np.abs(raw))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(logged[source],axis=0)
    median=np.where(np.isfinite(median),median,0.)
    imputed=np.where(np.isnan(logged),median,logged)
    candidates=np.flatnonzero(imputed[source].var(axis=0)>1e-12)
    if len(candidates)<8:raise ValueError('CNN requires at least eight nonconstant TRAIN features.')
    scores,_=f_classif(imputed[source][:,candidates],y[source])
    scores=np.nan_to_num(scores,nan=-1.,posinf=np.finfo(float).max)
    count=len(candidates) if max_features==0 else min(max_features,len(candidates))
    selected=np.sort(candidates[np.argsort(-scores,kind='stable')[:count]])
    scaler=StandardScaler().fit(imputed[source][:,selected])
    x=scaler.transform(imputed[:,selected]).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite preprocessed features.')
    processed_groups=pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy()
    for left,right in [('train','val'),('train','test'),('val','test')]:
        if set(processed_groups[splits[left]]) & set(processed_groups[splits[right]]):
            raise ValueError('Preprocessed feature duplicates cross splits; increase --max-features or review grouping.')
    for name in ('val','test'):
        if set(processed_groups[pools['target_'+name]]) & (set(processed_groups[pools['source_'+name]]) | set(processed_groups[pools['audit_'+name]])):
            raise ValueError('Preprocessed audit features overlap adaptation.')
    output.mkdir(parents=True,exist_ok=True)
    artifact=dict(feature_names=[names[i] for i in selected],candidate_names=names,
        label_encoder=LabelEncoder().fit([0,1]),median=median,selected=selected,scaler=scaler,
        seed=seed,source_sha256=digest(csv_path),max_features=max_features,
        row_groups=groups,pools=pools,roles=roles,excluded_columns=[c for c in frame if excluded(c)])
    joblib.dump(artifact,output/'preprocessing.joblib')
    # Separate files: TRAIN never loads TEST row labels/tensors.
    for key,ids in pools.items():
        np.savez_compressed(output/f'{key}.npz',x=x[ids],y=y[ids],ids=ids)
    runner.write_json(output/'split_manifest.json',dict(source_sha256=artifact['source_sha256'],
        seed=seed,selected_features=artifact['feature_names'],excluded_columns=artifact['excluded_columns'],
        preprocessing='signed log; source-TRAIN median, ANOVA top-k and StandardScaler',
        pools={k:dict(rows=len(v),ids=v.tolist(),roles=pd.Series(roles[v]).value_counts().to_dict()) for k,v in pools.items()},
        limitation='exact-feature grouping, not verified session/time separation; small-data pilot'))
    return artifact


def load_domain(prepared,name,device):
    with np.load(prepared/f'{name}.npz',allow_pickle=False) as z:
        return torch.tensor(z['x'],device=device).unsqueeze(1),z['y'].copy(),z['ids'].copy()


def snap(agent):return dict(policy_state_dict=deepcopy(agent.policy.state_dict()),
                           prerouter_state_dict=deepcopy(agent.pre_router.state_dict()))


def make_agent(a,cnn,artifact,saved=None,paired=False,offline=False):
    return runner.new_agent(a,cnn,artifact,offline=offline,saved=saved,
                           query_value_only=paired,paired_query_only=paired)


def validation(a,cnn,initial,artifact,state,source,stream,audit,anchors):
    core=runner.legacy.core
    bank=[core.ContextAwareNetwork(len(artifact['feature_names'])).to(core.device).eval()]
    bank[0].load_state_dict(initial);agent=make_agent(a,cnn,artifact,state)
    continuation={}
    m,_=episode(a,cnn,bank,initial,agent,stream,anchors,reference=source[:2],continuation=continuation)
    h=continuation['history']
    pred,_,_=observer(cnn,bank,agent,*audit[:2],h,m['labels'],m['label_cap'],m['samples'],a.seed,a.config)
    accuracy=float(np.mean(pred==audit[1]));drop=m['reference']['drop_pp']
    score=(accuracy+m['accuracy'])/2-max(0.,drop-2.)/100
    return dict(score=score,audit_accuracy=accuracy,online_accuracy=m['accuracy'],source_drop_pp=drop,labels=m['labels'])


def train(a,prepared,output,artifact):
    core=runner.legacy.core
    source=load_domain(prepared,'source_train',core.device)
    target=load_domain(prepared,'target_train',core.device)
    clean_val=load_domain(prepared,'source_val',core.device)
    stream=load_domain(prepared,'target_val',core.device)
    audit=load_domain(prepared,'audit_val',core.device)
    width=len(artifact['feature_names']);models=output/'Models';models.mkdir(parents=True)
    cnn=core.CNN1D(width,2).to(core.device)
    source_train=(source[0],torch.tensor(source[1],dtype=torch.long,device=core.device))
    source_val=(clean_val[0],torch.tensor(clean_val[1],dtype=torch.long,device=core.device))
    # The original CNN uses BatchNorm; avoid singleton final training batches.
    cnn_log=fit(a,cnn,None,source_train,source_val)
    cnn.eval();cnn.requires_grad_(False)
    torch.save({'model_state_dict':cnn.state_dict()},models/'cnn.pth')
    context=core.ContextAwareNetwork(width).to(core.device).eval()
    context_log=fit(a,context,cnn,source_train,source_val,context=True)
    initial=deepcopy(context.state_dict());torch.save(initial,models/'context.pth')
    logs={};chosen=None
    for stage in ('warmup','paironly'):
        paired=stage=='paironly'
        a.train_epsilon=a.eval_epsilon if paired else .2
        agent=make_agent(a,cnn,artifact,chosen,paired=paired,offline=not paired)
        best=-float('inf');stale=0;selected=None;stage_log=[]
        inherited=snap(agent)
        for epoch in range(-1 if paired else 0,a.pair_epochs if paired else a.warm_epochs):
            if epoch>=0:
                agent.td_memory.clear();agent.pair_memory.clear();agent.route_memory.clear()
                continuation={}
                for domain in (source,target):
                    episode(a,cnn,[context],initial,agent,domain,source[:2],training=True,
                        paid_supervision=paired,continuation=continuation,
                        rollout_seed=a.seed+200003+epoch*1009)
            state=snap(agent)
            if paired:
                if any(not torch.equal(v,inherited['policy_state_dict'][k]) for k,v in state['policy_state_dict'].items() if not k.startswith('update_value.')):
                    raise RuntimeError('Frozen selector heads changed during PairOnly fitting.')
            metrics=validation(a,cnn,initial,artifact,state,clean_val,stream,audit,source[:2])
            stage_log.append(dict(epoch=epoch+1,**metrics))
            print(stage,stage_log[-1],flush=True)
            if metrics['score']>best:
                best=metrics['score'];selected=deepcopy(state);stale=0;selected_epoch=epoch+1
            else:
                stale+=1
                if stale>=a.patience:break
        chosen=selected
        logs[stage]=dict(selected_epoch=selected_epoch,epochs=stage_log,
            supervision='paid TRAIN only; paired update_value objective' if paired else 'dense TRAIN labels for selector initialization')
        torch.save(selected,models/f'{stage}.pth')
    runner.write_json(output/'training.json',dict(cnn=cnn_log,context=context_log,selector=logs,
        method=getattr(a,'dataset_name','ASNM-TUN')+' one-Context three-action PairOnly pilot; dataset-specific warmup, no imported Burst weights',
        feature_count=width,heldout_test_scored=False,context_update='original single acquired row plus protected replay; no batch5',
        source_preprocessing='fitted on source TRAIN only'))
    runner.write_json(output/'training_complete.json',dict(outputs={p.relative_to(output).as_posix():digest(p)
        for p in [*models.iterdir(),output/'training.json']}))


def evaluate(a,prepared,output,artifact):
    core=runner.legacy.core;width=len(artifact['feature_names']);models=output/'Models'
    cnn=core.CNN1D(width,2).to(core.device)
    cnn.load_state_dict(torch.load(models/'cnn.pth',map_location='cpu',weights_only=False)['model_state_dict'])
    cnn.eval();cnn.requires_grad_(False)
    initial=torch.load(models/'context.pth',map_location='cpu',weights_only=False)
    state=torch.load(models/'paironly.pth',map_location='cpu',weights_only=False)
    anchors=load_domain(prepared,'source_train',core.device)[:2]
    clean=load_domain(prepared,'source_test',core.device)
    stream=load_domain(prepared,'target_test',core.device)
    audit=load_domain(prepared,'audit_test',core.device)
    agent=make_agent(a,cnn,artifact,state)
    bank=[core.ContextAwareNetwork(width).to(core.device).eval()];bank[0].load_state_dict(initial)
    h=DeviceReliability(2,core.device,2);cap=int(.2*len(stream[1]))
    before,_,_=observer(cnn,bank,agent,*audit[:2],h,0,cap,0,a.seed,a.config)
    clean_before,_,_=observer(cnn,bank,agent,*clean[:2],h,0,cap,0,a.seed,a.config)
    cnn_before=deepcopy(cnn.state_dict());continuation={}
    m,traces=episode(a,cnn,bank,initial,agent,stream,anchors,reference=clean[:2],continuation=continuation)
    pred,_,_=observer(cnn,bank,agent,*audit[:2],continuation['history'],m['labels'],m['label_cap'],m['samples'],a.seed,a.config)
    clean_after,_,_=observer(cnn,bank,agent,*clean[:2],continuation['history'],m['labels'],m['label_cap'],m['samples'],a.seed,a.config)
    from sklearn.metrics import f1_score, recall_score
    z,_=runner.legacy.base_cache(cnn,stream[0]);frozen=z.argmax(1).cpu().numpy()
    result=dict(method='PairOnly simple '+getattr(a,'dataset_name','ASNM-TUN')+' pilot',seed=a.seed,context_lr=.0005,
        online_accuracy=m['accuracy'],online_macro_f1=m['macro_f1'],labels=m['labels'],cap=m['label_cap'],
        frozen_online_accuracy=float((frozen==stream[1]).mean()),
        frozen_online_macro_f1=float(f1_score(stream[1],frozen,average='macro',zero_division=0)),
        target_audit_initial_accuracy=float(np.mean(before==audit[1])),
        target_audit_final_accuracy=float(np.mean(pred==audit[1])),
        target_audit_final_macro_f1=float(f1_score(audit[1],pred,average='macro',zero_division=0)),
        target_audit_attack_recall=float(recall_score(audit[1],pred,zero_division=0)),
        source_initial_accuracy=m['reference']['initial'],source_final_accuracy=m['reference']['final'],
        source_drop_pp=m['reference']['drop_pp'],
        source_initial_macro_f1=float(f1_score(clean[1],clean_before,average='macro',zero_division=0)),
        source_final_macro_f1=float(f1_score(clean[1],clean_after,average='macro',zero_division=0)),
        source_initial_attack_recall=float(recall_score(clean[1],clean_before,zero_division=0)),
        source_final_attack_recall=float(recall_score(clean[1],clean_after,zero_division=0)),
        classifier_unchanged=all(torch.equal(v,cnn_before[k]) for k,v in cnn.state_dict().items()),
        samples=m['samples'],source_audit_rows=len(clean[1]),target_audit_rows=len(audit[1]),
        forgetting_definition='source accuracy before minus after; one target domain, no multi-domain BWT claimed')
    root=output/'Results';runner.write_json(root/'summary.json',result)
    runner.write_json(root/'stream_metrics.json',m);runner.write_csv(root/'trace.csv',traces)
    runner.write_csv(root/'summary.csv',[result]);print(json.dumps(result,indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-name', default='ASNM-TUN')
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--csv',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    p.add_argument('--max-features',type=int,default=64)
    p.add_argument('--cnn-epochs',type=int,default=60);p.add_argument('--context-epochs',type=int,default=10)
    p.add_argument('--warm-epochs',type=int,default=6);p.add_argument('--pair-epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=5)
    args=p.parse_args(argv)
    if args.max_features!=0 and args.max_features<8:p.error('--max-features is 0 (all) or at least 8.')
    if min(args.cnn_epochs,args.context_epochs,args.warm_epochs,args.pair_epochs,args.patience)<1:p.error('Invalid training schedule.')
    base=Path(__file__).resolve().parent/'ASNM_TUN_TEST'
    if args.csv is None:
        paths=list((base/'Dataset_official_v1').rglob('ASNM-TUN.csv'))
        if len(paths)!=1:p.error('Exactly one ASNM-TUN.csv required; use --csv.')
        args.csv=paths[0]
    args.csv=args.csv.resolve();args.output=(args.output or base/'protocol_runs'/f'paironly_simple_seed{args.seed}').resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    spec=dict(source_sha256=digest(args.csv),seed=args.seed,max_features=args.max_features,
              schedule={k:getattr(args,k) for k in ('cnn_epochs','context_epochs','warm_epochs','pair_epochs','patience')})
    if args.dataset_name!='ASNM-TUN':spec['dataset_name']=args.dataset_name
    plan=args.output/'plan.json';prepared=args.output/'Prepared'
    if plan.exists():
        if json.loads(plan.read_text())!=spec:raise ValueError('Inputs/schedule changed; use a new --output.')
        artifact=joblib.load(prepared/'preprocessing.joblib')
    else:
        if args.action=='Evaluate':raise ValueError('Train this run first.')
        if any(args.output.iterdir()):raise ValueError('Use an empty output directory.')
        artifact=prepare(args.csv,prepared,args.seed,args.max_features)
        runner.write_json(plan,spec)
        runner.write_json(args.output/'prepared_complete.json',
            {p.name:digest(p) for p in prepared.iterdir() if p.is_file()})
    prepared_record=json.loads((args.output/'prepared_complete.json').read_text())
    if not all((prepared/name).is_file() and digest(prepared/name)==value for name,value in prepared_record.items()):
        raise ValueError('Prepared data changed; use a new output run.')
    print('TUN selected features:',len(artifact['feature_names']),'pool sizes:',
          {k:len(v) for k,v in artifact['pools'].items()},flush=True)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable.')
    core=runner.legacy.core;core.device=torch.device(device);core.set_seed(args.seed)
    a=runner.parser().parse_args(['train','--output',str(args.output),'--burst-dir',str(prepared)])
    a.dataset_name=args.dataset_name
    a.config=EnhancementConfig.preset('accuracy');a.classes=['0','1'];a.context_count=1
    a.context_lr=.0005;a.context_alpha=.8;a.seed=args.seed;a.paid_replay_mode='off'
    for key in ('cnn_epochs','context_epochs','warm_epochs','pair_epochs','patience'):setattr(a,key,getattr(args,key))
    core.SELECTION_COST=a.selection_cost;core.GAIN_WEIGHT=a.gain_weight;core.RETENTION_WEIGHT=a.retention_weight
    marker=args.output/'training_complete.json'
    if marker.exists():
        recorded=json.loads(marker.read_text())
        if not all((args.output/k).exists() and digest(args.output/k)==v for k,v in recorded['outputs'].items()):raise ValueError('Saved training artifacts changed.')
    elif args.action in ('All','Train'):
        if (args.output/'Models').exists():raise ValueError('Partial training preserved; use a new --output.')
        train(a,prepared,args.output,artifact)
    else:raise ValueError('Training incomplete.')
    if args.action in ('All','Evaluate'):evaluate(a,prepared,args.output,artifact)
    print(args.dataset_name+' output:',args.output,flush=True)


if __name__=='__main__':main()
