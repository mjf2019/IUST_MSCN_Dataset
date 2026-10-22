"""NPBO full data or a fast CTU/Neris DReLAB pilot; no new attacks generated."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import amcal_tun_paironly as pilot
from amcal_external_data_audit import excluded

# Published DReLAB modifications affect these values and derived quantities.
MUTABLE = {'Dur','InBytes','OutBytes','TotPkts','TotBytes','BytesPerSec',
           'BytesPerPkt','PktsPerSec','RatioOutIn'}


def capped(ids,roles,limit,rng):
    """Proportional per-role sampling without replacement, after split assignment."""
    ids=np.asarray(ids,dtype=np.int64)
    if not limit or len(ids)<=limit:return ids
    labels,counts=np.unique(roles[ids],return_counts=True)
    quotas=np.floor(limit*counts/counts.sum()).astype(int)
    quotas=np.maximum(quotas,1)
    while quotas.sum()>limit:
        options=np.flatnonzero(quotas>1)
        if not len(options):raise ValueError('Cap is too small for the available roles.')
        quotas[options[np.argmax(quotas[options])]]-=1
    while quotas.sum()<limit:
        options=np.flatnonzero(quotas<counts)
        quotas[options[np.argmax(limit*counts[options]/counts.sum()-quotas[options])]]+=1
    return np.sort(np.concatenate([rng.choice(ids[roles[ids]==label],int(q),replace=False)
                                  for label,q in zip(labels,quotas)]))


def grouped_pools(splits,roles,groups,seed,max_stream_rows,max_audit_rows):
    """Reserve whole conservative groups before selecting source/target roles."""
    pools={}
    for offset,(name,ids) in enumerate(splits.items()):
        if name=='train':
            pools['source_train']=ids[roles[ids]!='adversarial']
            pools['target_train']=ids[roles[ids]!='direct']
            continue
        source_idx,target_idx=next(GroupShuffleSplit(n_splits=1,test_size=.6,
            random_state=seed+1500+offset).split(ids,groups=groups[ids]))
        source_ids=ids[source_idx];source_ids=source_ids[roles[source_ids]!='adversarial']
        target_ids=ids[target_idx];target_ids=target_ids[roles[target_ids]!='direct']
        if len(np.unique(groups[target_ids]))<2:raise ValueError('Too few independent target groups for '+name)
        ti,ai=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1700+offset)
                   .split(target_ids,groups=groups[target_ids]))
        stream,audit=target_ids[ti],target_ids[ai]
        rng=np.random.default_rng(seed+1900+offset)
        source_ids=capped(source_ids,roles,max_audit_rows,rng)
        stream=capped(stream,roles,max_stream_rows,rng)
        audit=capped(audit,roles,max_audit_rows,rng)
        for label,part,expected in [('source',source_ids,{'benign','direct'}),
                                    ('target',stream,{'benign','adversarial'}),
                                    ('audit',audit,{'benign','adversarial'})]:
            if set(roles[part])!=expected:
                raise ValueError(f'{name}/{label}: missing class after conservative grouping; do not silently switch to row splitting.')
        if set(groups[stream]) & (set(groups[source_ids])|set(groups[audit])):
            raise RuntimeError('Observer group overlaps adaptation.')
        pools.update({f'source_{name}':source_ids,f'target_{name}':stream,f'audit_{name}':audit})
    return pools


def drelab_input(a):
    root=Path(__file__).resolve().parent/'DReLAB_TEST'
    if a.data_root:root=a.data_root.resolve()
    paths=list((root/'Dataset_official_v1').rglob('CTU/benign.csv'))
    if len(paths)!=1:raise ValueError('Exactly one CTU/benign.csv required beneath --data-root/Dataset_official_v1.')
    ctu=paths[0].parent
    files=[ctu/'benign.csv',ctu/'malicious/Neris.csv',ctu/'RF/DDQN/Neris.csv']
    for path in files:
        if not path.is_file():raise ValueError('Missing '+str(path))
    spec=dict(source_files={p.relative_to(root).as_posix():pilot.digest(p) for p in files},
              seed=a.seed,fraction=a.fraction,attack='CTU RF DDQN Neris')
    prepared=root/'Dataset_prepared_pilots'/f'CTU_Neris_RF_DDQN_f{a.fraction:g}_seed{a.seed}'
    csv_path=prepared/'pilot.csv';group_path=prepared/'groups.npy';meta=prepared/'input_manifest.json'
    if meta.exists():
        record=json.loads(meta.read_text())
        if record['spec']!=spec or record['csv_sha256']!=pilot.digest(csv_path) or record['groups_sha256']!=pilot.digest(group_path):
            raise ValueError('Prepared pilot inputs changed; use a new input path/seed.')
        return csv_path,np.load(group_path,allow_pickle=False),prepared,root
    if prepared.exists() and any(prepared.iterdir()):raise ValueError('Partial prepared pilot preserved: '+str(prepared))
    prepared.mkdir(parents=True,exist_ok=True)
    header=pd.read_csv(files[0],nrows=0)
    raw_features=[c for c in header if not c.startswith('Unnamed:') and c!='Label']
    features=[c for c in raw_features if not excluded(c)]
    immutable=[c for c in raw_features if c not in MUTABLE]
    if not immutable:raise ValueError('No immutable columns available for conservative grouping.')
    original=pd.read_csv(files[1],usecols=immutable)
    original_keys=set(pd.util.hash_pandas_object(original[immutable].astype(float),index=False).to_numpy())
    del original
    rng=np.random.default_rng(a.seed+4001);groups=[];counts=[];first=True
    for path,role,code in zip(files,['benign','direct','adversarial'],[3,1,2]):
        total=kept=0
        for frame in pd.read_csv(path,chunksize=25000,low_memory=False):
            required=[*raw_features,'Label']
            if any(c not in frame for c in required):raise ValueError('DReLAB schema mismatch: '+str(path))
            labels=frame['Label'].astype(str).str.strip().str.lower()
            accepted={'0','0.0'} if code==3 else {'1','1.0','adversarial'} if code==2 else {'1','1.0'}
            if not labels.isin(accepted).all():raise ValueError('DReLAB label/path mismatch.')
            keys=pd.util.hash_pandas_object(frame[immutable].astype(float),index=False).to_numpy()
            if code==2 and not set(keys)<=original_keys:
                raise ValueError('Attack changes supposedly immutable fields; conservative identity grouping cannot be used.')
            # Retain immutable keys with sampled rows; the effective v2 grouped
            # split is assigned by data_hooks, before TRAIN-only preprocessing.
            mask=rng.random(len(frame))<a.fraction
            chosen=frame.loc[mask,features].copy()
            chosen['label_3']=code;chosen['label_2']=code!=3
            chosen.to_csv(csv_path,mode='w' if first else 'a',index=False,header=first)
            first=False;groups.append(keys[mask]);total+=len(frame);kept+=int(mask.sum())
        counts.append(dict(role=role,original_rows=total,sampled_rows=kept,actual_fraction=kept/total))
    group_values=np.concatenate(groups);np.save(group_path,group_values,allow_pickle=False)
    pilot.runner.write_json(meta,dict(spec=spec,counts=counts,group_columns=immutable,
        group_policy='conservative immutable-feature equivalence, not verified one-to-one source identity',
        split_rule='assigned by external run protocol; balanced_v2 uses fixed-seed role-stratified whole groups',
        numeric_host_type_fields='grouping only; excluded from model features',
        independent_groups=int(len(np.unique(group_values))),
        csv_sha256=pilot.digest(csv_path),groups_sha256=pilot.digest(group_path)))
    print('Prepared DReLAB fraction:',counts,'conservative groups:',len(np.unique(group_values)),flush=True)
    return csv_path,group_values,prepared,root


def purge_processed_overlaps(splits,pools,groups,processed_groups,roles):
    """Purge whole held-out groups; never move rows or refit on held-out data."""
    splits={k:v.copy() for k,v in splits.items()}
    pools={k:v.copy() for k,v in pools.items()}
    report={'policy':'whole-group purge; TRAIN > VAL > TEST; observers > stream',
            'removed_split_rows':{},'removed_stream_rows':{}}
    seen=set()
    for name in ('train','val','test'):
        ids=splits[name]
        bad=ids[np.isin(processed_groups[ids],list(seen))]
        bad_groups=np.unique(groups[bad])
        kept=ids[~np.isin(groups[ids],bad_groups)]
        report['removed_split_rows'][name]=int(len(ids)-len(kept))
        splits[name]=kept
        seen.update(processed_groups[kept])
        for key in pools:
            if key.endswith('_'+name):pools[key]=pools[key][np.isin(pools[key],kept)]
    for name in ('val','test'):
        observer=np.concatenate([pools['source_'+name],pools['audit_'+name]])
        stream=pools['target_'+name]
        bad=stream[np.isin(processed_groups[stream],processed_groups[observer])]
        kept=stream[~np.isin(groups[stream],np.unique(groups[bad]))]
        report['removed_stream_rows'][name]=int(len(stream)-len(kept))
        pools['target_'+name]=kept
    for key,ids in pools.items():
        expected={'benign','adversarial'} if key in ('target_test','audit_test') else {'benign','direct'}
        if set(roles[ids])!=expected:
            raise ValueError(f'{key} loses a class after whole-group overlap purge: {report}')
    print('Processed overlap purge:',report,flush=True)
    return splits,pools,report


@contextmanager
def data_hooks(a,group_values):
    old_partition,old_domains=pilot.partition_ids,pilot.domain_splits
    old_overlap=pilot.processed_overlap_policy
    def partition(features,roles,seed):
        if group_values is None:
            splits,groups=old_partition(features,roles,seed)
        else:
            if len(group_values)!=len(features):raise ValueError('Prepared group map length mismatch.')
            groups=group_values
            # Fixed seed, role-stratified whole groups. No score-based seed search.
            folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
                       .split(features,roles,groups))
            splits={'train':np.sort(np.concatenate([folds[i][1] for i in (0,1,2)])),
                    'val':np.sort(folds[3][1]),'test':np.sort(folds[4][1])}
            for name,ids in splits.items():
                if set(roles[ids])!={'benign','direct','adversarial'}:
                    raise ValueError(f'{name} lacks roles under stratified group splitting.')
        for name in ('train','val'):
            splits[name]=splits[name][roles[splits[name]]!='adversarial']
            if set(roles[splits[name]])!={'benign','direct'}:
                raise ValueError(name+' lacks benign/direct data for source-only training.')
        splits['train']=capped(splits['train'],roles,a.max_train_rows,np.random.default_rng(seed+2011))
        return splits,groups
    def pools(splits,roles,groups,seed):
        result={}
        for offset,(name,ids) in enumerate(splits.items()):
            if name=='train':
                result['source_train']=ids[roles[ids]!='adversarial']
                result['target_train']=ids.copy()
                continue
            folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed+1500+offset)
                       .split(ids,roles[ids],groups[ids]))
            source=ids[np.concatenate([folds[i][1] for i in (0,1)])]
            target=ids[np.concatenate([folds[i][1] for i in (2,3,4)])]
            source=source[roles[source]!='adversarial']
            target=target[roles[target]!=('direct' if name=='test' else 'adversarial')]
            ti,ai=next(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed+1700+offset)
                       .split(target,roles[target],groups[target]))
            result.update({f'source_{name}':source,f'target_{name}':target[ti],f'audit_{name}':target[ai]})
        for key,ids in result.items():
            expected={'benign','adversarial'} if key in ('target_test','audit_test') else {'benign','direct'}
            if set(roles[ids])!=expected:raise ValueError(key+' lacks expected source-only protocol classes.')
        return result
    pilot.partition_ids=partition;pilot.domain_splits=pools
    if group_values is not None:
        def balanced_overlap(splits,pools,groups,processed_groups,roles):
            splits,pools,report=purge_processed_overlaps(splits,pools,groups,processed_groups,roles)
            counts={}
            for offset,(key,ids) in enumerate(pools.items()):
                if key.endswith('_train'):continue
                cap=a.max_stream_rows if key.startswith('target_') else a.max_audit_rows
                labels=np.unique(roles[ids])
                n=min(int(np.sum(roles[ids]==label)) for label in labels)
                if cap:n=min(n,cap//2)
                if n<20:
                    raise ValueError(f'{key}: only {n} rows per class after group separation/purge; '
                                     'need at least 20. Increase Fraction with a new Output; no row-split fallback.')
                rng=np.random.default_rng(a.seed+2500+offset)
                pools[key]=rng.permutation(np.concatenate([rng.choice(ids[roles[ids]==label],n,replace=False) for label in labels]))
                counts[key]={str(label):n for label in labels}
            report['balanced_evaluation_roles']=counts
            report['evaluation_population']='50:50 benign/attack case-control pilot, not natural prevalence'
            print('Balanced DReLAB evaluation:',counts,flush=True)
            return splits,pools,report
        pilot.processed_overlap_policy=balanced_overlap
    try:yield
    finally:
        pilot.partition_ids=old_partition;pilot.domain_splits=old_domains
        pilot.processed_overlap_policy=old_overlap


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['NPBO','DReLAB'],required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--data-root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--fraction',type=float,default=.1)
    p.add_argument('--max-train-rows',type=int,default=0);p.add_argument('--max-stream-rows',type=int,default=0)
    p.add_argument('--max-audit-rows',type=int,default=0)
    p.add_argument('--warm-epochs',type=int,default=3);p.add_argument('--pair-epochs',type=int,default=6)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cuda')
    a=p.parse_args(argv)
    if not 0<a.fraction<=1:p.error('--fraction must be in (0,1].')
    if min(a.max_train_rows,a.max_stream_rows,a.max_audit_rows)<0:p.error('Caps must be nonnegative; zero means no cap.')
    if min(a.warm_epochs,a.pair_epochs)<1:p.error('Epochs must be positive.')
    if a.dataset=='DReLAB':
        csv_path,groups,input_path,root=drelab_input(a)
        name='DReLAB-CTU-Neris-RF-DDQN'
        default_output=root/'protocol_runs'/f'paironly_sourceonly_CTUNeris_f{a.fraction:g}_seed{a.seed}'
    else:
        root=a.data_root or Path(__file__).resolve().parent/'ASNM_NPBO_TEST'
        paths=list((root/'Dataset_official_v1').rglob('*.csv'))
        if len(paths)!=1:raise ValueError('Exactly one NPBO CSV required beneath Dataset_official_v1.')
        csv_path=paths[0];groups=None;input_path=csv_path.parent;name='ASNM-NPBO'
        default_output=root/'protocol_runs'/f'paironly_sourceonly_seed{a.seed}'
    output=(a.output or default_output).resolve()
    protocol=dict(training_protocol='source_only_no_adversarial_train_or_val_v1',dataset=name,seed=a.seed,fraction=a.fraction if groups is not None else 1.,
        max_train_rows=a.max_train_rows,max_stream_rows=a.max_stream_rows,max_audit_rows=a.max_audit_rows,
        warm_epochs=a.warm_epochs,pair_epochs=a.pair_epochs,
        source_sha256=pilot.digest(csv_path),
        groups_sha256=pilot.digest(input_path/'groups.npy') if groups is not None else None)
    if groups is not None:protocol['processed_overlap_policy']='whole_group_purge_balanced_v2'
    if groups is not None:protocol['split_policy']='role_stratified_group_60_20_20_v2'
    protocol_path=output.parent/(output.name+'_external_protocol.json')
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Sampling/protocol changed; choose a new --output.')
    elif a.action=='Evaluate':raise ValueError('No trained protocol found.')
    else:pilot.runner.write_json(protocol_path,protocol)
    arguments=['--source-only-training','--dataset-name',name,'--csv',str(csv_path),'--output',str(output),'--action',a.action,
        '--seed',str(a.seed),'--device',a.device,'--warm-epochs',str(a.warm_epochs),'--pair-epochs',str(a.pair_epochs)]
    with data_hooks(a,groups):pilot.main(arguments)


if __name__=='__main__':main()
