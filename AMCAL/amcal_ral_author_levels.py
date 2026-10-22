"""Unmodified author RAL: clean-0 bootstrap, fresh reload for each held-out attack."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from baselines import ral_author
from amcal_external_data_audit import excluded

COMMIT='2a34a4b1b1e11054a718233e19de0c2134fd68e8'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def read_domain(root,level,artifact):
    path=root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
    mapping=path.with_name(path.stem+'_rows.csv')
    frame=pd.read_csv(path)
    names=artifact['feature_names'];label=artifact['label_name']
    if list(frame.columns)!=names+[label]:raise ValueError('Domain schema differs: '+str(path))
    if any(excluded(c) for c in names):raise ValueError('Unrelated/identity feature in supplied schema.')
    ids=pd.to_numeric(pd.read_csv(mapping)['source_row'],errors='raise').to_numpy()
    if (len(ids)!=len(frame) or not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all()
            or len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=artifact['source_rows']).any()):
        raise ValueError('Invalid explicit row map: '+str(mapping))
    ids=ids.astype(np.int64)
    if not np.array_equal(frame[label].to_numpy(),artifact['source_labels'][ids]):raise ValueError('Source-row labels do not match: '+str(path))
    x=frame[names].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    y=frame[label].to_numpy()
    if not np.isfinite(x).all() or pd.isna(y).any():raise ValueError('Nonfinite input; no extra imputation added.')
    return x,y,ids,dict(csv_sha256=sha(path),row_map_sha256=sha(mapping))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget-fraction',type=float,default=.2)
    p.add_argument('--initial-fraction',type=float,default=.005)
    p.add_argument('--levels',nargs='+',type=int,default=[1,3,5,7,10,12,15,17,20])
    p.add_argument('--preprocessing',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not np.isfinite(a.budget_fraction) or not 0<a.budget_fraction<=1:p.error('Budget must be in (0,1].')
    if not np.isfinite(a.initial_fraction) or not 0<a.initial_fraction<=1:p.error('Initial fraction must be in (0,1].')
    if len(a.levels)!=len(set(a.levels)) or any(level<=0 for level in a.levels):p.error('Use distinct positive held-out levels.')
    root=Path(f'AMCAL/ANT_Burst_{a.dataset}_TEST').resolve()
    data=root/f'Dataset_official_v1/AdvBurst_FTSC-{a.dataset}'
    template=a.preprocessing or root/'protocol_runs'/('clean0_original_iat_paironly_batch5_trainseed42/source/Models/protocol_preprocessing.joblib' if a.dataset=='IAT' else 'light_ps_paironly_batch5_trainseed42/source/Models/protocol_preprocessing.joblib')
    artifact=joblib.load(template)
    partitions=[set(map(int,artifact[key])) for key in ('train_ids','val_ids','test_ids')]
    if any(not v for v in partitions) or any(partitions[i]&partitions[j] for i in range(3) for j in range(i)):
        raise ValueError('Invalid source partitions.')
    out=(a.output or root/'protocol_runs'/f'ral_author_clean0_levels_budget{a.budget_fraction:g}_seed{a.seed}').resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output; prior results are preserved.')
    x0,y0,ids0,source_files=read_domain(data,0,artifact)
    if source_files['csv_sha256']!=artifact['source_sha256']:
        raise ValueError('Clean CSV differs from the partition artifact source.')
    # Retain the author's 0.5%-of-supplied-clean-file bootstrap size, but only
    # take eligible TRAIN identities. No attack row, TEST or VAL identity enters fit.
    count=int(round(a.initial_fraction*len(y0)))
    eligible=np.flatnonzero(np.isin(ids0,list(partitions[0])))
    if count<5 or count>len(eligible):raise ValueError('Bootstrap must contain 5 or more eligible clean TRAIN rows; no automatic enlargement.')
    chosen=eligible[:count]
    source_val=np.flatnonzero(np.isin(ids0,list(partitions[1])))
    if len(source_val)!=len(partitions[1]):raise ValueError('Clean VAL coverage mismatch.')
    test_domains={}
    for level in a.levels:
        x,y,ids,hashes=read_domain(data,level,artifact)
        lookup={int(value):pos for pos,value in enumerate(ids)}
        if not partitions[2]<=set(lookup):raise ValueError('Missing TEST identities at level '+str(level))
        selected=np.asarray([lookup[int(value)] for value in artifact['test_ids']],dtype=np.int64)
        test_domains[level]=(x[selected],y[selected],ids[selected],hashes)
    random.seed(a.seed);np.random.seed(a.seed)
    learners=[KNeighborsClassifier(),RandomForestClassifier(),DecisionTreeClassifier()]
    model=ral_author.RAL(learners,.9,.025,.01,reward=1,penalty=-1,budget=a.budget_fraction,model_update_frequency=1)
    model.fit(x0[chosen],y0[chosen])
    initial_source_accuracy=model.evaluate_performance(x0[source_val],y0[source_val])
    models=out/'Models';models.mkdir(parents=True)
    checkpoint=models/'ral_author_clean0.joblib';joblib.dump(model,checkpoint)
    checkpoint_sha=sha(checkpoint)
    execution=dict(protocol='author RAL core; clean-only source and independent held-out levels',author_commit=COMMIT,
        author_source_sha256=sha(Path(ral_author.__file__)),dataset=a.dataset,seed=a.seed,budget_fraction=a.budget_fraction,
        source_files=source_files,partition_artifact_sha256=sha(template),initial_fraction=a.initial_fraction,
        initial_fraction_denominator='all rows in level-0 CSV; rows selected only from eligible TRAIN in file order',
        initial_labels=count,initial_source_ids=ids0[chosen].tolist(),initial_class_counts={str(k):int(v) for k,v in pd.Series(y0[chosen]).value_counts().items()},
        training_levels=[0],test_levels=a.levels,checkpoint_sha256=checkpoint_sha,
        model_reset='deserialize the same clean-0 checkpoint before EVERY level; counters/alpha/threshold/history reset',
        test_order='fixed seed permutation of held-out source IDs, matching AMCAL stream order',
        preprocessing='raw numeric permitted features; saved scaler ignored; template supplies schema/identity partitions only',
        parameters=dict(threshold_uncertainty=.9,threshold_greedy=.025,eta=.01,reward=1,penalty=-1,model_update_frequency=1),
        classifiers=[dict(name=type(m).__name__,parameters=m.get_params()) for m in learners],
        environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),
        adaptation_note='native author budget gate plus requested strict floor(budget*N) ceiling per level; not original one-file example split')
    write(out/'execution.json',execution)
    results=out/'Results';results.mkdir()
    rows=[]
    for level,(x,y,ids,domain_hashes) in test_domains.items():
        random.seed(a.seed);np.random.seed(a.seed)
        if sha(checkpoint)!=checkpoint_sha:raise ValueError('Initial checkpoint changed.')
        ral=joblib.load(checkpoint)  # Never carry learned target state across levels.
        if ral._n_acquired_samples or ral._n_seen:raise ValueError('Source checkpoint contains stream feedback.')
        order=np.random.default_rng(a.seed).permutation(len(y));x,y,ids=x[order],y[order],ids[order]
        n=len(y);cap=int(np.floor(a.budget_fraction*n));predictions=[];trace=[]
        initial_target=ral._committee_classifier.predict(x)
        start=time.perf_counter()
        for pos,xt in enumerate(x):
            prediction=ral._committee_classifier.predict(xt.reshape(1,-1))[0]
            committee,decisions=ral.ask_committee(xt)
            request=ral.should_label(committee)
            paid=bool(request and ral._n_acquired_samples<cap)
            if paid:ral.acquire_label(xt,y[pos],committee,decisions)
            predictions.append(prediction)
            trace.append(dict(step=pos+1,source_row=int(ids[pos]),truth=str(y[pos]),prediction=str(prediction),queried=paid,labels_used=ral._n_acquired_samples))
            if (pos+1)%200==0:print('Level',level,'rows',pos+1,'/',n,'labels',ral._n_acquired_samples,'/',cap,flush=True)
        seconds=time.perf_counter()-start
        final_source_accuracy=ral.evaluate_performance(x0[source_val],y0[source_val])
        final_target=ral._committee_classifier.predict(x)
        result=dict(method='RAL-author-core-clean0-bootstrap',level=level,accuracy=float(accuracy_score(y,predictions)),
            macro_f1=float(f1_score(y,predictions,average='macro',zero_division=0)),samples=n,labels=ral._n_acquired_samples,label_cap=cap,
            initial_target_accuracy=float(accuracy_score(y,initial_target)),final_target_accuracy=float(accuracy_score(y,final_target)),
            final_target_note='retrospective same-level score; includes queried rows, not independent audit',
            source_initial_accuracy=float(initial_source_accuracy),source_final_accuracy=float(final_source_accuracy),
            source_drop_pp=float(100*(initial_source_accuracy-final_source_accuracy)),initial_labels=count,
            checkpoint_sha256=checkpoint_sha,source_ids=ids.tolist(),target_files=domain_hashes,execution=execution,
            history_rows=len(ral._training_data_samples.y),stream_seconds=seconds)
        if result['history_rows']!=count+result['labels']:raise RuntimeError('History contains rows other than source and this level feedback.')
        write(results/f'ral_level{level}_metrics.json',result)
        pd.DataFrame(trace).to_csv(results/f'ral_level{level}_trace.csv',index=False)
        rows.append(dict(Level=level,Accuracy=100*result['accuracy'],MacroF1=100*result['macro_f1'],Labels=result['labels'],Cap=cap,
            SourceInitial=100*initial_source_accuracy,SourceFinal=100*final_source_accuracy,SourceDrop=result['source_drop_pp']))
        print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(results/'combined_results.csv',index=False)
    write(results/'summary.json',dict(method='RAL-author-core-clean0-bootstrap',budget=a.budget_fraction,
        mean_online_accuracy=float(np.mean([r['Accuracy'] for r in rows])),mean_macro_f1=float(np.mean([r['MacroF1'] for r in rows])),
        labels=sum(r['Labels'] for r in rows),mean_source_drop_pp=float(np.mean([r['SourceDrop'] for r in rows])),
        independent_levels=True,initial_labels=count,training_levels=[0],test_levels=a.levels))
    print('Outputs:',out,flush=True)


if __name__=='__main__':main()
