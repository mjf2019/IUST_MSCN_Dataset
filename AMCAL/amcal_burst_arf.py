"""Budget-limited River ARF baseline on the exact AMCAL Burst partitions."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(y,pred):
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)))


def evaluate(model,rows):
    return np.array([model.predict_one(row) for row in rows],dtype=int)


def main():
    from river import forest
    import river
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True,help='Official Dataset_official_v1 parent')
    p.add_argument('--representation',choices=['IAT','PS'],required=True)
    p.add_argument('--amcal-run',type=Path,required=True,help='Existing ordinary AMCAL run supplies exact preprocessing/partition IDs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trees',type=int,default=10)
    p.add_argument('--budget-fraction',type=float,default=0.2)
    p.add_argument('--train-levels',type=int,nargs='+',default=[0,3,12])
    p.add_argument('--train-passes',type=int,default=1)
    a=p.parse_args()
    if not 0<=a.budget_fraction<=1 or a.train_passes<1 or a.trees<1:p.error('Invalid budget/passes/trees')
    if len(set(a.train_levels))!=len(a.train_levels) or not set(a.train_levels)<=set([0,3,12]):p.error('Training levels must be a unique subset of 0,3,12')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output must be empty; choose a new directory')
    artifact_path=a.amcal_run/'Models/protocol_preprocessing.joblib';art=joblib.load(artifact_path)
    if art.get('seed')!=a.seed:p.error('Use the AMCAL split seed for this comparison')
    root=a.data_dir/('AdvBurst_FTSC-'+a.representation)
    def path(level):return root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
    clean=pd.read_csv(path(0));columns=art['feature_names']+[art['label_name']]
    if list(clean.columns)!=columns or sha(path(0))!=art['source_sha256']:raise ValueError('Source differs from AMCAL run')
    train_ids,val_ids,test_ids=[np.asarray(art[k],dtype=int) for k in ('train_ids','val_ids','test_ids')]
    if set(train_ids)&set(test_ids) or set(val_ids)&set(test_ids):raise ValueError('Partition IDs overlap')
    def load(level,ids):
        frame=pd.read_csv(path(level))
        if list(frame.columns)!=columns or len(frame)!=len(clean) or not frame.iloc[:,-1].equals(clean.iloc[:,-1]):raise ValueError('Domain schema/labels differ')
        mapping=pd.read_csv(path(level).with_name(path(level).stem+'_rows.csv'))
        if not np.array_equal(mapping['source_row'],np.arange(len(frame))):raise ValueError('Official row mapping differs')
        frame=frame.iloc[ids];x=art['scaler'].transform(frame.iloc[:,:-1].to_numpy()).astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Non-finite features')
        y=art['label_encoder'].transform(frame.iloc[:,-1])
        return [dict(enumerate(row.tolist())) for row in x],y
    train_sets=[load(level,train_ids) for level in a.train_levels]
    train_rows=[row for rows,y in train_sets for row in rows];train_y=np.concatenate([y for rows,y in train_sets])
    model=forest.ARFClassifier(n_models=a.trees,seed=a.seed)
    start=time.perf_counter()
    for epoch in range(a.train_passes):
        for i in np.random.default_rng(a.seed+epoch).permutation(len(train_y)):
            model.learn_one(train_rows[i],int(train_y[i]))
        print(f'ARF training pass {epoch+1}/{a.train_passes}; samples={len(train_y)}',flush=True)
    train_seconds=time.perf_counter()-start
    a.output.mkdir(parents=True,exist_ok=True)
    reference_rows,reference_y=load(0,val_ids)
    reference_initial=evaluate(model,reference_rows)
    clean_rows,clean_y=load(0,test_ids)
    clean_score=score(clean_y,evaluate(model,clean_rows))
    print('ARF clean heldout accuracy:',clean_score['accuracy'],flush=True)
    results=[]
    for level in (1,5,7,10,15,17,20):
        current=deepcopy(model)  # Reset all trees, drift detectors and RNG for every intensity.
        rows,y=load(level,test_ids);order=np.random.default_rng(a.seed).permutation(len(y));rows=[rows[i] for i in order];y=y[order]
        initial=evaluate(current,rows);pred=[];used=0;cap=int(np.floor(a.budget_fraction*len(y)))
        rng=np.random.default_rng(a.seed+719003);trace=[];start=time.perf_counter()
        for step,(row,truth) in enumerate(zip(rows,y),1):
            prediction=int(current.predict_one(row));pred.append(prediction)
            # Same query positions as the random three-action AMCAL control.
            action=int(rng.integers(0,3 if used<cap else 2));query=action==2 and used<cap
            if query:
                current.learn_one(row,int(truth));used+=1
            trace.append(dict(step=step,source_row=int(test_ids[order[step-1]]),truth=int(truth),prediction=prediction,queried=query))
        elapsed=time.perf_counter()-start
        final=evaluate(current,rows);ref_final=evaluate(current,reference_rows)
        def forgetting(before,after,truth):
            return dict(initial_accuracy=float(np.mean(before==truth)),final_accuracy=float(np.mean(after==truth)),drop_pp=float(100*(np.mean(before==truth)-np.mean(after==truth))),correct_to_wrong=int(np.sum((before==truth)&(after!=truth))),wrong_to_correct=int(np.sum((before!=truth)&(after==truth))))
        report=dict(perturbation_level=level,method='River ARFClassifier',**score(y,pred),no_adaptation_accuracy=float(np.mean(initial==y)),labels_used=used,label_cap=cap,forgetting=dict(same_level=forgetting(initial,final,y),source_validation=forgetting(reference_initial,ref_final,reference_y)),stream_seconds=elapsed,seed=a.seed,river_version=river.__version__,trees=a.trees,train_levels=a.train_levels,train_passes=a.train_passes,training_samples_per_pass=len(train_y),train_seconds=train_seconds,clean_heldout=clean_score,source_sha256=sha(path(0)),attack_sha256=sha(path(level)),preprocessing_sha256=sha(artifact_path),protocol='predict-score-query-update; identical saved AMCAL split and random-control query schedule; reset model each level; reference observer only')
        (a.output/f'arf_level{level}_metrics.json').write_text(json.dumps(report,indent=2))
        pd.DataFrame(trace).to_csv(a.output/f'arf_level{level}_trace.csv',index=False)
        results.append(dict(PER=level,Initial=100*report['no_adaptation_accuracy'],Online=100*report['accuracy'],MacroF1=100*report['macro_f1'],Labels=used,Ref0=100*report['forgetting']['source_validation']['initial_accuracy'],RefFinal=100*report['forgetting']['source_validation']['final_accuracy'],RefDrop=report['forgetting']['source_validation']['drop_pp'],CurrentFinal=100*report['forgetting']['same_level']['final_accuracy']))
        print('PER',level,'online',round(100*report['accuracy'],2),'labels',used,'/',cap,flush=True)
    table=pd.DataFrame(results);table=pd.concat([table,pd.DataFrame([dict(table.mean(numeric_only=True),PER='Mean')])],ignore_index=True)
    print('\nARF combined results (%; positive RefDrop = forgetting):\n'+table.to_string(index=False,float_format=lambda v:f'{v:.2f}'))
    table.to_csv(a.output/'arf_combined.csv',index=False)
    print('Saved:',a.output.resolve())

if __name__=='__main__':main()
