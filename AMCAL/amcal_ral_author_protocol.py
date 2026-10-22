"""Run the author's unmodified RAL class and example split on one supplied CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from baselines import ral_author
from amcal_external_data_audit import excluded

AUTHOR_COMMIT='2a34a4b1b1e11054a718233e19de0c2134fd68e8'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['IAT','PS'],required=True)
    p.add_argument('--level',type=int,default=0)
    p.add_argument('--csv',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--budget-fraction',type=float,default=.2)
    a=p.parse_args()
    if a.level<0:p.error('Level must be nonnegative.')
    if not np.isfinite(a.budget_fraction) or not 0<=a.budget_fraction<=1:p.error('Budget must be finite and in [0,1].')
    root=Path(f'AMCAL/ANT_Burst_{a.dataset}_TEST')
    source=(a.csv or root/f'Dataset_official_v1/AdvBurst_FTSC-{a.dataset}'/f'AdvBurstTrainedBurst_size_{a.level}_Step_2000.csv').resolve()
    out=(a.output or root/'protocol_runs'/f'ral_author_git_level{a.level}_budget{a.budget_fraction:g}_seed{a.seed}').resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output is not empty; use a new --output. No previous results overwritten.')
    frame=pd.read_csv(source)
    label=frame.columns[-1]
    names=[c for c in frame.columns[:-1] if not excluded(c)]
    if not names:raise ValueError('No permitted features.')
    x=frame[names].apply(pd.to_numeric,errors='raise').to_numpy(dtype=np.float64)
    y=frame[label].to_numpy()
    if not np.isfinite(x).all() or pd.isna(y).any():raise ValueError('Input has missing/nonfinite values; no automatic imputation added to author protocol.')
    # Exactly the example's three consecutive index slices; no shuffle/scaler.
    train_portion=.005;validation_portion=.3
    streaming_portion=1-train_portion-validation_portion
    n=len(y);nt=int(round(train_portion*n));ns=int(round(streaming_portion*n))
    train_ids=np.arange(nt);stream_ids=np.arange(nt,nt+ns);valid_ids=np.arange(nt+ns,n)
    if nt<5 or not len(stream_ids) or not len(valid_ids):raise ValueError('Author split needs at least 5 initial rows for default kNN; no automatic bootstrap enlargement.')
    xb,yb=x[train_ids],y[train_ids];xs,ys=x[stream_ids],y[stream_ids];xv,yv=x[valid_ids],y[valid_ids]
    # Only reproducibility seed is added; constructors/algorithm match author code.
    random.seed(a.seed);np.random.seed(a.seed)
    knn=KNeighborsClassifier();rf10=RandomForestClassifier();dt=DecisionTreeClassifier()
    learners=[knn,rf10,dt]
    execution=dict(protocol='author Git example on one supplied CSV',author_commit=AUTHOR_COMMIT,
        author_source_sha256=digest(Path(ral_author.__file__)),input_csv=str(source),input_sha256=digest(source),
        dataset=a.dataset,level=a.level,seed=a.seed,seed_note='Python/NumPy seed for reproducibility; upstream example has no seed',
        train_portion=train_portion,validation_portion=validation_portion,stream_portion=streaming_portion,
        rows=dict(initial=len(yb),stream=len(ys),validation=len(yv)),
        label_column=label,feature_names=names,excluded_columns=[c for c in frame.columns[:-1] if c not in names],
        order='supplied CSV row order, not independently verified timestamps',
        preprocessing='none: raw numeric permitted features; no scaling, class balancing or imputation',
        parameters=dict(threshold_greedy=.025,threshold_uncertainty=.9,eta=.01,reward=1,penalty=-1,budget=a.budget_fraction,model_update_frequency=1),
        classifiers=[dict(name=type(m).__name__,parameters=m.get_params()) for m in learners],
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        caveat='Not the matched AMCAL TRAIN/TEST protocol; initial rows come from this supplied level, not necessarily clean level 0.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf-8')
    pd.DataFrame({'row':np.arange(n),'partition':np.where(np.arange(n)<nt,'initial',np.where(np.arange(n)<nt+ns,'stream','validation'))}).to_csv(out/'split_manifest.csv',index=False)
    print('Author RAL protocol:',execution['rows'],f'budget={a.budget_fraction:g}; RF default n_estimators=',rf10.n_estimators,flush=True)
    ral=ral_author.RAL(learners,.9,.025,.01,reward=1,penalty=-1,budget=a.budget_fraction,model_update_frequency=1)
    ral.fit(xb,yb)
    initial_accuracy=ral.evaluate_performance(xv,yv)
    predictions=[];traces=[]
    started=time.perf_counter()
    for t,xt in enumerate(xs):
        # Extra observer metric only; predict() never fits or accesses current y.
        prediction=ral._committee_classifier.predict(xt.reshape(1,-1))[0]
        committee,decisions=ral.ask_committee(xt)
        selected=ral.should_label(committee)
        if selected:ral.acquire_label(xt,ys[t],committee,decisions)
        predictions.append(prediction)
        traces.append(dict(step=t+1,row=int(stream_ids[t]),truth=str(ys[t]),prediction=str(prediction),queried=bool(selected),labels_used=ral._n_acquired_samples))
        if (t+1)%200==0:print('Stream',t+1,'/',len(ys),'labels',ral._n_acquired_samples,flush=True)
    stream_seconds=time.perf_counter()-started
    final_accuracy=ral.evaluate_performance(xv,yv)
    final_predictions=ral._committee_classifier.predict(xv)
    result=dict(method='RAL-author-Git-example',dataset=a.dataset,level=a.level,seed=a.seed,
        initial_validation_accuracy=float(initial_accuracy),final_validation_accuracy=float(final_accuracy),
        final_validation_macro_f1=float(f1_score(yv,final_predictions,average='macro',zero_division=0)),
        online_accuracy=float(accuracy_score(ys,predictions)),online_macro_f1=float(f1_score(ys,predictions,average='macro',zero_division=0)),
        labels=ral._n_acquired_samples,stream_rows=len(ys),initial_labels=len(yb),
        actual_label_fraction=ral._n_acquired_samples/len(ys),native_budget=a.budget_fraction,
        hard_global_cap_added=False,online_update_window='all initial rows plus acquired feedback',
        final_history_rows=len(ral._training_data_samples.y),stream_seconds=stream_seconds,
        rf_actual_n_estimators=rf10.n_estimators,
        metrics_note='initial/final validation follows author evaluate_performance including its refit; online metric is additional predict-before-feedback observation')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame([result]).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(traces).to_csv(out/'trace.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)
    print('Author RAL output:',out,flush=True)


if __name__=='__main__':main()
