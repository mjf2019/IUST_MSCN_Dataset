"""Corrected author-structure AF on supplied attacks, with clean-only source.

Offline N-shot calibration, NOT the historical online AF extension. The shared
af_author_protocol implementation is used unchanged. No AMCAL model is trained.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import amcal_clean_supplied as data
from baselines.adversarial_domains import digest
from baselines.af_author_protocol import (
    AFParameters, AUTHOR_COMMIT, PROTOCOL_VERSION,
    per_class_indices, fit, embed, target_knn,
)


def arrays(root,artifact,level,split):
    x,y,ids=data.runner.legacy.read_domain(root,artifact,level,split)
    return x.squeeze(1).cpu().numpy(),np.asarray(y),np.asarray(ids)


def synchronize(device):
    if device=='cuda':torch.cuda.synchronize()


def run(a):
    data.core.device=torch.device(a.device);data.core.set_seed(a.seed)
    training,manifest=data.prepare(a)
    # This prepares/verifies only partitions and the clean TRAIN scaler.
    source_root=data.preprocessing_stage(a,training,manifest)
    preprocessing=source_root/'Models/protocol_preprocessing.joblib'
    artifact=joblib.load(preprocessing)
    classes=np.arange(len(artifact['label_encoder'].classes_))
    sx,sy,sids=arrays(training,artifact,0,'train_ids')
    selected=per_class_indices(sy,classes,25,np.random.default_rng(a.seed))
    source=(sx[selected],sy[selected],sids[selected])
    parameters=AFParameters(iterations=a.iterations,tuning_epochs=a.tuning_epochs)
    method_source=Path(__file__).parent/'baselines/af_author_protocol.py'
    config=dict(method='AF-corrected-author-structure-supplied-PyTorch-port',
        author_commit=AUTHOR_COMMIT,protocol_version=PROTOCOL_VERSION,
        author_port_sha256=digest(method_source),dataset=a.dataset,seed=a.seed,
        source_levels=[0],source_per_class=25,shots=a.shots,
        parameters=asdict(parameters),budget_fraction=.2,
        budget_scope='maximum target support labels / all held-out target rows BEFORE support removal',
        initialization='fresh AF model and both optimizers for every attack/strength; same seed; no cross-level state',
        amcal_model_training=False,preprocessing_sha256=digest(preprocessing),
        input_protocol_sha256=digest(a.output/'clean_supplied_protocol.json'),
        source_ids=source[2].tolist(),
        support_selection='class-balanced N-shot benchmark selection using target labels, NOT a label-blind online query policy',
        source_only_meaning='source examples and preprocessing use clean TRAIN only; target calibration necessarily participates in AF adaptation/tuning',
        comparison='offline target-calibrated accuracy on remaining rows; NOT AMCAL prequential accuracy on the entire stream',
        adaptation='supplied numeric network features instead of author WF traces; documented PyTorch port; no historical numerical equivalence claimed')
    execution=a.output/'af_author_execution.json'
    if execution.exists() and json.loads(execution.read_text())!=config:
        raise ValueError('AF source/parameters changed; use a fresh --output.')
    data.write(execution,config)
    tasks=[]
    for attack,radii in data.ATTACKS.items():
        root=a.output/'EvaluationDomains'/attack
        for level,radius in enumerate(radii,1):
            x,y,ids=arrays(root,artifact,level,'test_ids')
            cap=int(np.floor(.2*len(y)))
            if a.shots*len(classes)>cap:
                raise ValueError(f'{attack}/{radius}: {a.shots}-shot needs {a.shots*len(classes)} labels, cap is {cap}; use a smaller --shots in a fresh output.')
            # Match support identities across attack strengths where IDs match.
            indices=per_class_indices(y,classes,a.shots,np.random.default_rng(a.seed+81327))
            mask=np.ones(len(y),dtype=bool);mask[indices]=False
            if not mask.any() or any(not np.any(y[mask]==cls) for cls in classes):
                raise ValueError('Each class needs held-out query rows after support removal.')
            if np.intersect1d(source[2],ids).size:
                raise ValueError('Clean source and target pool identity overlap.')
            tasks.append((attack,level,radius,x,y,ids,indices,mask,cap))
    rows=[]
    # All 12 tasks are checked before the first fit; no query examples enter fit.
    for attack,level,radius,x,y,ids,si,mask,cap in tasks:
        root=a.output/'AF_author'/attack/f'level{level}_shot{a.shots}'
        metrics_path=root/'metrics.json';marker=root/'complete.json'
        expected=dict(execution=config,attack=attack,level=level,strength=radius,
            domain_sha256=digest(a.output/'EvaluationDomains'/attack/data.domain_name(level)),
            support_ids=ids[si].tolist(),query_ids=ids[mask].tolist())
        if marker.exists():
            complete=json.loads(marker.read_text())
            if complete['fingerprint']!=expected:
                raise ValueError('Completed AF task has different inputs; use fresh output.')
            for name,value in complete['outputs'].items():
                if digest(root/name)!=value:raise ValueError('Completed AF output changed: '+name)
            m=json.loads(metrics_path.read_text())
            print('Skipping verified AF task',attack,radius,flush=True)
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError('Partial AF task preserved; use a fresh --output: '+str(root))
            root.mkdir(parents=True,exist_ok=True)
            tx,ty=x[si],y[si];qx,qy=x[mask],y[mask]
            synchronize(a.device);start=time.perf_counter()
            # Unmodified corrected author implementation: no online/GRL adapter.
            model,log=fit(source[0],source[1],tx,ty,classes,a.seed,parameters,a.device)
            synchronize(a.device);training_seconds=time.perf_counter()-start
            synchronize(a.device);start=time.perf_counter()
            knn=target_knn(embed(model.extractor,tx,a.device),ty,a.shots)
            predicted=knn.predict(embed(model.extractor,qx,a.device))
            synchronize(a.device);inference_seconds=time.perf_counter()-start
            m=dict(method=config['method'],attack=attack,level=level,strength=radius,
                accuracy=float(accuracy_score(qy,predicted)),macro_f1=float(f1_score(qy,predicted,average='macro',zero_division=0)),
                source_labels=len(source[1]),target_rows_before_calibration=len(y),samples=len(qy),
                calibration_labels=len(ty),label_cap=cap,calibration_fraction=len(ty)/len(y),online_labels=0,
                support_origin='target calibration ONLY',k=a.shots,weights='distance',metric='cosine',
                source_ids=source[2].tolist(),target_calibration_ids=ids[si].tolist(),test_ids=ids[mask].tolist(),
                training_seconds=training_seconds,inference_seconds=inference_seconds,
                inference_ms_per_query=1000*inference_seconds/len(qy),
                forgetting='not reported: author fit jointly adapts source/target; no source-only pretrained checkpoint baseline',
                training=log,execution=config)
            torch.save(dict(state=model.state_dict(),classes=len(classes),parameters=asdict(parameters)),root/'af.pth')
            data.write(metrics_path,m)
            data.runner.write_csv(root/'predictions.csv',[dict(source_row=int(i),truth=int(y),prediction=int(p))
                for i,y,p in zip(ids[mask],qy,predicted)])
            data.write(marker,dict(fingerprint=expected,outputs={n:digest(root/n) for n in ('af.pth','metrics.json','predictions.csv')}))
        rows.append(dict(Dataset=a.dataset,Method='AF-corrected-author-structure',Attack=attack,Strength=radius,
            Shots=a.shots,Accuracy=100*m['accuracy'],MacroF1=100*m['macro_f1'],
            CalibrationLabels=m['calibration_labels'],Cap=cap,QueryRows=m['samples'],
            InferenceMsPerQuery=m['inference_ms_per_query']))
        data.runner.write_csv(a.output/'Results/af_author_comparison.csv',rows)
        print(rows[-1],flush=True)
    print('AF author-structure outputs:',a.output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=data.DATASETS,required=True)
    p.add_argument('--data-dir',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--shots',type=int,choices=[1,5,10,15,20],default=5)
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--tuning-epochs',type=int,default=30)
    p.add_argument('--assume-historical-pool-order',action='store_true')
    a=p.parse_args()
    if min(a.iterations,a.tuning_epochs)<1:p.error('Positive training lengths required.')
    root=Path('AMCAL')/data.DATASETS[a.dataset][0]
    a.data_dir=(a.data_dir or root/'Dataset').resolve()
    a.output=(a.output or root/'protocol_runs'/f'clean0_AF_author_shot{a.shots}_attacks_seed{a.seed}').resolve()
    a.device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable.')
    # Shared data-manifest fields only; these do not invoke any AMCAL trainer.
    a.cnn_epochs=60;a.context_epochs=20;a.epochs=12;a.utility_epochs=3;a.patience=5
    run(a)


if __name__=='__main__':main()
