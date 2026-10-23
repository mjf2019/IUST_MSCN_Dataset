"""Matched Burst baselines: CNN, RAL, AF, DFE, ARF, CALMID and MicFoal.

Training and evaluation are separate. All methods use the AMCAL source-row
partitions, train domains, global label cap and predict-score-query-update.
See BURST_BASELINES.md and CALMID_MICFOAL.md for paper mappings and adaptations.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time

import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import amcal_burst_structured_bank_protocol as shared
from amcal_burst_results_summary import display
from baselines.ral_existing import RAL, TrainingData
from baselines.stream_timing import StreamTimer, timing_row, timing_report
from baselines.calmid_micfoal import CALMID, MicFoal, VERSION as ACTIVE_VERSION
from baselines.dfe_burst import (DFEBackbone,FlowImages,GaussianCompression,
    TemplateLibrary,constrained_triplet,triplet_indices,knn_vote,embed)

core=shared.core
METHOD_NAMES={'cnn':'1D-CNN-existing-frozen','ral':'RAL-author-core-matched-source',
              'af':'AF-Burst-online-extension-NOT-paper','dfe':'DFE-Burst-adapted','arf':'River-ARFClassifier',
              'calmid':'CALMID-paper-River','micfoal':'MicFoal-paper-weighted-ARF'}


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def require_explicit_af_extension(a):
    if 'af' in a.methods and not getattr(a,'allow_af_extension',False):
        raise ValueError('The old --methods af is an online extension, NOT the original AF. '
                         'Use AMCAL/run_amcal_ral_af_audit.ps1 for corrected AF few-shot evaluation. '
                         'For an explicitly labelled historical extension only, add --allow-af-extension.')


def river_arf(trees,seed):
    try:
        from river import forest
        import river
    except ImportError as error:
        raise ImportError('ARF requires River: python -m pip install river==0.26.1') from error
    return forest.ARFClassifier(n_models=trees,seed=seed),river.__version__


def river_row(x):return {i:float(value) for i,value in enumerate(x)}


def train_arf(a,train):
    x,y=train;model,version=river_arf(a.arf_trees,a.seed)
    for epoch in range(a.arf_train_passes):
        for i in np.random.default_rng(a.seed+epoch).permutation(len(y)):
            model.learn_one(river_row(x[i]),int(y[i]))
        print(f'ARF training pass {epoch+1}/{a.arf_train_passes} | samples {len(y)} | trees {a.arf_trees}',flush=True)
    return dict(model=model,river_version=version,trees=a.arf_trees,train_passes=a.arf_train_passes,
                training_samples_per_pass=len(y))


def train_active(a,train,method):
    import river
    if river.__version__!='0.26.1':raise ValueError('CALMID/MicFoal require river==0.26.1 for the audited Python backend.')
    options=dict(seed=a.seed,models=getattr(a,'active_models',10),window=getattr(a,'active_window',500),
                 random_ratio=getattr(a,'active_random_ratio',.1),budget=a.budget_fraction)
    if method=='calmid':
        model=CALMID(range(len(a.classes)),threshold=getattr(a,'calmid_threshold',.5),
                     step=getattr(a,'calmid_step',.01),**options)
    else:
        model=MicFoal(threshold=getattr(a,'micfoal_threshold',.9),step=getattr(a,'micfoal_step',.1),
                      adaptive_budget=not getattr(a,'micfoal_fixed_budget',False),**options)
    x,y=train;order=np.random.default_rng(a.seed).permutation(len(y))
    print(f'{method.upper()} source learning | {len(y)} TRAIN rows | {options["models"]} trees | window {options["window"]}',flush=True)
    model.fit_source(x[order],y[order])
    model.begin_stream(a.seed,a.budget_fraction)
    return dict(model=model,active_protocol_version=ACTIVE_VERSION,river_version=river.__version__,
                training_samples_per_pass=len(y),initialization='all available source TRAIN rows; no test bootstrap labels',
                paper_doi='10.1016/j.knosys.2021.106778' if method=='calmid' else '10.1016/j.engappai.2022.105607')


def af_module():
    """Reuse corrected CDR_MLC AF modules, including the tested GRL sign."""
    name='amcal_burst_af_source'
    if name not in sys.modules:
        path=Path(__file__).resolve().parents[1]/'CDR_MLC/benchmarks/AF/af_single_source.py'
        if not path.exists():raise FileNotFoundError('Required corrected AF source: '+str(path))
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return sys.modules[name]


def af_network(width,classes,device):
    af=af_module();config=af.AFConfig()
    # Continuous Burst IAT/PS is not a packet-direction vector. Do not feed
    # continuous features to the direction-only DF network under an exact label.
    return af.AFDomainNetwork(af.TabularFeatureExtractor(width,config.embedding_dim),classes,config).to(device)


def af_step(model,optimizer,xs,ys,xt,device):
    size=min(len(xs),len(xt))
    if size<2:return None
    model.train()
    sx=torch.as_tensor(xs[:size],device=device);sy=torch.as_tensor(ys[:size],device=device)
    tx=torch.as_tensor(xt[:size],device=device)
    logits,domains=model(torch.cat([sx,tx]))
    labels=torch.cat([torch.zeros(size,device=device,dtype=torch.long),
                      torch.ones(size,device=device,dtype=torch.long)])
    loss=F.cross_entropy(logits[:size],sy)+F.cross_entropy(domains,labels)
    optimizer.zero_grad();loss.backward();optimizer.step();model.eval()
    return float(loss.detach())


def af_embed(model,x,device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([F.normalize(model.extractor(torch.as_tensor(x[i:i+256],device=device)),dim=1).cpu().numpy()
                               for i in range(0,len(x),256)])


def choose_k(templates,labels,queries,y):
    scores=[(f1_score(y,knn_vote(templates,labels,queries,k)[0],average='macro',zero_division=0),-k)
            for k in (1,3,5,7)]
    best=max(scores)
    return -best[1],float(best[0])


def train_dfe(a,train,validation):
    x,y=train;vx,vy=validation
    transform=FlowImages().fit(x)
    images=transform.transform(x);val_images=transform.transform(vx)
    model=DFEBackbone().to(core.device);compressor=GaussianCompression(tuple(a.dfe_betas)).to(core.device)
    optimizer=torch.optim.SGD(model.parameters(),lr=a.dfe_lr)
    library=TemplateLibrary(images,y,a.templates,a.seed);lx,ly=library.arrays()
    best=-1;state=None;stale=0;best_k=1;logs=[]
    for epoch in range(a.dfe_epochs):
        rng=np.random.default_rng(a.seed+epoch*1009)
        order=rng.permutation(len(y));model.train();losses=[]
        for start in range(0,len(y),a.batch_size):
            ids=order[start:start+a.batch_size]
            if len(ids)<2:continue
            positive,negative=triplet_indices(y,ids,rng)
            batch=torch.as_tensor(np.concatenate([images[ids],images[positive],images[negative]]),device=core.device)
            z,features=model(batch,return_features=True);anchor,pos,neg=z.chunk(3)
            loss=compressor(features)+a.dfe_lambda*constrained_triplet(anchor,pos,neg,a.dfe_alpha1,a.dfe_alpha2,a.dfe_eta)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite DFE training loss; no checkpoint written.')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        k,score=choose_k(embed(model,lx,core.device),ly,embed(model,val_images,core.device),vy)
        logs.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_macro_f1=score,k=k))
        print(f'DFE epoch {epoch+1}/{a.dfe_epochs} | loss {np.mean(losses):.4f} | validation Macro-F1 {100*score:.2f}% | k {k}',flush=True)
        if score>best+1e-10:best=score;state=cpu_state(model);stale=0;best_k=k
        else:
            stale+=1
            if stale>=a.patience:break
    if state is None:raise RuntimeError('No valid DFE model.')
    return dict(state=state,transform=transform,library=library,k=best_k,logs=logs,
                compression_estimator='same-transition diagonal Gaussian components from channel spatial moments; six mixture bounds',
                validation_macro_f1=best)


def train_af(a,train,validation,clean,shifted):
    x,y=train;vx,vy=validation;source,source_y=clean;target=shifted
    model=af_network(x.shape[1],len(a.classes),core.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.af_lr)
    rng=np.random.default_rng(a.seed+701)
    for epoch in range(a.af_epochs):
        losses=[]
        for start in range(0,len(source),a.batch_size):
            count=min(a.batch_size,len(source)-start)
            ids=rng.choice(len(source),count,replace=False)
            tid=rng.choice(len(target),count,replace=len(target)<count)
            loss=af_step(model,optimizer,source[ids],source_y[ids],target[tid],core.device)
            if loss is not None:losses.append(loss)
        print(f'AF epoch {epoch+1}/{a.af_epochs} | DANN loss {np.mean(losses):.4f}',flush=True)
    # Source-supervision fairness: all available training rows may support kNN.
    # Unlike DFE's 20-template cap, AF keeps the complete source support.
    k,score=choose_k(af_embed(model,x,core.device),y,af_embed(model,vx,core.device),vy)
    return dict(state=cpu_state(model),source_x=x,source_y=y,k=k,
                dann_source_x=source,dann_source_y=source_y,validation_macro_f1=score)


class Learner:
    def __init__(self,method,saved,a):
        self.method,self.saved,self.a=method,saved,a
        self.rng=np.random.default_rng(a.seed+81013)
        self.steps=0
        if method=='cnn':
            self.model=core.CNN1D(saved['width'],saved['classes']).to(core.device)
            self.model.load_state_dict(saved['state']);self.model.eval()
            for p in self.model.parameters():p.requires_grad_(False)
        elif method=='ral':
            self.model=deepcopy(saved['model'])
            # Paper / author code: q_so_far / seen_so_far < budget BEFORE
            # deciding to acquire the current sample. The wrapper additionally
            # retains the shared strict floor(budget*n) total ceiling.
            if getattr(a,'ral_update_window',0):
                self.model._window_source_labels=len(self.model._training_data_samples.y)
                self.model._training_data_samples=TrainingData([],[])
                self.model._online_update_window=a.ral_update_window
                self.model._online_refit_count=0
                self.model._online_max_fit_rows=0
            self.model._budget=a.budget_fraction if getattr(a,'ral_budget_mode','prefix')=='prefix' else None
        elif method in ('calmid','micfoal'):
            self.model=deepcopy(saved['model'])
            self.model.begin_stream(a.seed,a.budget_fraction)
        elif method=='arf':
            # Deep-copy trained trees, detectors, metrics and RNG; River clone()
            # would create an unfitted forest rather than restore this checkpoint.
            self.model=deepcopy(saved['model'])
        elif method=='dfe':
            self.model=DFEBackbone().to(core.device);self.model.load_state_dict(saved['state']);self.model.eval()
            for p in self.model.parameters():p.requires_grad_(False)
            self.library=deepcopy(saved['library'])
            self.refresh()
        else:
            self.model=af_network(saved['source_x'].shape[1],len(a.classes),core.device)
            self.model.load_state_dict(saved['state']);self.model.eval()
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=a.af_lr)
            self.paid_x=[];self.paid_y=[]
            self.refresh()

    def refresh(self):
        if self.method=='dfe':
            x,self.support_y=self.library.arrays()
            self.support_z=embed(self.model,x,core.device)
        elif self.method=='af':
            # Re-embed every stored support after DANN changes the extractor.
            # No future or unqueried test feature enters the DANN target batch.
            if self.paid_x:
                x=np.concatenate([self.saved['source_x'],np.asarray(self.paid_x)])
                self.support_y=np.concatenate([self.saved['source_y'],np.asarray(self.paid_y)])
            else:x=self.saved['source_x'];self.support_y=self.saved['source_y']
            self.support_z=af_embed(self.model,x,core.device)

    def predict(self,x):
        if self.method=='cnn':
            with torch.no_grad():
                return np.concatenate([self.model(torch.as_tensor(x[i:i+128],device=core.device).unsqueeze(1)).argmax(1).cpu().numpy()
                                       for i in range(0,len(x),128)])
        if self.method=='ral':return self.model._committee_classifier.predict(x)
        if self.method in ('calmid','micfoal'):return self.model.predict(x)
        if self.method=='arf':
            predictions=[self.model.predict_one(river_row(row)) for row in x]
            if any(value is None for value in predictions):raise RuntimeError('ARF produced no prediction after source training.')
            return np.asarray(predictions,dtype=np.int64)
        if self.method=='dfe':z=embed(self.model,self.saved['transform'].transform(x),core.device)
        else:z=af_embed(self.model,x,core.device)
        return knn_vote(self.support_z,self.support_y,z,self.saved['k'])[0]

    def request(self,x,can_query=True):
        if self.method in ('calmid','micfoal'):
            return self.model.request(x,can_query=can_query)
        if self.method=='ral':
            committee,decisions=self.model.ask_committee(x)
            return self.model.should_label(committee),(committee,decisions)
        return False,None

    def update(self,x,y,request_info=None):
        if self.method=='ral':
            committee,decisions=request_info
            self.model.acquire_label(x,y,committee,decisions)
        elif self.method in ('calmid','micfoal'):
            self.model.update(x,y,request_info)
        elif self.method=='dfe':
            self.library.add(self.saved['transform'].transform(x[None,:])[0],y);self.refresh()
        elif self.method=='arf':
            self.model.learn_one(river_row(x),int(y))
        elif self.method=='af':
            self.paid_x.append(x.copy());self.paid_y.append(int(y))
            source=self.saved['dann_source_x'];ys=self.saved['dann_source_y']
            count=min(self.a.batch_size,len(self.paid_x),len(source))
            sid=self.rng.choice(len(source),count,replace=False)
            tid=self.rng.choice(len(self.paid_x),count,replace=False)
            if af_step(self.model,self.optimizer,source[sid],ys[sid],np.asarray(self.paid_x)[tid],core.device) is not None:self.steps+=1
            self.refresh()


def summary_pair(before,after,y,mask):
    mask=np.asarray(mask,dtype=bool)
    if not mask.any():return dict(n=0,initial=None,final=None,drop_pp=None,broken=0,fixed=0)
    b=before[mask]==y[mask];f=after[mask]==y[mask]
    return dict(n=int(mask.sum()),initial=float(b.mean()),final=float(f.mean()),
                drop_pp=float(100*(b.mean()-f.mean())),broken=int((b&~f).sum()),fixed=int((~b&f).sum()))


def evaluate_level(a,method,saved,domain,reference,observer=True):
    x,y,ids=domain;x=x.squeeze(1).cpu().numpy()
    order=np.random.default_rng(a.seed).permutation(len(y));x,y,ids=x[order],y[order],ids[order]
    n=len(y);cap=int(np.floor(n*a.budget_fraction));learner=Learner(method,saved,a)
    reference_x,reference_y=reference
    reference_x=reference_x.squeeze(1).cpu().numpy()
    initial=learner.predict(x) if observer else None
    ref_initial=learner.predict(reference_x) if observer else None
    schedule=set(np.random.default_rng(a.seed+819003).choice(n,cap,replace=False).tolist())
    queried=np.zeros(n,dtype=bool);predictions=[];trace=[];used=0
    random.seed(a.seed)
    timer=StreamTimer('cpu' if method in ('ral','arf','calmid','micfoal') else core.device)
    timer.start();start=time.perf_counter()
    for pos in range(n):
        with timer.stage('prediction'):
            prediction=int(learner.predict(x[pos:pos+1])[0])
        # Freeze prediction before querying or exposing truth to the learner.
        with timer.stage('selection'):
            if method in ('ral','calmid','micfoal'):request,info=learner.request(x[pos],can_query=used<cap)
            else:request,info=(method!='cnn' and pos in schedule),None
        paid=bool(request and used<cap)
        if paid:
            with timer.stage('update'):
                learner.update(x[pos],int(y[pos]),info)
            used+=1;queried[pos]=True
        predictions.append(prediction)
        trace.append(dict(step=pos+1,source_row=int(ids[pos]),truth=int(y[pos]),prediction=prediction,
                          queried=paid,labels_used=used))
        if method in ('calmid','micfoal'):
            trace[-1].update(selected_random=info.random,selected_uncertain=info.uncertain,
                selected_selective=info.selective,native_budget=learner.model.budget,
                training_weight=learner.model.last_weight if paid else None)
        every=getattr(a,'progress_every',0)
        if method in ('ral','calmid','micfoal') and every>0 and (pos+1)%every==0:
            print(f'  {method.upper()} progress {pos+1}/{n} | labels {used}/{cap} | elapsed {time.perf_counter()-start:.1f}s',flush=True)
    timer.stop()
    metrics=dict(method=METHOD_NAMES[method],accuracy=float(accuracy_score(y,predictions)),
                 macro_f1=float(f1_score(y,predictions,average='macro',zero_division=0)),
                 weighted_f1=float(f1_score(y,predictions,average='weighted',zero_division=0)),
                 samples=n,labels=used,label_cap=cap,actual_label_fraction=used/n,
                 budget_utilization_pct=100*used/cap if cap else 0.,labels_remaining=cap-used,
                 online_gradient_steps=learner.steps,
                 query_protocol='paper hybrid random/uncertainty/selective with native ratio gate and strict global cap' if method in ('calmid','micfoal') else 'none' if method=='cnn' else
                    ('RAL paper prefix-ratio gate plus strict global cap' if getattr(a,'ral_budget_mode','prefix')=='prefix'
                     else 'RAL committee/greedy with global cap only') if method=='ral' else 'label-blind uniform exact cap',
                 protocol='predict-score-query-update; fresh model per level; reference/same-level observers never feed learning')
    metrics.update(timer.metrics(n,used))
    if method in ('calmid','micfoal'):
        metrics.update(active_learning=learner.model.statistics(),river_version=saved['river_version'],
                       online_update_calls=used,source_initialization=saved['initialization'])
    if method=='ral':
        if getattr(a,'ral_update_window',0):
            metrics['method']='RAL-recent5-modified'
            metrics['ral_online_window']=dict(size=a.ral_update_window,includes_current=True,
                source_rows_in_online_fit=0,retained_feedback=len(learner.model._training_data_samples.y),
                first_fit_after_labels=a.ral_update_window,
                refits=learner.model._online_refit_count,max_rows_per_fit=learner.model._online_max_fit_rows,
                semantics='sliding-window refit, not incremental learning; source model retained only until first refit')
        metrics.update(ral_budget_mode=getattr(a,'ral_budget_mode','prefix'),
                       source_labels=saved.get('source_labels',getattr(learner.model,'_window_source_labels',len(learner.model._training_data_samples.y)-used)),
                       source_initialization=saved.get('source_initialization','matched source TRAIN; historical checkpoint'),
                       author_code_commit='2a34a4b1b1e11054a718233e19de0c2134fd68e8',
                       ral_protocol_version=saved.get('ral_protocol_version',0),
                       ral_trees=next((member.n_estimators for member in learner.model._committee_classifiers
                                       if isinstance(member,RandomForestClassifier)),None),
                       ral_seen=learner.model._n_seen,
                       ral_final_threshold=learner.model._threshold_uncertainty,
                       ral_final_decision_powers=list(learner.model._alpha))
    if method=='arf':
        metrics.update(river_version=saved['river_version'],trees=saved['trees'],
                       train_passes=saved['train_passes'],online_learn_one_calls=used,
                       training_samples_per_pass=saved['training_samples_per_pass'])
    if observer:
        final=learner.predict(x);ref_final=learner.predict(reference_x)
        metrics['same_level']=dict(all_rows=summary_pair(initial,final,y,np.ones(n,dtype=bool)),
                                   unqueried=summary_pair(initial,final,y,~queried))
        metrics['reference']=summary_pair(ref_initial,ref_final,reference_y,np.ones(len(reference_y),dtype=bool))
    return metrics,trace


def training_data(a,artifact):
    parts={level:shared.read_domain(a.burst_dir,artifact,level,'train_ids') for level in a.train_levels}
    vals=[shared.read_domain(a.burst_dir,artifact,level,'val_ids') for level in a.train_levels]
    x=np.concatenate([v[0].squeeze(1).cpu().numpy() for v in parts.values()]);y=np.concatenate([v[1] for v in parts.values()])
    vx=np.concatenate([v[0].squeeze(1).cpu().numpy() for v in vals]);vy=np.concatenate([v[1] for v in vals])
    return (x,y),(vx,vy),parts


def train(a,artifact):
    require_explicit_af_extension(a)
    if a.output.exists():raise ValueError('Use a fresh baseline output directory.')
    a.output.mkdir(parents=True);models=a.output/'Models';models.mkdir()
    shutil.copy2(a.source_run/'Models/protocol_preprocessing.joblib',models/'protocol_preprocessing.joblib')
    train,validation,parts=training_data(a,artifact)
    domain_hashes={str(level):core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv') for level in a.train_levels}
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}
    provenance=dict(version=1,train_levels=a.train_levels,training_domains_sha256=domain_hashes,
                    source_preprocessing_sha256=core.file_sha256(models/'protocol_preprocessing.joblib'),
                    partition_sha256={key:hashlib.sha256(np.asarray(artifact[key],dtype='<i8').tobytes()).hexdigest()
                                      for key in ('train_ids','val_ids','test_ids')},
                    config=config,classes=a.classes,feature_names=artifact['feature_names'])
    for method in a.methods:
        core.set_seed(a.seed)
        if method=='cnn':
            checkpoint=torch.load(a.source_run/'Models/initial_cnn_model.pth',map_location='cpu',weights_only=False)
            saved=dict(state=checkpoint['model_state_dict'],width=train[0].shape[1],classes=len(a.classes),
                       source_cnn_sha256=core.file_sha256(a.source_run/'Models/initial_cnn_model.pth'))
            print('1D-CNN: reused identical frozen CNN checkpoint; no redundant retraining.',flush=True)
        elif method=='ral':
            model=RAL([KNeighborsClassifier(n_neighbors=5),RandomForestClassifier(n_estimators=10,random_state=a.seed,n_jobs=a.jobs),
                       DecisionTreeClassifier(random_state=a.seed)],.9,.025,.01,
                       reward=1,penalty=-1,budget=a.budget_fraction,model_update_frequency=1)
            model.fit(*train);saved=dict(model=model,ral_protocol_version=2,ral_trees=10,
                                        algorithm_reference='Wassermann et al. 2021 Algorithm 2; kNN5, DT, RF10; source-code prefix ratio gate',
                                        author_code_commit='2a34a4b1b1e11054a718233e19de0c2134fd68e8',
                                        source_initialization='all matched source TRAIN rows across declared levels; not original paper small chronological bootstrap',
                                        source_labels=len(train[1]),
                                        budget_protocol='paper prefix ratio plus common strict total cap')
            print(f'RAL trained on {len(train[1])} rows | kNN k=5, decision tree, RF=10 | paper reward/EXP4/threshold updates.',flush=True)
        elif method=='dfe':saved=train_dfe(a,train,validation)
        elif method=='arf':saved=train_arf(a,train)
        elif method in ('calmid','micfoal'):saved=train_active(a,train,method)
        else:
            if 0 not in parts or len(parts)<2:raise ValueError('AF DANN requires clean source and training shift domains.')
            clean=(parts[0][0].squeeze(1).cpu().numpy(),parts[0][1])
            shifted=np.concatenate([v[0].squeeze(1).cpu().numpy() for k,v in parts.items() if k!=0])
            saved=train_af(a,train,validation,clean,shifted)
        saved['provenance']=provenance
        joblib.dump(saved,models/f'{method}.joblib')
        learner=Learner(method,saved,a)
        prediction=learner.predict(validation[0]);score=f1_score(validation[1],prediction,average='macro',zero_division=0)
        print(f'{METHOD_NAMES[method]} ready | validation Macro-F1 {100*score:.2f}%',flush=True)
    (a.output/'training_manifest.json').write_text(json.dumps(provenance,indent=2))
    print('Training complete. No heldout test scored.')


def evaluate(a,artifact):
    require_explicit_af_extension(a)
    root=a.output/'Results';root.mkdir(exist_ok=True)
    reference=shared.read_domain(a.burst_dir,artifact,0,'val_ids')[:2]
    config=dict(seed=a.seed,budget=a.budget_fraction,af_online_lr=a.af_lr,protocol_version=1,timing_version=1)
    if getattr(a,'endpoint_metrics',False):
        config['endpoint_metrics_version']=1
        config['endpoint_metrics_methods']=a.methods
    if 'af' in a.methods:config['af_historical_online_extension']=True
    attack_artifact=artifact
    if getattr(a,'test_scaler','training')=='notebook':
        from baselines.evaluation_scaling import notebook_scaler
        attack_artifact,scaler_info=notebook_scaler(a.burst_dir,artifact,a.levels)
        config['test_scaler']=scaler_info
        print(f'Notebook test scaler: fitted once on level {scaler_info["fit_level"]}; checkpoints unchanged.',flush=True)
    if getattr(a,'full_unseen_levels',False):config['unseen_rows']='all supplied rows; source IDs may occur in training at other intensities'
    if 'ral' in a.methods:
        config.update(ral_protocol_version=2,ral_budget_mode=getattr(a,'ral_budget_mode','prefix'))
        if getattr(a,'ral_update_window',0):config['ral_online_update_window']=a.ral_update_window
    if set(a.methods)&{'calmid','micfoal'}:config['active_protocol_version']=ACTIVE_VERSION
    tag=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:8]
    rows=[];same_rows=[];time_rows=[]
    for method in a.methods:
        saved=joblib.load(a.output/'Models'/f'{method}.joblib');p=saved['provenance']
        if method=='ral' and saved.get('ral_protocol_version',0)<2:
            raise ValueError('Old RAL checkpoint used default RF=100. Retrain --methods ral in a fresh output for paper RF=10.')
        if method in ('calmid','micfoal') and saved.get('active_protocol_version')!=ACTIVE_VERSION:
            raise ValueError('CALMID/MicFoal protocol differs; train a fresh matching checkpoint.')
        if method in ('calmid','micfoal'):
            import river
            if saved['river_version']!=river.__version__:raise ValueError('River version differs from this CALMID/MicFoal checkpoint.')
        if p['source_preprocessing_sha256']!=core.file_sha256(a.output/'Models/protocol_preprocessing.joblib'):raise ValueError('Baseline preprocessing changed.')
        if set(a.levels)&set(p['train_levels']) and not a.include_seen_levels:raise ValueError('Test levels overlap training levels.')
        for level,digest in p['training_domains_sha256'].items():
            if core.file_sha256(a.training_domain_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')!=digest:raise ValueError('Training domain provenance changed.')
        for level in a.levels:
            core.set_seed(a.seed)
            full=getattr(a,'full_unseen_levels',False) and level not in p['train_levels']
            domain=shared.read_domain(a.burst_dir,attack_artifact if level>0 else artifact,level,'test_ids',full_domain=full)
            if method=='ral':
                print(f'RAL | PER {level} | starting {len(domain[1])} rows | RF=10 | budget {getattr(a,"ral_budget_mode","prefix")} | strict cap {int(a.budget_fraction*len(domain[1]))}',flush=True)
            metrics,trace=evaluate_level(a,method,saved,domain,reference)
            metrics.update(level=level,execution_config=config,training_provenance=p,
                           target_sha256=core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'))
            stem=f'{method}_per{level}_seed{a.seed}_{tag}'
            (root/(stem+'_metrics.json')).write_text(json.dumps(metrics,indent=2))
            shared.base.write_trace(root/(stem+'_trace.csv'),trace)
            ref=metrics['reference'];same=metrics['same_level']['unqueried']
            row=dict(Method=method,PER=level,Online=100*metrics['accuracy'],MacroF1=100*metrics['macro_f1'],
                     Labels=metrics['labels'],Cap=metrics['label_cap'],
                     **{'Label%':100*metrics['actual_label_fraction'],'CapUsed%':metrics['budget_utilization_pct']},
                     Ref0=100*ref['initial'],RefFinal=100*ref['final'],RefDrop=ref['drop_pp'])
            rows.append(row)
            time_rows.append(timing_row(method,level,metrics))
            same_rows.append(dict(Method=method,PER=level,N=same['n'],Initial=None if same['initial'] is None else 100*same['initial'],
                                  Final=None if same['final'] is None else 100*same['final'],Drop=same['drop_pp'],Broken=same['broken'],Fixed=same['fixed']))
            print(f'{method.upper()} | PER {level} | online {row["Online"]:.2f}% | labels {metrics["labels"]}/{metrics["label_cap"]} ({row["Label%"]:.2f}% of stream; {row["CapUsed%"]:.2f}% of cap) | clean {row["Ref0"]:.2f}% -> {row["RefFinal"]:.2f}% | drop {row["RefDrop"]:+.2f} pp',flush=True)
            print(timing_report(metrics),flush=True)
            if method in ('calmid','micfoal'):
                active=metrics['active_learning']
                flags=active['query_counts']
                print(f'  Native budget {active["native_budget_initial"]:.4f} -> {active["native_budget_final"]:.4f} | query flags random/uncertain/selective {flags.get("random",0)}/{flags.get("uncertain",0)}/{flags.get("selective",0)} | drift replacements {active["drifts"]}',flush=True)
            if same['n']:print(f'  Same-level unqueried: {100*same["initial"]:.2f}% -> {100*same["final"]:.2f}% | drop {same["drop_pp"]:+.2f} pp | broken {same["broken"]} | fixed {same["fixed"]}')
    for method in a.methods:
        own=[r for r in rows if r['Method']==method]
        rows.append(dict(Method=method,PER='Mean',**{key:float(np.mean([r[key] for r in own])) for key in rows[0] if key not in ('Method','PER')}))
    print('\nCombined baselines (%; Drop in pp; positive Drop = net forgetting):')
    display(rows,list(rows[0]));shared.base.write_trace(root/f'combined_baselines_seed{a.seed}_{tag}.csv',rows)
    print('Labels: acquired labels; Cap: maximum allowed labels; Label%: percentage of stream; CapUsed%: percentage of cap used (0 if cap is zero).')
    for method in a.methods:
        own=[r for r in time_rows if r['Method']==method]
        time_rows.append(dict(Method=method,PER='Mean',**{
            key:float(np.mean([r[key] for r in own])) for key in time_rows[0] if key not in ('Method','PER')}))
    print('\nOnline runtime (seconds; synchronized CUDA; excludes endpoint observers and file I/O):')
    display(time_rows,list(time_rows[0]));shared.base.write_trace(root/f'combined_baselines_seed{a.seed}_{tag}_timing.csv',time_rows)
    print('Pred includes Cache (not an extra charge). Update includes all paid adaptation. Other includes bookkeeping. Run methods sequentially for timing comparisons.')
    for method in a.methods:
        own=[r for r in same_rows if r['Method']==method and r['N']>0]
        if own:same_rows.append(dict(Method=method,PER='Mean',**{
            key:float(np.mean([r[key] for r in own])) for key in same_rows[0] if key not in ('Method','PER')}))
    print('\nSame-level retention on unqueried rows (retrospective):')
    display(same_rows,list(same_rows[0]));shared.base.write_trace(root/f'combined_baselines_seed{a.seed}_{tag}_same_level.csv',same_rows)
    print('Saved:',root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--independent-source',action='store_true',help='Use a separately trained matched-domain CNN/preprocessing source, without an AMCAL selector.')
    parser.add_argument('--training-domain-dir',type=Path,help='Original training files when evaluation domains are stored separately.')
    parser.add_argument('--test-scaler',choices=('training','notebook'),default='training',help='Evaluation-only notebook preprocessing ablation.')
    parser.add_argument('--full-unseen-levels',action='store_true',help='Evaluate complete supplied files at unseen intensities, including source IDs seen at training intensities.')
    parser.add_argument('--include-seen-levels',action='store_true',help='Score seen intensities on disjoint TEST source IDs only.')
    parser.add_argument('--burst-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--methods',nargs='+',choices=list(METHOD_NAMES),default=[m for m in METHOD_NAMES if m!='af'])
    parser.add_argument('--endpoint-metrics',action='store_true',help='Observer-only AC/PR/RC/F1 endpoint audit; distinct result tag.')
    parser.add_argument('--allow-af-extension',action='store_true',help='Explicitly reproduce historical AF online extension; NEVER report it as original AF.')
    parser.add_argument('--train-levels',type=int,nargs='+',default=[0,1,3])
    parser.add_argument('--levels',type=int,nargs='+',default=[5,7,10,12,15,17,20])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--budget-fraction',type=float,default=.2)
    parser.add_argument('--dfe-epochs',type=int,default=200)
    parser.add_argument('--af-epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--dfe-lr',type=float,default=.001)
    parser.add_argument('--af-lr',type=float,default=1e-5)
    parser.add_argument('--dfe-alpha1',type=float,default=.5)
    parser.add_argument('--dfe-alpha2',type=float,default=2.)
    parser.add_argument('--dfe-eta',type=float,default=.5)
    parser.add_argument('--dfe-lambda',type=float,default=1.)
    parser.add_argument('--dfe-betas',type=float,nargs=6,default=[.01,.02,.03,.04,.05,.05])
    parser.add_argument('--templates',type=int,default=20)
    parser.add_argument('--arf-trees',type=int,default=10)
    parser.add_argument('--arf-train-passes',type=int,default=1)
    parser.add_argument('--active-models',type=int,default=10,help='CALMID/MicFoal ensemble size.')
    parser.add_argument('--active-window',type=int,default=500,help='Time-based label windows, including placeholders.')
    parser.add_argument('--active-random-ratio',type=float,default=.1)
    parser.add_argument('--calmid-threshold',type=float,default=.5)
    parser.add_argument('--calmid-step',type=float,default=.01)
    parser.add_argument('--micfoal-threshold',type=float,default=.9)
    parser.add_argument('--micfoal-step',type=float,default=.1)
    parser.add_argument('--micfoal-fixed-budget',action='store_true',help='MicFoal_hw ablation; default enables periodic budget adaptation.')
    parser.add_argument('--ral-update-window',type=int,choices=[0,5],default=0,help='Evaluation-only modified RAL: online refit on last 5 acquired rows, no source replay. Zero preserves original RAL.')
    parser.add_argument('--ral-budget-mode',choices=['prefix','global'],default='prefix',
                        help='prefix: paper ratio gate plus common global cap; global: previous budget ablation.')
    parser.add_argument('--progress-every',type=int,default=200,help='Active-method progress interval in stream rows; 0 disables.')
    parser.add_argument('--jobs',type=int,default=-1)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=parser.parse_args()
    if a.ral_update_window and (a.command!='evaluate' or a.methods!=['ral']):
        parser.error('--ral-update-window requires evaluate --methods ral.')
    try:require_explicit_af_extension(a)
    except ValueError as error:parser.error(str(error))
    if not 0<=a.budget_fraction<=1 or min(a.dfe_epochs,a.af_epochs,a.patience,a.templates)<1 or a.batch_size<2:parser.error('Invalid budget/epochs/batch/templates.')
    if min(a.arf_trees,a.arf_train_passes)<1:parser.error('ARF trees and train passes must be positive.')
    if min(a.active_models,a.active_window)<1 or not 0<a.active_random_ratio<=1:parser.error('Invalid active ensemble/window/random ratio.')
    if any(not np.isfinite(v) or not 0<v<=1 for v in (a.calmid_threshold,a.micfoal_threshold)) or any(not np.isfinite(v) or not 0<v<1 for v in (a.calmid_step,a.micfoal_step)):parser.error('Invalid active threshold/step.')
    if a.progress_every<0:parser.error('Progress interval must be nonnegative.')
    if any(not np.isfinite(v) or v<=0 for v in [a.dfe_lr,a.af_lr]):parser.error('Learning rates must be finite and positive.')
    if any(not np.isfinite(v) or v<0 for v in [a.dfe_alpha1,a.dfe_alpha2,a.dfe_eta,a.dfe_lambda,*a.dfe_betas]):parser.error('Invalid DFE objective weights.')
    if len(set(a.methods))!=len(a.methods) or any(len(set(values))!=len(values) or min(values)<0
                                                 for values in (a.levels,a.train_levels)):parser.error('Duplicate methods or invalid levels.')
    core.set_seed(a.seed)
    if a.device!='auto':core.device=torch.device(a.device)
    a.output=a.output.resolve();a.burst_dir=a.burst_dir.resolve()
    a.training_domain_dir=(a.training_domain_dir or a.burst_dir).resolve()
    if a.command=='train' and a.training_domain_dir!=a.burst_dir:parser.error('Training domain directory must match --burst-dir during training.')
    if a.command=='train' and a.test_scaler!='training':parser.error('--test-scaler notebook is evaluation only.')
    if a.command=='train' and a.full_unseen_levels:parser.error('--full-unseen-levels is evaluation only.')
    if a.command=='train' and a.include_seen_levels:parser.error('--include-seen-levels is evaluation only.')
    if a.command=='train':
        if a.source_run is None:parser.error('Training requires --source-run from the final structured AMCAL run.')
        a.source_run=a.source_run.resolve()
        if a.independent_source:
            checkpoint=json.loads((a.source_run/'baseline_source.json').read_text())
        else:
            checkpoint=torch.load(a.source_run/'Models/structured_bank_ddqn.pth',map_location='cpu',weights_only=False)
        if set(checkpoint['training_levels'])!=set(a.train_levels):parser.error('Baseline training domains must match source development domains.')
        for name,digest in checkpoint['model_hashes'].items():
            if core.file_sha256(a.source_run/'Models'/name)!=digest:parser.error('Source checkpoint provenance changed.')
        path=a.source_run/'Models/protocol_preprocessing.joblib'
    else:path=a.output/'Models/protocol_preprocessing.joblib'
    artifact=joblib.load(path)
    partitions=[np.asarray(artifact[key],dtype=np.int64) for key in ('train_ids','val_ids','test_ids')]
    ids=np.concatenate(partitions)
    if any(not len(v) for v in partitions) or len(np.unique(ids))!=len(ids) or (ids<0).any() or (ids>=artifact['source_rows']).any():parser.error('Invalid/disjoint source partitions.')
    if core.file_sha256(a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv')!=artifact['source_sha256']:parser.error('Clean source provenance mismatch.')
    a.classes=[str(c) for c in artifact['label_encoder'].classes_]
    if a.endpoint_metrics:
        if a.command!='evaluate':parser.error('--endpoint-metrics is evaluation only.')
        from baselines.endpoint_metrics import install_baseline
        install_baseline(sys.modules[__name__],range(len(a.classes)))
    if a.command=='train':train(a,artifact)
    else:evaluate(a,artifact)


if __name__=='__main__':main()


