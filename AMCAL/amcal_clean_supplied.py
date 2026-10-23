"""Clean-only AMCAL PairOnly batch5 and existing AF/DFE on supplied CSV attacks."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import amcal_batch5_pipeline as pipeline
from amcal_external_data_audit import excluded
from baselines.adversarial_domains import domain_name, digest

DATASETS = {
    'SDNCampus': ('SDNCampus_TEST', 'SDNCampus_original.csv'),
    'ISCX-TOR': ('ISCX_TOR_TEST', 'ISCX_TOR_original.csv'),
    'IUST': ('IUST_MSCN_TEST', 'IUST_MSCN_original.csv'),
    'IoT': ('UNSW_IoT_TEST', 'UNSW_IoT_original.csv'),
}
ATTACKS = {'CW': (.1, 1., 5.), 'DeepFool': (.1, 1., 5.),
           'FGSM': (.01, .1, .5), 'PGD': (.01, .1, .5)}
runner = pipeline.runner
core = runner.legacy.core


@contextmanager
def cwd(path):
    old = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def write(path, value):
    runner.write_json(path, value)


def invoke(values):
    print(' '.join(map(str, values)), flush=True)
    subprocess.run(list(map(str, values)), check=True)


def supplied(root, attack, radius):
    candidates = []
    for path in (root / attack).glob(f'{attack}_eps_*.csv'):
        try:
            if float(path.stem.split('_eps_')[-1]) == radius:
                candidates.append(path)
        except ValueError:
            continue
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one {attack} CSV at strength {radius}: {candidates}')
    return candidates[0]


def prepare(a):
    """Attacks are only exported for TEST; training domains contain clean rows only."""
    source = a.data_dir / DATASETS[a.dataset][1]
    raw = pd.read_csv(source)
    if raw.columns.duplicated().any() or raw.shape[1] < 2:
        raise ValueError('Invalid source schema.')
    label = raw.columns[-1]
    features = [c for c in raw.columns[:-1] if not excluded(c)]
    if not features:
        raise ValueError('No permitted numeric features.')
    frame = raw[features + [label]].copy()
    frame[features] = frame[features].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all() or frame[label].isna().any():
        raise ValueError('Nonfinite clean data; no automatic imputation.')
    ids = np.arange(len(raw))
    # Attack generation used this fixed 80/20 pool. Never change its RNG seed.
    dev, historical_pool = train_test_split(ids, test_size=.2, random_state=42, stratify=raw[label])
    train, val = train_test_split(dev, test_size=.25, random_state=a.seed, stratify=raw.iloc[dev][label])
    groups = core.feature_groups(frame)
    seen = set(); pools = []
    for part in (train, val, historical_pool):
        keep = np.asarray([i for i in part if groups[i] not in seen], dtype=np.int64)
        pools.append(keep)
        seen.update(groups[part])
    train, val, test = pools
    if min(map(len, pools)) < 2 or set(frame.iloc[train][label]) != set(frame[label]):
        raise ValueError('Insufficient separated source partitions or missing TRAIN classes.')
    input_records = []
    for attack, radii in ATTACKS.items():
        for radius in radii:
            path = supplied(a.data_dir, attack, radius)
            mapping = path.with_name(path.stem + '_rows.csv')
            if not mapping.exists() and not a.assume_historical_pool_order:
                raise ValueError('Row map absent: supply --assume-historical-pool-order only if appropriate: ' + str(path))
            input_records.append(dict(attack=attack, strength=radius, path=str(path), sha256=digest(path),
                mapping=str(mapping) if mapping.exists() else None,
                map_sha256=digest(mapping) if mapping.exists() else None))
    recipe = dict(version=1, dataset=a.dataset, seed=a.seed, raw_source_sha256=digest(source),
        source_file=str(source), dropped_features=[c for c in raw.columns[:-1] if c not in features],
        features=features, label=label, training_levels=[0], validation_levels=[0],
        partitions={k:v.tolist() for k,v in zip(('train','validation','test'),pools)},
        split='60/20/20; historical seed42 attack pool held out entirely; exact-feature overlap purge TRAIN > VAL > TEST',
        assume_historical_pool_order=a.assume_historical_pool_order, attacks=input_records,
        budget=.2, independent_levels=True, architecture='original CNN',
        alignment='explicit row maps where supplied; otherwise UNVERIFIED historical stratified seed42 80/20 pool order',
        training_settings=dict(cnn_epochs=a.cnn_epochs, context_epochs=a.context_epochs,
            epochs=a.epochs, utility_epochs=a.utility_epochs, patience=a.patience))
    manifest = a.output / 'clean_supplied_protocol.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['recipe'] != recipe:
            raise ValueError('Inputs/settings changed; use a fresh --output.')
        for relative, expected in old['prepared_hashes'].items():
            if digest(a.output / relative) != expected:
                raise ValueError('Prepared domain changed: ' + relative)
        return a.output / 'TrainingDomains', old
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Partial output preserved; use a fresh --output.')
    training = a.output / 'TrainingDomains'; training.mkdir(parents=True)
    clean = training / domain_name(0)
    frame.to_csv(clean, index=False)
    pd.DataFrame({'source_row':ids}).to_csv(clean.with_name(clean.stem+'_rows.csv'),index=False)
    prepared = [clean, clean.with_name(clean.stem+'_rows.csv')]
    for attack in ATTACKS:
        target = a.output / 'EvaluationDomains' / attack; target.mkdir(parents=True)
        for name in prepared[:2]:
            shutil.copy2(name, target / name.name)
            prepared.append(target / name.name)
        for ordinal, record in enumerate([r for r in input_records if r['attack']==attack], 1):
            path = Path(record['path']); shifted = pd.read_csv(path)
            if list(shifted.columns) != list(raw.columns):
                raise ValueError('Attack/clean schema mismatch: ' + str(path))
            mapped = (pd.to_numeric(pd.read_csv(record['mapping'])['source_row'],errors='raise').to_numpy()
                      if record['mapping'] else historical_pool)
            if (len(mapped)!=len(shifted) or not np.isfinite(mapped).all()
                    or not np.equal(mapped,np.floor(mapped)).all()):
                raise ValueError('Invalid attack mapping: '+str(path))
            mapped = mapped.astype(np.int64)
            if len(np.unique(mapped))!=len(mapped) or (mapped<0).any() or (mapped>=len(raw)).any():
                raise ValueError('Attack source IDs are duplicate/out of range.')
            if not np.array_equal(shifted[label].to_numpy(),raw.iloc[mapped][label].to_numpy()):
                raise ValueError('Attack labels contradict asserted source IDs: '+str(path))
            if not set(test)<=set(mapped):
                raise ValueError('Attack does not cover common held-out TEST IDs: '+str(path))
            selected = shifted[features+[label]].copy()
            selected[features] = selected[features].apply(pd.to_numeric,errors='raise')
            if not np.isfinite(selected[features].to_numpy(dtype=float)).all():
                raise ValueError('Nonfinite attack inputs.')
            output = target/domain_name(ordinal)
            selected.to_csv(output,index=False)
            mapping = output.with_name(output.stem+'_rows.csv')
            pd.DataFrame({'source_row':mapped}).to_csv(mapping,index=False)
            prepared.extend([output,mapping])
    manifest_value = dict(recipe=recipe, prepared_hashes={p.relative_to(a.output).as_posix():digest(p) for p in prepared})
    write(manifest,manifest_value)
    return training, manifest_value


def source_stage(a, training, manifest):
    root=a.output/'source'; marker=root/'clean_source_complete.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        if value['recipe']!=manifest['recipe']:
            raise ValueError('Source recipe changed.')
        for name, expected in value['outputs'].items():
            if digest(root/name)!=expected:raise ValueError('Source checkpoint changed: '+name)
        print('Skipping verified clean source',flush=True)
        return root
    if root.exists() and any(root.iterdir()):raise ValueError('Partial source preserved; use a fresh --output.')
    root.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(training/domain_name(0)); parts=manifest['recipe']['partitions']
    train,val,test=[np.asarray(parts[k],dtype=np.int64) for k in ('train','validation','test')]
    encoder=LabelEncoder().fit(frame.iloc[train,-1]);scaler=StandardScaler().fit(frame.iloc[train,:-1])
    counts=np.bincount(encoder.transform(frame.iloc[train,-1]),minlength=len(encoder.classes_))
    core.set_seed(a.seed);core.RUN_SEED=a.seed
    with cwd(root):core.save_preprocessing(frame,str(training),domain_name(0),scaler,encoder,counts,train,val,test)
    models=root/'Models';artifact=joblib.load(models/'protocol_preprocessing.joblib')
    # Own partitions were checked above; no inherited weights or preprocessing.
    a.classes=list(encoder.classes_)
    values=[pipeline.domain(a,artifact,0,k) for k in ('train_ids','val_ids')]
    datasets=[(v[0],torch.as_tensor(v[1],dtype=torch.long,device=core.device)) for v in values]
    cnn=core.CNN1D(len(artifact['feature_names']),len(a.classes)).to(core.device)
    cnn_log=pipeline.fit(a,cnn,None,*datasets)
    cnn.eval();cnn.requires_grad_(False)
    preprocessing=digest(models/'protocol_preprocessing.joblib')
    torch.save(dict(architecture='legacy',preprocessing_sha256=preprocessing,model_state_dict=cnn.state_dict(),
        input_length=len(artifact['feature_names']),num_classes=len(a.classes)),models/'initial_cnn_model.pth')
    context=core.ContextAwareNetwork(len(artifact['feature_names'])).to(core.device)
    context_log=pipeline.fit(a,context,cnn,*datasets,context=True)
    torch.save(dict(preprocessing_sha256=preprocessing,context_model_state_dict=context.state_dict()),models/'adversarial_best_context.pth')
    torch.save(dict(training_perturbations=[0],scope='clean source provenance; selector trained in structured stage'),models/'adversarial_best_dqn.pth')
    write(root/'source_training.json',dict(training_levels=[0],architecture='legacy',cnn=cnn_log,context=context_log,
        source_partition_sha256=preprocessing,selection='clean VAL loss only; no TEST selection'))
    write(marker,dict(recipe=manifest['recipe'],outputs={p.relative_to(root).as_posix():digest(p) for p in models.iterdir() if p.is_file()}))
    return root


def train_amcal(a, source):
    pipeline.TRAIN=[0];pipeline.TEST=[1,2,3]
    stages,_=pipeline.make_plan(a)
    # The source above is freshly trained without a template or pretrained model.
    for stage in stages[1:]:pipeline.execute(a,stage)


def evaluate_amcal(a):
    final=a.output/'paironly_batch5'
    if not (final/'batch5_stage_complete.json').exists():raise ValueError('Complete AMCAL training first.')
    for attack in ATTACKS:
        # Each evaluate call reloads CNN/Context/RL and creates fresh feedback memory.
        for level in (1,2,3):
            runner.main(['evaluate','--output',str(final),'--burst-dir',str(a.output/'EvaluationDomains'/attack),
                '--levels',str(level),'--context-count','1','--query-schedule','policy','--update-policy','learned',
                '--modes','learned','--paid-replay-mode','batch5','--context-lr','.00025','--budget-fraction','.2',
                '--eval-epsilon','.05','--query-budget-bonus','1','--online-memory-ce-weight','.1',
                '--routing-oracle-audit','--skip-existing-completed','--evaluation-tag',f'clean0_{attack}_independent_budget20',
                '--seed',str(a.seed),'--device',a.device])


def baseline(a,source,train):
    script=Path(__file__).with_name('amcal_burst_baselines.py')
    # DFE and historical AF-online reuse the existing implementations unchanged.
    for method in [m.lower() for m in a.methods if m!='AMCAL']:
        target=a.output/('AF_online_extension' if method=='af' else 'DFE')
        common=[sys.executable,script,'train' if train else 'evaluate','--source-run',source,
            '--output',target,'--methods',method,'--train-levels','0','--budget-fraction','.2','--seed',a.seed,'--device',a.device]
        if method=='af':common+=['--allow-af-extension']
        if train:
            if target.exists():
                saved=joblib.load(target/'Models'/f'{method}.joblib')
                if (set(saved['provenance']['train_levels'])!={0} or
                    saved['provenance']['source_preprocessing_sha256']!=digest(source/'Models/protocol_preprocessing.joblib')):
                    raise ValueError('Existing baseline source mismatch; use fresh output.')
                print('Skipping existing completed baseline',method,flush=True)
                continue
            if method=='af':
                train_clean_af(a,source,target)
            else:
                invoke(common+['--burst-dir',a.burst_dir,'--independent-source'])
        else:
            for attack in ATTACKS:
                # Separate result folders avoid ordinal 1/2/3 collisions across attacks.
                run=a.output/'BaselineEvaluation'/method/attack
                models=run/'Models';models.mkdir(parents=True,exist_ok=True)
                for name in (f'{method}.joblib','protocol_preprocessing.joblib'):
                    dst=models/name;src=target/'Models'/name
                    if dst.exists() and digest(dst)!=digest(src):raise ValueError('Evaluation checkpoint mismatch.')
                    if not dst.exists():shutil.copy2(src,dst)
                values=list(common);values[values.index('--output')+1]=run
                invoke(values+['--burst-dir',a.output/'EvaluationDomains'/attack,'--training-domain-dir',a.burst_dir,
                               '--levels',1,2,3])


def train_clean_af(a,source,target):
    """Source-only initialization; no attack target exists before inference.

    This uses the existing tabular AF network/online adapter, not original AF's
    offline target-calibrated few-shot protocol. Initial fit is supervised clean
    CE; online DANN steps remain the existing paid-feedback implementation.
    """
    import hashlib
    import torch.nn.functional as F
    import amcal_burst_baselines as existing
    existing.core.device=core.device;core.set_seed(a.seed)
    artifact=joblib.load(source/'Models/protocol_preprocessing.joblib')
    labels=list(artifact['label_encoder'].classes_)
    domains=[pipeline.domain(a,artifact,0,k) for k in ('train_ids','val_ids')]
    (x,y),(vx,vy)=[(v[0].squeeze(1).cpu().numpy(),v[1]) for v in domains]
    model=existing.af_network(x.shape[1],len(labels),core.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    rng=np.random.default_rng(a.seed+701);logs=[]
    for epoch in range(30):
        order=rng.permutation(len(y));losses=[];model.train()
        for start in range(0,len(y),32):
            ids=order[start:start+32]
            if len(ids)<2:continue
            bx=torch.as_tensor(x[ids],device=core.device)
            by=torch.as_tensor(y[ids],device=core.device)
            logits,_=model(bx)
            loss=F.cross_entropy(logits,by)
            if not torch.isfinite(loss):raise ValueError('Nonfinite AF source loss.')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        logs.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        print('AF clean-source CE',logs[-1],flush=True)
    model.eval()
    k,score=existing.choose_k(existing.af_embed(model,x,core.device),y,existing.af_embed(model,vx,core.device),vy)
    models=target/'Models';models.mkdir(parents=True)
    shutil.copy2(source/'Models/protocol_preprocessing.joblib',models/'protocol_preprocessing.joblib')
    provenance=dict(version=1,train_levels=[0],
        training_domains_sha256={'0':digest(a.burst_dir/domain_name(0))},
        source_preprocessing_sha256=digest(models/'protocol_preprocessing.joblib'),
        partition_sha256={key:hashlib.sha256(np.asarray(artifact[key],dtype='<i8').tobytes()).hexdigest()
                          for key in ('train_ids','val_ids','test_ids')},
        classes=[str(c) for c in labels],feature_names=artifact['feature_names'],
        config=dict(seed=a.seed,initialization='clean supervised CE only; no target domain',epochs=30,lr=1e-5,
                    architecture='existing tabular AF network',online='existing historical AF paid-feedback DANN extension; NOT original AF'))
    saved=dict(state=existing.cpu_state(model),source_x=x,source_y=y,k=k,
        dann_source_x=x,dann_source_y=y,validation_macro_f1=score,provenance=provenance,logs=logs)
    joblib.dump(saved,models/'af.joblib')
    write(target/'training_manifest.json',provenance)


def collect(a):
    rows=[]
    if 'AMCAL' in a.methods:
        for path in sorted((a.output/'paironly_batch5/Results').glob('evaluation_*/*_metrics.json')):
            m=json.loads(path.read_text());e=m.get('execution',{})
            tag=e.get('evaluation_tag','')
            for attack,radii in ATTACKS.items():
                if tag!=f'clean0_{attack}_independent_budget20':continue
                level=m['level']
                rows.append(dict(Dataset=a.dataset,Method='AMCAL-PairOnly-batch5',Attack=attack,Strength=radii[level-1],
                    N=m['samples'],Accuracy=100*m['accuracy'],MacroF1=100*m['macro_f1'],Labels=m['labels'],Cap=m['label_cap'],
                    SourceDrop=m['reference']['drop_pp'],MsPerRow=m['online_ms_per_sample']))
    for method in [m.lower() for m in a.methods if m!='AMCAL']:
        for attack,radii in ATTACKS.items():
            root=a.output/'BaselineEvaluation'/method/attack/'Results'
            for path in sorted(root.glob(f'{method}_per*_metrics.json')):
                m=json.loads(path.read_text());level=m['level']
                rows.append(dict(Dataset=a.dataset,Method='AF-clean-source-online-extension' if method=='af' else 'DFE-existing-online-adapter',
                    Attack=attack,Strength=radii[level-1],N=m['samples'],Accuracy=100*m['accuracy'],MacroF1=100*m['macro_f1'],
                    Labels=m['labels'],Cap=m['label_cap'],SourceDrop=m['reference']['drop_pp'],MsPerRow=m['online_ms_per_sample']))
    # Never silently declare a partial evaluation complete.
    if len(rows)!=12*len(a.methods):raise ValueError('Incomplete/duplicate result set; inspect per-level files.')
    runner.write_csv(a.output/'Results/clean_attack_comparison.csv',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--methods',nargs='+',choices=['AMCAL','AF','DFE'],default=['AMCAL','AF','DFE'])
    p.add_argument('--data-dir',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--assume-historical-pool-order',action='store_true')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--cnn-epochs',type=int,default=60);p.add_argument('--context-epochs',type=int,default=20)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--utility-epochs',type=int,default=3)
    p.add_argument('--patience',type=int,default=5)
    a=p.parse_args()
    if len(set(a.methods))!=len(a.methods) or min(a.cnn_epochs,a.context_epochs,a.epochs,a.utility_epochs,a.patience)<1:
        p.error('Distinct methods and positive training lengths required.')
    root=Path('AMCAL')/DATASETS[a.dataset][0]
    a.data_dir=(a.data_dir or root/'Dataset').resolve()
    a.output=(a.output or root/'protocol_runs'/f'clean0_paironly_batch5_attacks_seed{a.seed}').resolve()
    device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable.')
    a.device=device;core.device=torch.device(device);core.set_seed(a.seed)
    training,manifest=prepare(a);a.burst_dir=training
    a.clean_only=True;a.light_cnn=False;a.original_cnn=True;a.batch_size=5
    a.output_root=a.output;a.source_run=a.output/'source'
    a.online_context_lr=.00025;a.query_budget_bonus=1.;a.replay_weight=.1;a.evaluation_tag=''
    if a.action in ('All','Train'):
        source=source_stage(a,training,manifest)
        models=source/'Models'
        write(source/'baseline_source.json',dict(training_levels=[0],seed=a.seed,
            model_hashes={n:digest(models/n) for n in runner.SOURCE_FILES},test_used=False))
        if 'AMCAL' in a.methods:train_amcal(a,source)
        baseline(a,source,True)
    if a.action in ('All','Evaluate'):
        if 'AMCAL' in a.methods:evaluate_amcal(a)
        baseline(a,a.source_run,False)
        collect(a)
    print('Outputs:',a.output,flush=True)


if __name__=='__main__':main()
