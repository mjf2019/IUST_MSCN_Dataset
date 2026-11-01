"""Clean0 full-refit RAL and corrected author-structure AF at a 20% target cap.

No AMCAL training. RAL is pre-feedback online; AF is post-calibration offline.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import amcal_clean_meta_beta as shared
import amcal_burst_baselines as baseline
from amcal_supplied_light_meta_beta import verify_input
from baselines.ral_existing import RAL
from baselines.af_author_protocol import (
    AFParameters, AUTHOR_COMMIT, PROTOCOL_VERSION, per_class_indices, fit, embed, target_knn)

ROOT = Path(__file__).resolve().parent
api = shared.api
METRICS = ('AC','PR','RC','F1')


def arrays(domain):
    return domain[0].squeeze(1).detach().cpu().numpy(),np.asarray(domain[1]),np.asarray(domain[2])


def scores(truth,prediction,classes):
    result = dict(AC=100*accuracy_score(truth,prediction))
    for average,prefix in [('macro',''),('weighted','weighted_')]:
        pr,rc,f1,_ = precision_recall_fscore_support(truth,prediction,
            labels=list(classes),average=average,zero_division=0)
        result.update({prefix+'PR':100*pr,prefix+'RC':100*rc,prefix+'F1':100*f1})
    return result


def balanced_budget_indices(y,classes,cap,seed):
    """Exact cap, balanced round-robin; leave one query row per class."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    if cap < len(classes):
        raise ValueError('AF budget cannot cover every target class.')
    queues = {}
    for cls in classes:
        at = rng.permutation(np.flatnonzero(y == cls))
        if len(at) < 2:
            raise ValueError(f'AF class {cls} needs support and held-out query rows.')
        queues[int(cls)] = at[:-1].tolist()
    if sum(map(len,queues.values())) < cap:
        raise ValueError('AF exact cap would exhaust the held-out query classes.')
    selected = []
    while len(selected) < cap:
        for cls in rng.permutation(classes):
            if queues[int(cls)]:
                selected.append(queues[int(cls)].pop())
            if len(selected) == cap:
                break
    return np.asarray(sorted(selected),dtype=np.int64)


def af_query_mask(x,source_x,selected):
    blocked = {row.tobytes() for row in source_x}
    blocked.update(x[int(i)].tobytes() for i in selected)
    mask = np.ones(len(x),dtype=bool); mask[selected] = False
    before = int(mask.sum())
    mask &= np.asarray([row.tobytes() not in blocked for row in x])
    return mask,before-int(mask.sum())


class DataView:
    def __init__(self,a,manifest,artifact):
        self.a = a; self.source_root = a.source_root
        self.manifest = manifest; self.artifact = artifact

    def domain(self,directory,level,split):
        return api.base.legacy.read_domain(directory,self.artifact,level,split)


def verify_or_write(path,value):
    if path.exists():
        if shared.read(path) != value:
            raise ValueError('Existing experiment recipe differs; use a fresh --output.')
    else:
        shared.write(path,value)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def train_ral(a,view,protocol):
    file = a.output/'Models/ral_full.joblib'; done = a.output/'training_complete.json'
    if done.exists():
        record = shared.read(done)
        if record['protocol'] != protocol or record['sha256'] != api.digest(file):
            raise ValueError('Completed RAL source model changed.')
        return joblib.load(file)
    if a.action == 'Evaluate':
        raise ValueError('Train the clean-only full RAL first.')
    x,y,ids = arrays(view.domain(a.burst_dir,0,'train_ids'))
    shared.check_classes((torch.as_tensor(x),y,ids),len(a.classes),'RAL clean TRAIN')
    baseline.core.set_seed(a.seed)
    model = RAL([KNeighborsClassifier(n_neighbors=5),
        RandomForestClassifier(n_estimators=10,random_state=a.seed,n_jobs=a.jobs),
        DecisionTreeClassifier(random_state=a.seed)],.9,.025,.01,
        reward=1,penalty=-1,budget=.2,model_update_frequency=1)
    print(f'RAL FULL SOURCE DATASET={a.dataset} TRAIN_LEVELS=[0] ROWS={len(y)} '
        'KNN=5 RF=10 DT | online refit includes all source + all acquired feedback',flush=True)
    start = time.perf_counter(); model.fit(x,y); elapsed = time.perf_counter()-start
    if len(model._training_data_samples.y) != len(y) or getattr(model,'_online_update_window',0):
        raise RuntimeError('RAL full source contract violated.')
    saved = dict(model=model,ral_protocol_version=2,ral_trees=10,source_labels=len(y),
        source_initialization='all clean TRAIN rows; each online refit retains them and all paid feedback',
        protocol=protocol,source_ids=ids.tolist(),training_seconds=elapsed)
    file.parent.mkdir(parents=True,exist_ok=True)
    temporary = file.with_suffix('.tmp'); joblib.dump(saved,temporary,compress=3); temporary.replace(file)
    shared.write(done,dict(protocol=protocol,sha256=api.digest(file),source_labels=len(y),training_seconds=elapsed))
    return saved


def run_ral(a,view,protocol):
    saved = train_ral(a,view,protocol)
    if a.action == 'Train':
        return
    domains,descriptions = shared.attack_cases(view)
    reference = view.domain(a.burst_dir,0,'val_ids')[:2]
    rows = []
    for case,description in descriptions.items():
        target = a.output/'Results'/description['attack']/f'level_{description["level"]}'
        mf,tf,cf = (target/name for name in ('metrics.json','trace.csv','completed.json'))
        expected = dict(protocol=protocol,description=description,
            model_sha256=api.digest(a.output/'Models/ral_full.joblib'),
            target_sha256=shared.base.domain_hash(domains[case]))
        if cf.exists():
            record = shared.read(cf)
            if record['fingerprint'] != expected or any(api.digest(p) != record['sha256'][p.name] for p in (mf,tf)):
                raise ValueError('Completed RAL result changed.')
            m = shared.read(mf)
        else:
            baseline.core.set_seed(a.seed)
            m,trace = baseline.evaluate_level(a,'ral',saved,domains[case],reference)
            m.update(scores([r['truth'] for r in trace],[r['prediction'] for r in trace],range(len(a.classes))))
            m.update(description=description,train_levels=[0],full_refit_events=m['labels'],
                full_source_rows=saved['source_labels'],maximum_rows_per_refit=saved['source_labels']+m['labels'],
                online_window=0,evaluation='pre-feedback online; source + accumulated paid feedback only')
            if (m['samples'] != len(domains[case][1]) or m['label_cap'] != int(.2*m['samples'])
                    or m['labels'] != sum(r['queried'] for r in trace) or m['labels'] > m['label_cap']):
                raise RuntimeError('RAL TEST budget/coverage contract violated.')
            shared.write(mf,m); shared.csv(tf,trace)
            shared.write(cf,dict(fingerprint=expected,sha256={p.name:api.digest(p) for p in (mf,tf)}))
        row = dict(dataset=a.dataset,method='RAL-full',evaluation='online',**description,
            **{k:m[k] for k in (*METRICS,'weighted_PR','weighted_RC','weighted_F1')},
            samples=m['samples'],labels=m['labels'],cap=m['label_cap'],
            labels_percent=100*m['actual_label_fraction'],budget_used_percent=m['budget_utilization_pct'],
            original_forgetting_AC=m['reference']['drop_pp'],
            full_refit_events=m['full_refit_events'],maximum_rows_per_refit=m['maximum_rows_per_refit'],
            milliseconds_per_row=m['online_ms_per_sample'])
        rows.append(row); print_case(row)
        shared.csv(a.output/'Results/summary_by_intensity.csv',rows)
    finish(a,rows)


def run_af(a,view,protocol):
    classes = np.arange(len(a.classes))
    sx,sy,sids = arrays(view.domain(a.burst_dir,0,'train_ids'))
    si = per_class_indices(sy,classes,25,np.random.default_rng(a.seed))
    source_x,source_y,source_ids = sx[si],sy[si],sids[si]
    shared.write(a.output/'clean_source_support.json',dict(ids=source_ids.tolist(),source_levels=[0],
        source_per_class=25,source_labels=len(si),
        scope='author source support; AF model/optimizers are fitted afresh per target intensity'))
    if a.action == 'Train':
        print('AF clean source support prepared; joint AF fit occurs during target calibration in All/Evaluate.',flush=True)
        return
    domains,descriptions = shared.attack_cases(view)
    parameters = AFParameters(iterations=a.iterations,tuning_epochs=a.tuning_epochs)
    tasks = []
    for case,description in descriptions.items():
        x,y,ids = arrays(domains[case]); cap = int(.2*len(y))
        selected = balanced_budget_indices(y,classes,cap,a.seed+81327)
        mask,purged = af_query_mask(x,source_x,selected)
        if np.intersect1d(source_ids,ids).size or not mask.any() or any(not np.any(y[mask]==k) for k in classes):
            raise ValueError('AF source/calibration/query separation lost a class or overlaps IDs.')
        tasks.append((description,x,y,ids,selected,mask,purged,cap))
    rows = []
    for description,x,y,ids,selected,mask,purged,cap in tasks:
        case_parameters = replace(parameters,target_pool_per_class=max(
            int(np.sum(y[selected]==cls)) for cls in classes))
        target = a.output/'Results'/description['attack']/f'level_{description["level"]}'
        mf,pf,cf = (target/name for name in ('metrics.json','predictions.csv','completed.json'))
        k = max(1,cap//len(classes))
        expected = dict(protocol=protocol,description=description,source_ids=source_ids.tolist(),
            support_ids=ids[selected].tolist(),query_ids=ids[mask].tolist(),
            parameters=asdict(case_parameters),k=k,
            target_sha256=shared.base.domain_hash((torch.as_tensor(x),y,ids)))
        if cf.exists():
            record = shared.read(cf)
            if record['fingerprint'] != expected or any(api.digest(p) != record['sha256'][p.name] for p in (mf,pf)):
                raise ValueError('Completed AF result changed.')
            m = shared.read(mf)
        else:
            print(f'AF AUTHOR FULL20 DATASET={a.dataset} ATTACK={description["attack"]} '
                f'STRENGTH={description["strength"]} CALIBRATION={len(selected)}/{cap} '
                f'QUERY={int(mask.sum())} K={k} | offline; fresh model/optimizers',flush=True)
            sync(a.device); start = time.perf_counter()
            model,log = fit(source_x,source_y,x[selected],y[selected],classes,a.seed,case_parameters,a.device)
            sync(a.device); training_seconds = time.perf_counter()-start
            sync(a.device); start = time.perf_counter()
            classifier = target_knn(embed(model.extractor,x[selected],a.device),y[selected],k)
            predicted = classifier.predict(embed(model.extractor,x[mask],a.device))
            sync(a.device); inference_seconds = time.perf_counter()-start
            m = dict(method='AF-author-full20',evaluation='offline post-calibration',description=description,
                **scores(y[mask],predicted,classes),train_levels=[0],source_labels=len(si),
                target_rows_before_calibration=len(y),samples=int(mask.sum()),labels=len(selected),cap=cap,
                labels_percent=100*len(selected)/len(y),budget_used_percent=100*len(selected)/max(cap,1),
                support_counts={str(k):int(np.sum(y[selected]==k)) for k in classes},
                k=k,weights='distance',metric='cosine',query_feature_overlap_rows_purged=purged,
                source_ids=source_ids.tolist(),calibration_ids=ids[selected].tolist(),query_ids=ids[mask].tolist(),
                original_forgetting_AC=None,forgetting_definition='not defined: author model is jointly fitted per target; no clean-only AF checkpoint baseline',
                training_seconds=training_seconds,inference_seconds=inference_seconds,
                milliseconds_per_row=1000*inference_seconds/int(mask.sum()),training=log)
            if m['labels'] != cap:
                raise RuntimeError('AF did not consume the full strict target cap.')
            shared.save(a.output/'Models'/description['attack']/f'level_{description["level"]}.pth',
                dict(state=model.state_dict(),parameters=asdict(case_parameters),support_ids=ids[selected].tolist(),k=k))
            shared.write(mf,m)
            shared.csv(pf,[dict(source_row=int(i),truth=int(t),prediction=int(p)) for i,t,p in zip(ids[mask],y[mask],predicted)])
            shared.write(cf,dict(fingerprint=expected,sha256={p.name:api.digest(p) for p in (mf,pf)}))
        row = dict(dataset=a.dataset,method='AF-author-full20',evaluation='offline post-calibration',**description,
            **{k:m[k] for k in (*METRICS,'weighted_PR','weighted_RC','weighted_F1')},
            samples=m['samples'],target_rows_before_calibration=m['target_rows_before_calibration'],
            labels=m['labels'],cap=m['cap'],labels_percent=m['labels_percent'],
            budget_used_percent=m['budget_used_percent'],original_forgetting_AC=None,
            calibration_seconds=m['training_seconds'],milliseconds_per_row=m['milliseconds_per_row'])
        rows.append(row); print_case(row)
        shared.csv(a.output/'Results/summary_by_intensity.csv',rows)
    finish(a,rows)


def print_case(row):
    values = ' '.join(f'{k}={row[k]:.2f}%' for k in METRICS)
    forgetting = row['original_forgetting_AC']
    extra = 'FORGETTING=not-defined' if forgetting is None else f'ORIGINAL_FORGETTING={forgetting:+.2f}'
    print(f'RESULT DATASET={row["dataset"]} METHOD={row["method"]} EVALUATION={row["evaluation"]} '
        f'ATTACK={row["attack"]} STRENGTH={row["strength"]} {values} N={row["samples"]} '
        f'LABELS={row["labels"]}/{row["cap"]} ({row["labels_percent"]:.2f}%) '
        f'CAP_USED={row["budget_used_percent"]:.2f}% {extra}',flush=True)


def finish(a,rows):
    if len(rows) != 12:
        raise RuntimeError('Incomplete 12-case evaluation.')
    means = []
    for attack in [*shared.ATTACKS,'ALL']:
        chosen = [r for r in rows if attack == 'ALL' or r['attack'] == attack]
        mean = dict(dataset=a.dataset,method=rows[0]['method'],evaluation=rows[0]['evaluation'],
            attack=attack,intensities=len(chosen))
        for key in rows[0]:
            if key in ('dataset','method','evaluation','attack','strength','level'):
                continue
            mean[key] = None if any(r[key] is None for r in chosen) else float(np.mean([r[key] for r in chosen]))
        means.append(mean); print('MEAN '+str(mean),flush=True)
    shared.csv(a.output/'Results/summary_means.csv',means)
    shared.write(a.output/'evaluation_complete.json',dict(cases=12,
        summary_sha256=api.digest(a.output/'Results/summary_by_intensity.csv')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=list(shared.DATASETS),required=True)
    p.add_argument('--method',choices=['RAL','AF'],required=True)
    p.add_argument('--action',choices=['All','Train','Evaluate'],default='All')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--source-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--jobs',type=int,default=-1)
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--ral-budget-mode',choices=['prefix','global'],default='prefix')
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--tuning-epochs',type=int,default=30)
    a = p.parse_args()
    if min(a.iterations,a.tuning_epochs)<1 or a.progress_every<0:
        p.error('Positive AF lengths and nonnegative progress interval required.')
    runs = ROOT/shared.DATASETS[a.dataset][0]/'protocol_runs'
    a.source_root = (a.source_root or runs/f'clean0_AMCAL_attacks_seed{a.seed}').resolve()
    label = 'RAL_full20' if a.method == 'RAL' else 'AF_author_full20'
    a.output = (a.output or runs/f'clean0_{label}_seed{a.seed}_v1').resolve()
    if a.output == a.source_root or a.output in a.source_root.parents or a.source_root in a.output.parents:
        p.error('Output must be separate from the existing source.')
    a.burst_dir = a.source_root/'TrainingDomains'; a.budget_fraction=.2; a.ral_update_window=0
    a.af_lr=1e-5; a.batch_size=32
    a.device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if a.method == 'AF' and a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; choose --device cpu.')
    if a.method == 'RAL':
        a.device='cpu'
    api.base.legacy.core.device=torch.device(a.device); baseline.core.device=torch.device(a.device)
    baseline.core.set_seed(a.seed)
    manifest,artifact,original_marker=verify_input(a)
    a.classes=[str(c) for c in artifact['label_encoder'].classes_]
    view=DataView(a,manifest,artifact)
    protocol=dict(version=1,dataset=a.dataset,method=a.method,seed=a.seed,train_levels=[0],
        validation_levels=[0],budget_fraction=.2,reset_per_intensity=True,
        source_recipe=manifest['recipe'],
        preprocessing_sha256=api.digest(a.source_root/'source/Models/protocol_preprocessing.joblib'),
        partition_sha256={k:hashlib.sha256(np.asarray(artifact[k],dtype='<i8').tobytes()).hexdigest()
                          for k in ('train_ids','val_ids','test_ids')},
        implementation_sha256={name:api.digest(ROOT/name) for name in
            ('amcal_supplied_ral_af20.py','amcal_burst_baselines.py','baselines/ral_existing.py',
             'baselines/af_author_protocol.py','amcal_clean_meta_beta.py','amcal_supplied_light_meta_beta.py')},
        method_settings=(dict(committee='KNN5 / RF10 / DT',refit='all source TRAIN plus all accumulated paid feedback',
            update_frequency=1,window=0,budget_mode=a.ral_budget_mode,evaluation='pre-feedback online')
            if a.method=='RAL' else dict(author_commit=AUTHOR_COMMIT,protocol_version=PROTOCOL_VERSION,
            parameters=asdict(AFParameters(iterations=a.iterations,tuning_epochs=a.tuning_epochs)),
            target_pool_parameter='set per case to maximum observed support count; no implicit 20-row cap',
            source_per_class=25,support='exact floor(.2*N) target labels; balanced capacity-aware round-robin',
            k='floor(total support / number of classes), at least one; determined before fitting',
            adaptation='budget-sized support rather than fixed N-shot; same corrected author fit/heads/optimizers/tuning',
            evaluation='offline post-calibration; support rows excluded; no pre-feedback online claim')))
    pf=a.output/'protocol.json'
    if not pf.exists() and a.output.exists() and any(a.output.iterdir()):
        p.error('Unmatched output preserved; use a fresh --output.')
    verify_or_write(pf,protocol)
    print(f'RUN DATASET={a.dataset} METHOD={a.method} ACTION={a.action} TRAIN=[0] '
        f'FULL_TEST_PER_INTENSITY={len(artifact["test_ids"])} CAP=20% | no AMCAL training; no row cap',flush=True)
    (run_ral if a.method=='RAL' else run_af)(a,view,protocol)
    if any(api.digest(a.source_root/'source'/name)!=sha for name,sha in original_marker['outputs'].items()):
        raise RuntimeError('An original source file changed.')
    print('OUTPUT: '+str(a.output),flush=True)


if __name__=='__main__':
    main()
