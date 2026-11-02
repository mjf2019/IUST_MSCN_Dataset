"""Checkpoint AF with pre-feedback stream scores and paired retention audits."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import joblib
import numpy as np
import torch

import amcal_supplied_ral_af20 as existing
from baselines.af_author_protocol import AFNetwork, AFParameters, fit, embed, target_knn

ROOT = Path(__file__).resolve().parent
DATASETS = {**existing.shared.DATASETS,
            'IAT': ('ANT_Burst_IAT_TEST', ''), 'PS': ('ANT_Burst_PS_TEST', '')}
METRICS = ('AC', 'PR', 'RC', 'F1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def clean_split(x, y, ids, classes, seed):
    """25/class source; exactly floor(.2 * original clean TRAIN count)."""
    si = existing.per_class_indices(y, classes, 25, np.random.default_rng(seed))
    keys = {row.tobytes() for row in x[si]}
    at = np.flatnonzero([row.tobytes() not in keys for row in x])
    ci = at[existing.balanced_budget_indices(y[at], classes, int(.2*len(y)), seed+81327)]
    if np.intersect1d(ids[si], ids[ci]).size:
        raise ValueError('Clean source and calibration IDs overlap.')
    return si, ci


def shared_reference_mask(x, ids, blocked_x, blocked_ids, y, classes):
    keys = {row.tobytes() for part in blocked_x for row in part}
    seen = {int(v) for part in blocked_ids for v in part}
    mask = np.asarray([int(i) not in seen and row.tobytes() not in keys
                       for row, i in zip(x, ids)])
    if not mask.any() or set(map(int, y[mask])) != set(map(int, classes)):
        raise ValueError('Common clean VAL reference loses a class after overlap purge.')
    return mask


def metric_drop(initial, final):
    return {key: float(initial[key]-final[key]) for key in METRICS}


def prefix_indices(rows, fraction=.2):
    """A label-blind causal prefix. Never rebalance using future labels."""
    cap = int(fraction*rows)
    if cap < 2 or cap >= rows:
        raise ValueError('AF prefix calibration needs >=2 labels and remaining stream rows.')
    return np.arange(cap, dtype=np.int64)


def prequential_batch_calibration(x, cap, predict_initial, adapt):
    """Predict the entire prefix before its one batch update; then the suffix.

    adapt receives only prefix positions and returns the final predictor plus
    metadata. This helper never receives truth labels or re-scores the prefix.
    Batched prediction is causal because weights stay fixed within each phase.
    """
    if not 0 < cap < len(x):
        raise ValueError('Calibration prefix must leave a nonempty suffix.')
    prefix = np.asarray(predict_initial(x[:cap])).copy()
    if prefix.shape != (cap,):
        raise ValueError('Initial predictor returned the wrong number of rows.')
    predict_final, metadata = adapt(np.arange(cap, dtype=np.int64))
    suffix = np.asarray(predict_final(x[cap:])).copy()
    if suffix.shape != (len(x)-cap,):
        raise ValueError('Final predictor returned the wrong number of rows.')
    return np.concatenate((prefix, suffix)), metadata


def digest(path):
    return existing.api.digest(path)


def domain_hash(x, y, ids):
    return existing.shared.base.domain_hash((torch.as_tensor(x), y, ids))


def parameters_for(y, classes, a):
    return replace(AFParameters(iterations=a.iterations, tuning_epochs=a.tuning_epochs),
                   target_pool_per_class=max(int(np.sum(y == cls)) for cls in classes))


def predict(saved, support_x, support_y, x, classes, device):
    model = AFNetwork(len(classes), AFParameters(**saved['parameters'])).to(device)
    model.load_state_dict(saved['state'], strict=True)
    model.eval()
    if not 1 <= saved['k'] <= len(support_y):
        raise ValueError('Invalid AF k.')
    knn = target_knn(embed(model.extractor, support_x, device), support_y, saved['k'])
    return knn.predict(embed(model.extractor, x, device))


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def compatible_clean_checkpoint(old_recipe, new_recipe):
    """Only clean-fitting inputs/settings matter; target evaluation can change."""
    keys = ('dataset','seed','train_levels','source_per_class','source_ids',
            'calibration_ids','clean_train_rows','clean_calibration_rows',
            'calibration_denominator','parameters','clean_csv_sha256','clean_train_hash',
            'original_input_hashes')
    return all(key in old_recipe and old_recipe[key] == new_recipe[key] for key in keys)


def load_view(a):
    runs = ROOT / DATASETS[a.dataset][0] / 'protocol_runs'
    supplied = a.dataset not in ('IAT', 'PS')
    a.source_root = (a.source_root or runs/f'clean0_AMCAL_attacks_seed{a.seed}').resolve()
    if supplied:
        a.burst_dir = a.source_root/'TrainingDomains'
        manifest, artifact, marker = existing.verify_input(a)
        view = existing.DataView(a, manifest, artifact)
        models = a.source_root/'source/Models'
        source_files = [a.source_root/'source'/name for name in marker['outputs']]
    else:
        models = a.source_root/'Models'
        artifact = joblib.load(models/'protocol_preprocessing.joblib')
        from amcal_af_paper_burst import shared
        clean = a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'
        if digest(clean) != artifact['source_sha256']:
            raise ValueError('Burst clean CSV differs from preprocessing.')
        view = SimpleNamespace(domain=lambda directory, level, split:
                               shared.read_domain(directory, artifact, level, split))
        source_files = [models/'protocol_preprocessing.joblib']
    parts = [np.asarray(artifact[key]) for key in ('train_ids','val_ids','test_ids')]
    all_ids = np.concatenate(parts)
    if any(not len(p) for p in parts) or len(np.unique(all_ids)) != len(all_ids):
        raise ValueError('Source partitions must be nonempty and disjoint.')
    a.classes = np.arange(len(artifact['label_encoder'].classes_))
    a.preprocessing = models/'protocol_preprocessing.joblib'
    a.output = (a.output or runs/f'AF_checkpoint20_online_seed{a.seed}_v2').resolve()
    if a.output == a.source_root or a.source_root in a.output.parents or a.output in a.source_root.parents:
        raise ValueError('Output must be separate from original source.')
    return view, supplied, source_files


def prepare_tasks(a, view, supplied, clean_fit_x):
    if supplied:
        domains, descriptions = existing.shared.attack_cases(view)
    else:
        domains = {level:view.domain(a.burst_dir, level, 'test_ids')
                   for level in (5,7,10,12,15,17,20)}
        descriptions = {level:dict(attack='Burst', strength=level, level=level) for level in domains}
    tasks = []
    for key, domain in domains.items():
        x, y, ids = existing.arrays(domain)
        raw_rows = len(y)
        # Same seed-only permutation as the baseline evaluator; no target-label
        # stratification or class-balancing of the acquired prefix.
        order = np.random.default_rng(a.seed).permutation(len(y))
        x, y, ids = x[order], y[order], ids[order]
        eligible, clean_duplicates = existing.af_query_mask(x, clean_fit_x, [])
        x, y, ids = x[eligible], y[eligible], ids[eligible]
        selected = prefix_indices(len(y))
        # This mask is ONLY for independent before/after target observers.
        # All eligible stream rows, including selected/duplicate rows, are scored
        # online before their own feedback. Observer labels never affect fitting.
        mask, purged = existing.af_query_mask(x, clean_fit_x, selected)
        if not mask.any() or len(np.unique(ids)) != len(ids):
            raise ValueError('Target query empty or stream IDs duplicated.')
        tasks.append(dict(description=descriptions[key], x=x, y=y, ids=ids,
                          selected=selected, mask=mask, purged=purged, raw_rows=raw_rows,
                          clean_duplicates=clean_duplicates,
                          hash=domain_hash(x,y,ids)))
    return tasks


def run(a):
    view, supplied, source_files = load_view(a)
    x, y, ids = existing.arrays(view.domain(a.burst_dir, 0, 'train_ids'))
    si, ci = clean_split(x, y, ids, a.classes, a.seed)
    tasks = prepare_tasks(a, view, supplied, np.concatenate((x[si],x[ci])))
    if any(np.intersect1d(ids[si], task['ids']).size or np.intersect1d(ids[ci], task['ids']).size for task in tasks):
        raise ValueError('Clean fitting IDs overlap target evaluation IDs.')
    rx, ry, rids = existing.arrays(view.domain(a.burst_dir, 0, 'val_ids'))
    reference_mask = shared_reference_mask(rx, rids,
        [x[si], x[ci]] + [task['x'][task['selected']] for task in tasks],
        [ids[si], ids[ci]] + [task['ids'][task['selected']] for task in tasks], ry, a.classes)
    rx, ry, rids = rx[reference_mask], ry[reference_mask], rids[reference_mask]
    parameters = parameters_for(y[ci], a.classes, a)
    source_hashes = {str(p):digest(p) for p in sorted(set(source_files+[a.preprocessing]), key=str)}
    recipe = dict(version=2, dataset=a.dataset, seed=a.seed, train_levels=[0],
        source_per_class=25, source_ids=ids[si].tolist(), calibration_ids=ids[ci].tolist(),
        clean_train_rows=len(y), clean_calibration_rows=len(ci), budget_fraction=.2,
        calibration_denominator='all clean TRAIN before source25 selection',
        parameters=asdict(parameters), clean_reference_ids=rids.tolist(),
        clean_reference_rows_purged=int((~reference_mask).sum()),
        clean_csv_sha256=digest(a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'),
        clean_train_hash=domain_hash(x,y,ids), original_input_hashes=source_hashes,
        cases=[dict(**task['description'], target_hash=task['hash'],
                    raw_rows=task['raw_rows'], stream_rows=len(task['y']),
                    clean_feature_duplicate_rows_excluded=task['clean_duplicates'],
                    support_ids=task['ids'][task['selected']].tolist(),
                    online_stream_ids=task['ids'].tolist(),
                    query_ids=task['ids'][task['mask']].tolist()) for task in tasks],
        implementation_sha256={name:digest(ROOT/name) for name in
            ('amcal_af_clean_reference.py','baselines/af_author_protocol.py')},
        reset='same clean checkpoint per intensity; fresh optimizers; no transfer across target cases',
        author_procedure='two optimizers; author feature-layer reset and supervised tuning retained',
        query_schedule='first floor(.2*N) eligible stream positions; seed-only permutation; no future-label class balancing',
        evaluation='pre-feedback predictions on every eligible stream row including calibration; one batch update after prefix',
        observers='common original VAL and fixed unqueried target rows; excluded exact fitting feature duplicates')
    rp = a.output/'recipe.json'
    if rp.exists() and read(rp) != recipe:
        raise ValueError('Reference recipe differs; use a fresh --output.')
    if not rp.exists() and a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Unmatched output preserved; use a fresh --output.')
    if not rp.exists():
        write(rp, recipe)
    checkpoint = a.output/'Models/af_clean_reference.pth'
    completed = a.output/'training_complete.json'
    if completed.exists():
        if read(completed)['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Clean AF checkpoint changed.')
        print('Skipping verified complete clean AF checkpoint.', flush=True)
    else:
        previous = a.output.parent/f'AF_checkpoint20_seed{a.seed}_v1'
        old_checkpoint = previous/'Models/af_clean_reference.pth'
        old_done = previous/'training_complete.json'
        old_recipe = previous/'recipe.json'
        reused = False
        if old_checkpoint.exists() and old_done.exists() and old_recipe.exists():
            if not compatible_clean_checkpoint(read(old_recipe), recipe):
                raise ValueError('Previous clean checkpoint settings differ; specify a fresh output/source recipe.')
            previous_record = read(old_done)
            if previous_record['checkpoint_sha256'] != digest(old_checkpoint):
                raise ValueError('Previous clean checkpoint hash changed.')
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old_checkpoint,checkpoint)
            write(completed,dict(previous_record,reused_from=str(old_checkpoint)))
            print('Reusing verified existing level0 AF checkpoint; no clean retraining.',flush=True)
            reused = True
        if not reused:
            train_clean(a,x,y,si,ci,ids[ci].tolist(),parameters,checkpoint,completed)
    clean_sha = digest(checkpoint)
    clean = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if clean['support_ids'] != ids[ci].tolist() or clean['preprocessing_sha256'] != digest(a.preprocessing):
        raise ValueError('Clean checkpoint support or preprocessing differs from recipe.')
    before_original = predict(clean, x[ci], y[ci], rx, a.classes, a.device)
    initial_original = existing.scores(ry, before_original, a.classes)
    write(a.output/'clean_reference_metrics.json',dict(scores=initial_original, reference_ids=rids.tolist()))
    evaluate_tasks(a,tasks,x,y,si,ci,rx,ry,rids,initial_original,before_original,clean,clean_sha,checkpoint,rp,source_hashes)


def train_clean(a,x,y,si,ci,support_ids,parameters,checkpoint,completed):
    print(f'AF CLEAN CHECKPOINT DATASET={a.dataset} SOURCE25={len(si)} '
          f'CLEAN_CALIBRATION={len(ci)}/{len(y)} | author fit; level0 only', flush=True)
    tick = time.perf_counter()
    model, log = fit(x[si], y[si], x[ci], y[ci], a.classes, a.seed, parameters, a.device)
    existing.shared.save(checkpoint, dict(state=cpu_state(model), parameters=asdict(parameters),
        support_ids=support_ids, k=max(1,len(ci)//len(a.classes)),
        preprocessing_sha256=digest(a.preprocessing)))
    write(completed, dict(checkpoint_sha256=digest(checkpoint), training=log,
                          training_seconds=time.perf_counter()-tick))
    del model


def evaluate_tasks(a,tasks,x,y,si,ci,rx,ry,rids,initial_original,before_original,
                   clean,clean_sha,checkpoint,rp,source_hashes):
    rows = []
    for task in tasks:
        description = task['description']
        tx, ty, tids = task['x'], task['y'], task['ids']
        selected, mask = task['selected'], task['mask']
        out = a.output/'Results'/description['attack']/f'level_{description["level"]}'
        mf, cf = out/'metrics.json', out/'completed.json'
        target_checkpoint = a.output/'Models'/description['attack']/f'level_{description["level"]}.pth'
        pf = out/'paired_predictions.csv'
        trace_file = out/'online_trace.csv'
        fingerprint = dict(recipe_sha256=digest(rp), clean_checkpoint_sha256=clean_sha,
                           **description, target_hash=task['hash'])
        if cf.exists():
            record = read(cf)
            paths = (mf, pf, trace_file, target_checkpoint)
            if record['fingerprint'] != fingerprint or any(digest(path) != record['hashes'][str(path.relative_to(a.output))] for path in paths):
                raise ValueError('Completed AF case changed.')
            row = read(mf)
            print(f'Skipping verified AF case {a.dataset} {description}.', flush=True)
        else:
            print(f'AF CONTINUE DATASET={a.dataset} CASE={description} CLEAN_CHECKPOINT={clean_sha[:12]} '
                  f'CALIBRATION={len(selected)}/{len(ty)} QUERY={int(mask.sum())} '
                  '| predict prefix -> batch calibration -> predict suffix; fresh optimizers', flush=True)
            before_target = predict(clean, x[ci], y[ci], tx[mask], a.classes, a.device)
            initial_target = existing.scores(ty[mask], before_target, a.classes)
            def adapt(prefix):
                if not np.array_equal(prefix, selected):
                    raise ValueError('AF update attempted to use non-prefix feedback.')
                p = parameters_for(ty[prefix], a.classes, a)
                tick = time.perf_counter()
                model, log = fit(x[si], y[si], tx[prefix], ty[prefix], a.classes, a.seed,
                    p, a.device, initial_state={key:value.clone() for key,value in clean['state'].items()})
                elapsed = time.perf_counter()-tick
                saved = dict(state=cpu_state(model), parameters=asdict(p),
                    support_ids=tids[prefix].tolist(), k=max(1,len(prefix)//len(a.classes)),
                    parent_clean_checkpoint_sha256=clean_sha)
                del model
                return (lambda batch:predict(saved,tx[prefix],ty[prefix],batch,a.classes,a.device)), dict(
                    saved=saved, log=log, training_seconds=elapsed)
            stream_tick = time.perf_counter()
            online_predictions, metadata = prequential_batch_calibration(tx, len(selected),
                lambda batch:predict(clean,x[ci],y[ci],batch,a.classes,a.device), adapt)
            stream_seconds = time.perf_counter()-stream_tick
            saved, log = metadata['saved'], metadata['log']
            training_seconds = metadata['training_seconds']
            existing.shared.save(target_checkpoint,saved)
            online = existing.scores(ty,online_predictions,a.classes)
            after_target = predict(saved, tx[selected], ty[selected], tx[mask], a.classes, a.device)
            after_original = predict(saved, tx[selected], ty[selected], rx, a.classes, a.device)
            final_target = existing.scores(ty[mask], after_target, a.classes)
            final_original = existing.scores(ry, after_original, a.classes)
            row = dict(dataset=a.dataset, seed=a.seed, **description,
                evaluation='pre-feedback prefix-batch AF', samples=len(ty), observer_samples=int(mask.sum()),
                target_rows=len(ty), labels=len(selected), cap=int(.2*len(ty)),
                labels_percent=100*len(selected)/len(ty), reference_samples=len(ry),
                **{'online_'+metric:online[metric] for metric in METRICS},
                calibration_prefix_online_AC=100*float(np.mean(online_predictions[selected]==ty[selected])),
                calibration_prefix_samples=len(selected), online_update_count=1,
                update_after_stream_position=len(selected), stream_seconds=stream_seconds,
                milliseconds_per_row=1000*stream_seconds/len(ty),
                support_class_counts={str(c):int(np.sum(ty[selected]==c)) for c in a.classes},
                raw_target_rows=task['raw_rows'], clean_duplicate_rows_excluded=task['clean_duplicates'],
                training_seconds=training_seconds)
            for scope, initial, final in [('original',initial_original,final_original),
                                           ('same_domain',initial_target,final_target)]:
                drops = metric_drop(initial,final)
                for metric in METRICS:
                    row[f'{scope}_initial_{metric}'] = initial[metric]
                    row[f'{scope}_final_{metric}'] = final[metric]
                    row[f'{scope}_forgetting_{metric}'] = drops[metric]
                    row[f'{scope}_gain_{metric}'] = -drops[metric]
            write(mf,row)
            write(out/'training.json',log)
            existing.shared.csv(trace_file,[dict(position=i+1,source_row=int(tids[i]),truth=int(ty[i]),
                prediction_before_feedback=int(online_predictions[i]),
                label_requested=int(i<len(selected)),
                model='clean_checkpoint' if i<len(selected) else 'adapted_checkpoint',
                labels_used_before_prediction=min(i,len(selected)),
                batch_update_after_prediction=int(i==len(selected)-1)) for i in range(len(ty))])
            existing.shared.csv(pf,[dict(domain=scope, source_row=int(i), truth=int(t),
                initial_prediction=int(b), final_prediction=int(f))
                for scope, dids, truth, before, after in
                [('original0',rids,ry,before_original,after_original),
                 ('same_target',tids[mask],ty[mask],before_target,after_target)]
                for i,t,b,f in zip(dids,truth,before,after)])
            write(cf,dict(fingerprint=fingerprint,hashes={
                str(path.relative_to(a.output)):digest(path) for path in (mf,pf,trace_file,target_checkpoint)}))
        rows.append(row)
        print(f'AF RESULT {a.dataset} {description} '
              f'ONLINE_AC={row["online_AC"]:.2f}% ONLINE_F1={row["online_F1"]:.2f}% '
              f'CALIBRATION_AC={row["calibration_prefix_online_AC"]:.2f}% '
              f'TARGET_AC={row["same_domain_initial_AC"]:.2f}->{row["same_domain_final_AC"]:.2f}% '
              f'ORIGINAL_AC={row["original_initial_AC"]:.2f}->{row["original_final_AC"]:.2f}% '
              f'FORGET_TARGET={row["same_domain_forgetting_AC"]:+.2f} '
              f'FORGET_ORIGINAL={row["original_forgetting_AC"]:+.2f} '
              f'LABELS={row["labels"]}/{row["cap"]}',flush=True)
        existing.shared.csv(a.output/'Results/summary_by_intensity.csv', rows)
    means = []
    for attack in sorted({row['attack'] for row in rows})+['ALL']:
        selected_rows = [row for row in rows if attack == 'ALL' or row['attack'] == attack]
        numeric = [key for key in rows[0] if key.startswith(('original_','same_domain_','online_'))
                   or key in ('labels_percent','calibration_prefix_online_AC','milliseconds_per_row')]
        means.append(dict(dataset=a.dataset, seed=a.seed, attack=attack, cases=len(selected_rows),
            **{key:float(np.mean([row[key] for row in selected_rows])) for key in numeric}))
    existing.shared.csv(a.output/'Results/summary_means.csv', means)
    if digest(checkpoint) != clean_sha or any(digest(Path(path)) != sha for path,sha in source_hashes.items()):
        raise RuntimeError('Clean checkpoint or an original source input changed.')
    write(a.output/'evaluation_complete.json',dict(cases=len(rows),
        summary_sha256=digest(a.output/'Results/summary_by_intensity.csv'), recipe_sha256=digest(rp)))
    print('MEAN '+json.dumps(means[-1]),flush=True)
    print('OUTPUT: '+str(a.output),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=list(DATASETS),required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--source-root',type=Path)
    p.add_argument('--burst-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--tuning-epochs',type=int,default=30)
    a = p.parse_args()
    if min(a.iterations,a.tuning_epochs) < 1:
        p.error('Positive training lengths required.')
    if a.dataset in ('IAT','PS') and (a.source_root is None or a.burst_dir is None):
        p.error('Burst requires --source-root and --burst-dir.')
    a.device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use --device cpu.')
    existing.api.base.legacy.core.device = torch.device(a.device)
    run(a)


if __name__ == '__main__':
    main()
