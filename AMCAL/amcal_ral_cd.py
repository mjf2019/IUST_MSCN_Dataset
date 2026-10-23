"""Independent Burst RAL-CD runner; no changes to historical RAL or AMCAL runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from copy import deepcopy


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['IAT', 'PS'], required=True)
    p.add_argument('--action', choices=['All', 'Train', 'Test'], default='All')
    p.add_argument('--source-run', type=Path, required=True,
                   help='Existing preprocessing source; no AMCAL training is performed.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--burst-dir', type=Path)
    p.add_argument('--train-levels', type=int, nargs='+', default=[0, 1, 3])
    p.add_argument('--levels', type=int, nargs='+', default=[5, 7, 10, 12, 15, 17, 20])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--budget-fraction', type=float, default=.2)
    p.add_argument('--adwin-backend', choices=['river', 'skmultiflow'], default='river')
    p.add_argument('--adwin-delta', type=float, default=.002)
    p.add_argument('--adwin-clock', type=int, default=32)
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--progress-every', type=int, default=200)
    return p


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = parser(); a = p.parse_args()
    if not 0 <= a.budget_fraction <= 1 or not 0 < a.adwin_delta < 1 or a.adwin_clock < 1:
        p.error('Invalid budget or ADWIN parameters.')
    if a.progress_every < 0 or any(not values or min(values) < 0 or len(set(values)) != len(values)
                                 for values in (a.train_levels, a.levels)):
        p.error('Invalid progress interval or levels.')
    if set(a.train_levels) & set(a.levels):
        p.error('Test levels must be unseen training intensities.')
    # Optional dependencies are imported only after --help/argument validation.
    import joblib
    import numpy as np
    import torch
    import sklearn
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import amcal_burst_structured_bank_protocol as shared
    from baselines.ral_cd import RALCD
    shared.core.device = torch.device('cpu')
    root = Path(__file__).resolve().parent
    a.burst_dir = (a.burst_dir or root / f'ANT_Burst_{a.dataset}_TEST' /
                   'Dataset_official_v1' / f'AdvBurst_FTSC-{a.dataset}').resolve()
    a.output = a.output.resolve(); a.source_run = a.source_run.resolve()
    random.seed(a.seed); np.random.seed(a.seed)
    source_path = a.source_run / 'Models/protocol_preprocessing.joblib'
    if a.action in ('All', 'Train'):
        if a.output.exists():
            raise ValueError('Output already exists; use a fresh --output or --action Test.')
        artifact = joblib.load(source_path)
    else:
        artifact = joblib.load(a.output / 'Models/protocol_preprocessing.joblib')
    parts = [np.asarray(artifact[k], dtype=np.int64) for k in ('train_ids', 'val_ids', 'test_ids')]
    ids = np.concatenate(parts)
    if any(not len(v) for v in parts) or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= artifact['source_rows']).any():
        raise ValueError('Invalid or overlapping source partitions.')
    if sha(a.burst_dir / 'AdvBurstTrainedBurst_size_0_Step_2000.csv') != artifact['source_sha256']:
        raise ValueError('Clean source provenance mismatch.')
    classes = list(range(len(artifact['label_encoder'].classes_)))

    def domain(level, split):
        x, y, row_ids = shared.read_domain(a.burst_dir, artifact, level, split)
        return x.squeeze(1).cpu().numpy(), np.asarray(y), np.asarray(row_ids)

    def scores(y, pred):
        return dict(accuracy=float(accuracy_score(y, pred)),
            precision=float(precision_score(y, pred, labels=classes, average='macro', zero_division=0)),
            recall=float(recall_score(y, pred, labels=classes, average='macro', zero_division=0)),
            macro_f1=float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
            confusion_matrix=confusion_matrix(y, pred, labels=classes).tolist(), samples=len(y))

    config = dict(seed=a.seed, train_levels=a.train_levels, budget_fraction=a.budget_fraction,
        adwin_backend=a.adwin_backend, adwin_delta=a.adwin_delta, adwin_clock=a.adwin_clock,
        source_preprocessing_sha256=sha(source_path) if source_path.exists() else None,
        training_domain_hashes={str(level): sha(a.burst_dir / f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv')
                                for level in a.train_levels},
        algorithm='RAL-CD section V-E, Wassermann et al. TNSM 2021',
        initial_training='matched source TRAIN; benchmark adaptation, not paper chronological bootstrap',
        reward_signal='raw +1/-1, committee-triggered acquired labels only',
        query='RAL original prefix ratio gate plus strict per-level global cap',
        drift_policy='drop initial memory and old feedback; retain ADWIN reward interval and epsilon labels inside it',
        defaults='ADWIN delta/clock are reconstruction choices; River ADWIN2 is an alternative to paper scikit-multiflow',
        sklearn_version=sklearn.__version__, classes=[str(c) for c in artifact['label_encoder'].classes_],
        feature_names=list(artifact['feature_names']))
    models = a.output / 'Models'; checkpoint = models / 'ral_cd.joblib'
    if a.action in ('All', 'Train'):
        # Check detector availability before creating output or doing model fits.
        model = RALCD([KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(n_estimators=10, random_state=a.seed, n_jobs=a.jobs),
            DecisionTreeClassifier(random_state=a.seed)], .9, .025, .01,
            reward=1, penalty=-1, budget=a.budget_fraction, model_update_frequency=1,
            adwin_backend=a.adwin_backend, adwin_delta=a.adwin_delta, adwin_clock=a.adwin_clock)
        data = [domain(level, 'train_ids') for level in a.train_levels]
        x = np.concatenate([v[0] for v in data]); y = np.concatenate([v[1] for v in data])
        model.fit(x, y)
        models.mkdir(parents=True)
        shutil.copy2(source_path, models / 'protocol_preprocessing.joblib')
        config['source_rows'] = len(y)
        config['adwin_library_version'] = model.monitor.library_version
        joblib.dump(dict(model=model, config=config), checkpoint)
        write_json(a.output / 'training_manifest.json', config)
        print(f'RAL-CD source trained: {len(y)} rows, levels {a.train_levels}', flush=True)
    if a.action == 'Train': return
    saved = joblib.load(checkpoint)
    for key in ('seed', 'train_levels', 'budget_fraction', 'adwin_backend', 'adwin_delta', 'adwin_clock', 'training_domain_hashes'):
        if saved['config'][key] != config[key]:
            raise ValueError(f'Checkpoint/config mismatch: {key}. Reuse original arguments.')
    if sha(models / 'protocol_preprocessing.joblib') != saved['config']['source_preprocessing_sha256']:
        raise ValueError('Saved preprocessing artifact changed.')
    results = a.output / 'Results'; results.mkdir(exist_ok=True)
    rx, ry, _ = domain(0, 'val_ids')
    summaries = [json.loads(path.read_text(encoding='utf-8'))
                 for path in sorted(results.glob('level_*/metrics.json'))]
    for level in a.levels:
        target = results / f'level_{level}'
        if target.exists(): raise ValueError(f'Results already exist: {target}; use a new output.')
        model = deepcopy(saved['model'])
        random.seed(a.seed); np.random.seed(a.seed)
        x, y, row_ids = domain(level, 'test_ids')
        order = np.random.default_rng(a.seed).permutation(len(y))
        x, y, row_ids = x[order], y[order], row_ids[order]
        n = len(y); cap = int(np.floor(a.budget_fraction * n)); used = 0
        initial = model._committee_classifier.predict(x)
        ref_initial = model._committee_classifier.predict(rx)
        predictions = []; traces = []; queried = np.zeros(n, dtype=bool)
        pred_s = query_s = update_s = 0.
        start = time.perf_counter()
        for pos, (row, truth) in enumerate(zip(x, y)):
            tick = time.perf_counter()
            prediction = int(model._committee_classifier.predict(row.reshape(1, -1))[0])
            pred_s += time.perf_counter() - tick
            tick = time.perf_counter()
            committee, decisions = model.ask_committee(row)
            request = model.should_label(committee)
            paid = bool(request and used < cap)
            query_s += time.perf_counter() - tick
            reward = None; drift = False
            if paid:
                tick = time.perf_counter()
                model.acquire_label(row, int(truth), committee, decisions)
                update_s += time.perf_counter() - tick
                used += 1; queried[pos] = True
                reward = model.last_reward; drift = model.last_drift
            predictions.append(prediction)
            traces.append(dict(step=pos+1, source_row=int(row_ids[pos]), truth=int(truth),
                prediction=prediction, queried=paid, labels_used=used, committee=committee,
                reward=reward, drift=drift, reward_window=int(model.monitor.detector.width),
                training_rows=len(model._training_data_samples.y)))
            if a.progress_every and (pos+1) % a.progress_every == 0:
                print(f'Level {level}: {pos+1}/{n}, labels {used}/{cap}, drifts {len(model.drifts)}', flush=True)
        online_s = time.perf_counter() - start
        final = model._committee_classifier.predict(x)
        ref_final = model._committee_classifier.predict(rx)
        metrics = dict(method='RAL-CD', level=level, seed=a.seed, online=scores(y, predictions),
            labels=used, label_cap=cap, initial=scores(y, initial), final=scores(y, final),
            reference_initial=scores(ry, ref_initial), reference_final=scores(ry, ref_final),
            gain_pp=100*(accuracy_score(y, final)-accuracy_score(y, initial)),
            forgetting_pp=100*(accuracy_score(ry, ref_initial)-accuracy_score(ry, ref_final)),
            unqueried_initial=scores(y[~queried], initial[~queried]) if (~queried).any() else None,
            unqueried_final=scores(y[~queried], final[~queried]) if (~queried).any() else None,
            timing=dict(online_seconds=online_s, prediction_seconds=pred_s, query_seconds=query_s,
                update_seconds=update_s, ms_per_row=1000*online_s/n,
                scope='CPU wall time; includes prediction, query, ADWIN and all refits; excludes initialization, observers and disk I/O'),
            adaptation=model.statistics(), protocol='predict-score-query-update; independent initial checkpoint per level',
            order='seeded global permutation; same convention as existing Burst baseline runner',
            test_domain_sha256=sha(a.burst_dir / f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'),
            checkpoint_sha256=sha(checkpoint))
        target.mkdir()
        write_json(target / 'metrics.json', metrics)
        with (target / 'trace.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(traces[0])); writer.writeheader(); writer.writerows(traces)
        summaries.append(metrics)
        summaries.sort(key=lambda item: item['level'])
        write_json(results / 'summary.json', summaries)
        print(f'Level {level}: online {100*metrics["online"]["accuracy"]:.2f}% | labels {used}/{cap} | drifts {len(model.drifts)}', flush=True)
    print('Saved:', results)


if __name__ == '__main__':
    main()
