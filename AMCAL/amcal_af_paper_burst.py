"""AF author-structure few-shot calibration on Burst; NOT an online AL baseline.

Only Burst representation/partitions, deterministic seed and PyTorch backend are
adaptations. See RAL_AF_FIDELITY_AUDIT.md before comparing these batch scores
with AMCAL prequential scores. No target TEST features or labels enter fitting.
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
import amcal_burst_structured_bank_protocol as shared
from baselines.af_author_protocol import (AFParameters, AUTHOR_COMMIT, PROTOCOL_VERSION,
                                         per_class_indices, fit, embed, target_knn)


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def arrays(domain):
    return domain[0].squeeze(1).cpu().numpy(), np.asarray(domain[1]), np.asarray(domain[2])


def prepare(a):
    path = a.source_run/'Models/protocol_preprocessing.joblib'
    artifact = joblib.load(path)
    partitions = [np.asarray(artifact[k], dtype=np.int64) for k in ('train_ids','val_ids','test_ids')]
    ids = np.concatenate(partitions)
    if any(not len(p) for p in partitions) or len(np.unique(ids)) != len(ids):
        raise ValueError('Source train/validation/test partitions must be nonempty and disjoint.')
    if (ids < 0).any() or (ids >= artifact['source_rows']).any():
        raise ValueError('Source row IDs out of bounds.')
    clean_file = a.burst_dir/'AdvBurstTrainedBurst_size_0_Step_2000.csv'
    if shared.core.file_sha256(clean_file) != artifact['source_sha256']:
        raise ValueError('Clean source hash differs from preprocessing artifact.')
    classes = np.arange(len(artifact['label_encoder'].classes_))
    sx, sy, source_ids = arrays(shared.read_domain(a.burst_dir, artifact, 0, 'train_ids'))
    selected = per_class_indices(sy, classes, 25, np.random.default_rng(a.seed))
    source = sx[selected], sy[selected], source_ids[selected]
    tasks = []
    for level in a.levels:
        tx, ty, target_ids = arrays(shared.read_domain(a.burst_dir, artifact, level, 'train_ids'))
        keep = ~np.isin(target_ids, source[2])
        tx, ty, target_ids = tx[keep], ty[keep], target_ids[keep]
        pool = per_class_indices(ty, classes, 20, np.random.default_rng(a.seed+level*1009))
        qx, qy, qids = arrays(shared.read_domain(a.burst_dir, artifact, level, 'test_ids'))
        if np.intersect1d(target_ids[pool], qids).size or np.intersect1d(source[2], qids).size:
            raise ValueError('AF calibration/source IDs overlap evaluation IDs.')
        if a.test_per_class:
            qi = per_class_indices(qy, classes, a.test_per_class, np.random.default_rng(a.seed+level*1013))
            qx, qy, qids = qx[qi], qy[qi], qids[qi]
        tasks.append((level, (tx[pool], ty[pool], target_ids[pool]), (qx, qy, qids)))
    return artifact, classes, source, tasks


def run(a):
    if a.output.exists():
        raise ValueError('Use a fresh --output directory; previous evaluations are preserved.')
    artifact, classes, source, tasks = prepare(a)  # Validate ALL levels before starting work.
    endpoint_audit=getattr(a,'endpoint_metrics',False)
    if endpoint_audit:
        from baselines.endpoint_metrics import paired
        reference=arrays(shared.read_domain(a.burst_dir,artifact,0,'val_ids'))
    p = AFParameters(iterations=a.iterations, tuning_epochs=a.tuning_epochs)
    config = dict(method='AF-author-structure-Burst-PyTorch-port', protocol_version=PROTOCOL_VERSION,
                  author_commit=AUTHOR_COMMIT, seed=a.seed, shots=a.shots, levels=a.levels,
                  parameters=asdict(p), test_per_class=a.test_per_class,
                  preprocessing_sha256=shared.core.file_sha256(a.source_run/'Models/protocol_preprocessing.joblib'),
                  domain_sha256={str(l):shared.core.file_sha256(a.burst_dir/f'AdvBurstTrainedBurst_size_{l}_Step_2000.csv')
                                 for l in [0]+a.levels},
                  source_ids=source[2].tolist(),
                  adaptations=['continuous Burst IAT/PS rather than direction-only WF traces',
                               'AMCAL train-fitted preprocessing and source partitions',
                               'target calibration from remaining TRAIN IDs at each target intensity',
                               'exclude identical original source flow IDs across domains',
                               'PyTorch port; no numerical equivalence with original Keras established',
                               'fixed seed; smaller domain minibatch when fewer than 64 support examples',
                               'single repetition by request; original experiment uses ten repetitions',
                               'all heldout TEST rows instead of 70/class' if not a.test_per_class else
                               f'{a.test_per_class} heldout TEST rows/class'],
                  comparison='offline target-calibrated few-shot accuracy, NOT online prequential accuracy; no stream label cap')
    if endpoint_audit:
        config['endpoint_metrics']=dict(version=1,phase='after_DANN_before_feature_reset_and_supervised_target_tuning_to_final',
            interpretation='supervised tuning stage audit; NOT whole-method forgetting from a source-only checkpoint',
            prediction='same target-only cosine kNN rule and same labelled target supports at both endpoints',
            reference='original-domain VAL source IDs; observer only')
    a.output.mkdir(parents=True)
    json_file(a.output/'execution.json', config)
    rows = []
    for level, pool, test in tasks:
        for shots in a.shots:
            root = a.output/f'level{level}_shot{shots}'
            root.mkdir()
            # Same pool across N; nested supports, selected without TEST labels.
            si = per_class_indices(pool[1], classes, shots, np.random.default_rng(a.seed+level*2003))
            tx, ty, tids = pool[0][si], pool[1][si], pool[2][si]
            qx, qy, qids = test
            if a.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            endpoint_before={}
            observer_seconds=[0.]
            def before_tuning(model):
                tick=time.perf_counter()
                support=target_knn(embed(model.extractor,tx,a.device),ty,shots)
                endpoint_before['target']=support.predict(embed(model.extractor,qx,a.device))
                endpoint_before['reference']=support.predict(embed(model.extractor,reference[0],a.device))
                if a.device=='cuda':torch.cuda.synchronize()
                observer_seconds[0]+=time.perf_counter()-tick
            model, training = fit(source[0], source[1], tx, ty, classes, a.seed, p, a.device,
                                  before_tuning=before_tuning if endpoint_audit else None)
            if a.device == 'cuda':
                torch.cuda.synchronize()
            training_seconds = time.perf_counter()-start-observer_seconds[0]
            start = time.perf_counter()
            knn = target_knn(embed(model.extractor, tx, a.device), ty, shots)
            predicted = knn.predict(embed(model.extractor, qx, a.device))
            if a.device == 'cuda':
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter()-start
            metrics = dict(level=level, shots=shots, accuracy=float(accuracy_score(qy, predicted)),
                           macro_f1=float(f1_score(qy, predicted, average='macro', zero_division=0)),
                           samples=len(qy), source_labels=len(source[1]), target_calibration_labels=len(ty),
                           online_labels=0, support_origin='target calibration ONLY', k=shots,
                           weights='distance', metric='cosine', training_seconds=training_seconds,
                           inference_seconds=inference_seconds, source_ids=source[2].tolist(),
                           target_calibration_ids=tids.tolist(), test_ids=qids.tolist(),
                           training=training, execution=config)
            if endpoint_audit:
                final_reference=knn.predict(embed(model.extractor,reference[0],a.device))
                target_pair=paired(endpoint_before['target'],predicted,qy,np.ones(len(qy),dtype=bool),classes)
                reference_pair=paired(endpoint_before['reference'],final_reference,reference[1],np.ones(len(reference[1]),dtype=bool),classes)
                metrics['endpoint_metrics_version']=1
                metrics['endpoint_phase']=config['endpoint_metrics']
                metrics['same_level']={'all_rows':{'metric_pairs':target_pair},'unqueried':{'metric_pairs':target_pair}}
                metrics['reference']={'metric_pairs':reference_pair}
                shared.base.write_trace(root/'endpoint_predictions.csv',
                    [dict(domain=domain,source_row=int(i),truth=int(y),initial_prediction=int(b),final_prediction=int(f))
                     for domain,ids,truth,before,after in [('target',qids,qy,endpoint_before['target'],predicted),
                                                         ('reference0',reference[2],reference[1],endpoint_before['reference'],final_reference)]
                     for i,y,b,f in zip(ids,truth,before,after)])
            torch.save(dict(state=model.state_dict(),classes=len(classes),parameters=asdict(p)), root/'af.pth')
            json_file(root/'metrics.json', metrics)
            shared.base.write_trace(root/'predictions.csv', [dict(source_row=int(i),truth=int(y),prediction=int(z))
                                                            for i,y,z in zip(qids,qy,predicted)])
            rows.append(dict(Level=level,Shots=shots,Accuracy=100*metrics['accuracy'],MacroF1=100*metrics['macro_f1'],
                             SourceLabels=len(source[1]),CalibrationLabels=len(ty),OnlineLabels=0,Samples=len(qy)))
            shared.base.write_trace(a.output/'combined_af_fewshot.csv', rows)
            print(f'AF | level {level} | N={shots} | accuracy {100*metrics["accuracy"]:.2f}% | '
                  f'target calibration labels {len(ty)} | test rows {len(qy)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--burst-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--levels', type=int, nargs='+', default=[5,7,10,12,15,17,20])
    parser.add_argument('--shots', type=int, nargs='+', choices=[1,5,10,15,20], default=[5])
    parser.add_argument('--endpoint-metrics',action='store_true',help='Observer-only audit of the supervised tuning stage; not whole-AF forgetting.')
    parser.add_argument('--test-per-class', type=int, default=0, help='0: common heldout TEST set; 70: author count.')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--tuning-epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    a = parser.parse_args()
    if min(a.iterations,a.tuning_epochs) < 1 or a.test_per_class < 0:
        parser.error('Invalid iteration/epoch/test count.')
    if len(set(a.levels)) != len(a.levels) or min(a.levels) <= 0 or len(set(a.shots)) != len(a.shots):
        parser.error('Duplicate levels/shots or nonpositive target levels.')
    if a.device == 'auto':
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable.')
    run(a)


if __name__ == '__main__':
    main()

