"""Audit/refine the existing Burst CNN without running Context or RL training.

The original architecture, source partitions, scaler and training intensities
are retained. Checkpoint selection uses validation rows only. Heldout source
rows/intensities are scored only by the explicit evaluate command.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_partitions(artifact):
    partitions = {}
    for key in ('train_ids', 'val_ids', 'test_ids'):
        raw = np.asarray(artifact[key])
        if raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError('Invalid source partition: ' + key)
        if not np.equal(raw, np.floor(raw)).all():
            raise ValueError('Non-integer source IDs: ' + key)
        partitions[key] = raw.astype(np.int64)
    ids = np.concatenate(list(partitions.values()))
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= artifact['source_rows']).any():
        raise ValueError('Source partitions must be unique, disjoint and within source bounds.')
    return partitions


def validate_levels(requested, seen, evaluating=False):
    seen = list(seen)
    levels = list(requested) if requested is not None else sorted(set(seen))
    if (not levels or len(set(levels)) != len(levels)
            or any(not np.isfinite(v) or v < 0 or int(v) != v for v in levels + seen)):
        raise ValueError('Levels must be distinct nonnegative integers.')
    if evaluating:
        if set(levels) & set(seen):
            raise ValueError('Heldout evaluation levels must exclude every intensity seen by the source CNN.')
    elif set(levels) != set(seen):
        raise ValueError('Use the same training levels as the source CNN; a warm start cannot erase previously seen levels.')
    return sorted(int(v) for v in levels)


def classification_metrics(logits, labels):
    """Natural-distribution, unsmoothed metrics, also for TRAIN in eval mode."""
    z = np.asarray(logits, dtype=np.float64)
    raw = np.asarray(labels)
    if (z.ndim != 2 or raw.ndim != 1 or len(z) != len(raw) or not len(raw)
            or not np.isfinite(z).all() or not np.isfinite(raw).all()
            or not np.equal(raw, np.floor(raw)).all()):
        raise ValueError('Expected finite logits and matching nonempty integer labels.')
    y = raw.astype(np.int64)
    if z.shape[1] < 2 or (y < 0).any() or (y >= z.shape[1]).any():
        raise ValueError('Labels fall outside model classes.')
    prediction = z.argmax(1)
    confusion = np.zeros((z.shape[1], z.shape[1]), dtype=np.int64)
    np.add.at(confusion, (y, prediction), 1)
    support = confusion.sum(1)
    hits = confusion.diagonal()
    recall = np.divide(hits, support, out=np.zeros(z.shape[1]), where=support > 0)
    denom = support + confusion.sum(0)
    f1 = np.divide(2 * hits, denom, out=np.zeros(z.shape[1]), where=denom > 0)
    shifted = z - z.max(1, keepdims=True)
    ce = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]
    return dict(n=len(y), accuracy=float((prediction == y).mean()),
                balanced_accuracy=float(recall[support > 0].mean()),
                macro_f1=float(f1.mean()), ce=float(ce.mean()))


def validation_score(rows, selection):
    losses = np.asarray([r['ce'] for r in rows], dtype=np.float64)
    if not len(losses) or not np.isfinite(losses).all():
        raise ValueError('Validation losses must be finite and nonempty.')
    if selection == 'worst-loss':
        return float(losses.max()), float(losses.mean())
    if selection == 'mean-loss':
        return float(losses.mean()), float(losses.max())
    raise ValueError('Unknown checkpoint selection rule.')


def batch_slices(count, size=64):
    """Visit each TRAIN row once, merging a singleton tail for BatchNorm."""
    if count < 2 or size < 2:
        raise ValueError('BatchNorm training needs at least two rows and batch size two.')
    result = [(start, min(start + size, count)) for start in range(0, count, size)]
    if len(result) > 1 and result[-1][1] - result[-1][0] == 1:
        result[-2] = (result[-2][0], count)
        result.pop()
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot save an empty report.')
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runtime(device_name, seed):
    import torch
    import amcal_burst_official_protocol as official
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device_name == 'auto' else device_name)
    official.core.device = device
    official.core.set_seed(seed)
    return torch, official, device


def load_source(torch, official, run, root, device):
    import joblib
    models = run / 'Models'
    artifact_path = models / 'protocol_preprocessing.joblib'
    artifact = joblib.load(artifact_path)
    validate_partitions(artifact)
    if sha256(root / 'AdvBurstTrainedBurst_size_0_Step_2000.csv') != artifact['source_sha256']:
        raise ValueError('Burst clean source differs from checkpoint provenance.')
    checkpoint = torch.load(models / 'initial_cnn_model.pth', map_location='cpu', weights_only=False)
    if checkpoint.get('preprocessing_sha256', sha256(artifact_path)) != sha256(artifact_path):
        raise ValueError('CNN/preprocessing mismatch.')
    seen = checkpoint.get('training_levels')
    if seen is None and (models / 'structured_bank_ddqn.pth').exists():
        saved = torch.load(models / 'structured_bank_ddqn.pth', map_location='cpu', weights_only=False)
        for name in ('initial_cnn_model.pth', 'protocol_preprocessing.joblib'):
            if saved.get('model_hashes', {}).get(name, sha256(models / name)) != sha256(models / name):
                raise ValueError('Source selector/model provenance mismatch: ' + name)
        seen = saved['training_levels']
    if seen is None and (run / 'two_stage_training.json').exists():
        seen = json.loads((run / 'two_stage_training.json').read_text())['cnn_refinement_domains']
    if seen is None and (models / 'adversarial_best_dqn.pth').exists():
        saved = torch.load(models / 'adversarial_best_dqn.pth', map_location='cpu', weights_only=False)
        seen = saved.get('training_perturbations')
    if seen is None:
        raise ValueError('CNN training-level provenance is missing; use the complete source run.')
    width, classes = len(artifact['feature_names']), len(artifact['label_encoder'].classes_)
    model = official.core.CNN1D(width, classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, artifact, validate_levels(None, seen)


def load_domains(torch, official, root, artifact, levels, splits):
    """Only the requested source partitions become tensors or metric inputs."""
    domains = {}
    for level in levels:
        path = root / f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
        frame, source_ids, _ = official.mapped_domain(path, artifact, root, False)
        lookup = {int(source_id): i for i, source_id in enumerate(source_ids)}
        for split in splits:
            wanted = np.asarray(artifact[split], dtype=np.int64)
            if not set(wanted) <= set(lookup):
                raise ValueError(f'Missing source IDs in level {level} / {split}.')
            selected = frame.iloc[[lookup[int(source_id)] for source_id in wanted]]
            features = selected.iloc[:, :-1]
            scaler = artifact['scaler']
            scaled = scaler.transform(features if hasattr(scaler, 'feature_names_in_')
                                      else features.to_numpy()).astype(np.float32)
            if not np.isfinite(scaled).all():
                raise ValueError('Nonfinite scaled domain features.')
            labels = artifact['label_encoder'].transform(selected.iloc[:, -1])
            domains[level, split] = (torch.from_numpy(scaled).unsqueeze(1),
                                     torch.as_tensor(labels, dtype=torch.long))
    return domains


def measure(torch, model, data, device):
    inputs, labels = data
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(labels), 128):
            logits.append(model(inputs[start:start + 128].to(device)).cpu().numpy())
    return classification_metrics(np.concatenate(logits), labels.numpy())


def domain_report(torch, model, domains, levels, device, epoch, stage):
    rows = []
    for level in levels:
        for split in ('train_ids', 'val_ids'):
            metric = measure(torch, model, domains[level, split], device)
            rows.append(dict(stage=stage, epoch=epoch, level=level,
                             split='train' if split == 'train_ids' else 'validation', **metric))
    print(' PER  Train%    Val%   Gap(pp)  TrainCE    ValCE', flush=True)
    for level in levels:
        train, val = [r for r in rows if r['level'] == level]
        print(f'{level:4d} {100*train["accuracy"]:7.2f} {100*val["accuracy"]:7.2f}'
              f' {100*(train["accuracy"]-val["accuracy"]):9.2f}'
              f' {train["ce"]:8.4f} {val["ce"]:8.4f}', flush=True)
    return rows


def train(a, torch, official, device, model, artifact, levels):
    if a.output.exists():
        raise ValueError('Use a fresh --output directory; existing runs are preserved.')
    domains = load_domains(torch, official, a.burst_dir, artifact, levels, ('train_ids', 'val_ids'))
    # The same source IDs appear once per allowed domain, just as in multishift refinement.
    inputs = torch.cat([domains[level, 'train_ids'][0] for level in levels])
    labels = torch.cat([domains[level, 'train_ids'][1] for level in levels])
    batches = batch_slices(len(labels))
    counts = torch.bincount(labels, minlength=len(artifact['label_encoder'].classes_)).float()
    if (counts == 0).any():
        raise ValueError('Every model class needs TRAIN support.')
    criterion = torch.nn.CrossEntropyLoss(weight=(1 / counts).to(device),
                                         label_smoothing=a.label_smoothing)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs,
                                                          eta_min=min(1e-6, a.lr / 100))
    models = a.output / 'Models'
    models.mkdir(parents=True)
    for name in ('protocol_preprocessing.joblib', 'protocol_manifest.json', 'official_reference_validation.csv'):
        source = a.source_run / 'Models' / name
        if source.exists():
            shutil.copy2(source, models / name)
    manifest = dict(protocol='Frozen-only refinement; original CNN architecture; no Context/RL training',
                    source_run=str(a.source_run), source_cnn_sha256=sha256(a.source_run / 'Models/initial_cnn_model.pth'),
                    preprocessing_sha256=sha256(models / 'protocol_preprocessing.joblib'),
                    training_levels=levels, training_domain_sha256={str(v): sha256(a.burst_dir / f'AdvBurstTrainedBurst_size_{v}_Step_2000.csv') for v in levels},
                    seed=a.seed, optimizer='AdamW', lr=a.lr, weight_decay=a.weight_decay,
                    label_smoothing=a.label_smoothing, selection=a.selection,
                    max_epochs=a.epochs, patience=a.patience, baseline_is_candidate=True,
                    heldout_scored_during_training=False, status='running', epochs_completed=0)
    manifest_path = a.output / 'frozen_training.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Baseline CNN; natural TRAIN/validation metrics in evaluation mode.', flush=True)
    history = domain_report(torch, model, domains, levels, device, 0, 'baseline')
    best_score = validation_score([r for r in history if r['split'] == 'validation'], a.selection)
    best_state, best_epoch, stale = deepcopy(model.state_dict()), 0, 0

    def save_best():
        torch.save(dict(protocol_version=3, preprocessing_sha256=manifest['preprocessing_sha256'],
                        model_state_dict={k: v.detach().cpu().clone() for k, v in best_state.items()},
                        input_length=len(artifact['feature_names']), num_classes=len(counts),
                        training_levels=levels, frozen_refinement=manifest,
                        best_epoch=best_epoch, validation_score=best_score), models / 'initial_cnn_model.pth')

    save_best()
    write_csv(a.output / 'Results/frozen_training_history.csv', history)
    for epoch in range(1, a.epochs + 1):
        model.train()
        running, seen = 0., 0
        permutation = torch.randperm(len(labels))
        for start, stop in batches:
            indices = permutation[start:stop]
            batch, target = inputs[indices], labels[indices]
            optimizer.zero_grad()
            loss = criterion(model(batch.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CNN training loss; best prior checkpoint is preserved.')
            loss.backward()
            optimizer.step()
            running += loss.item() * len(target)
            seen += len(target)
        print(f'\nCNN epoch {epoch}/{a.epochs} | objective {running/seen:.4f}'
              f' | LR {optimizer.param_groups[0]["lr"]:.6g}', flush=True)
        rows = domain_report(torch, model, domains, levels, device, epoch, 'epoch')
        history.extend(rows)
        score = validation_score([r for r in rows if r['split'] == 'validation'], a.selection)
        if score < best_score:
            best_score, best_state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch, 0
            save_best()
            print(f'  Saved best CNN: {a.selection} {score[0]:.4f}', flush=True)
        else:
            stale += 1
        scheduler.step()
        write_csv(a.output / 'Results/frozen_training_history.csv', history)
        manifest.update(epochs_completed=epoch, best_epoch=best_epoch, validation_score=best_score)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if stale >= a.patience:
            print(f'Early stopping after {epoch} epochs.', flush=True)
            break
    model.load_state_dict(best_state)
    final = domain_report(torch, model, domains, levels, device, best_epoch, 'selected')
    history.extend(final)
    write_csv(a.output / 'Results/frozen_training_history.csv', history)
    manifest.update(status='complete', best_epoch=best_epoch, validation_score=best_score)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    save_best()
    print(f'\nSelected epoch {best_epoch}; epoch 0 means the source CNN was retained.', flush=True)
    print(f'Saved CNN and report: {a.output}', flush=True)
    print('No heldout rows scored. Context and RL must be retrained against a changed CNN.', flush=True)


def evaluate(a, torch, official, device, model, artifact, levels):
    manifest_path = a.output / 'frozen_training.json'
    if not manifest_path.exists():
        raise ValueError('Expected the Frozen-only output run with frozen_training.json.')
    manifest = json.loads(manifest_path.read_text())
    source_run = Path(manifest['source_run'])
    if sha256(source_run / 'Models/initial_cnn_model.pth') != manifest['source_cnn_sha256']:
        raise ValueError('Comparison source CNN has changed.')
    original, _, _ = load_source(torch, official, source_run, a.burst_dir, device)
    if sha256(source_run / 'Models/protocol_preprocessing.joblib') != sha256(a.output / 'Models/protocol_preprocessing.joblib'):
        raise ValueError('Comparison requires identical preprocessing and source partitions.')
    domains = load_domains(torch, official, a.burst_dir, artifact, levels, ('test_ids',))
    rows = []
    print('Heldout CNN comparison (%; Gain in percentage points; no online updates).', flush=True)
    print(' PER     N  Source%  Refined%  Gain(pp)  SourceCE  RefinedCE', flush=True)
    for level in levels:
        old = measure(torch, original, domains[level, 'test_ids'], device)
        new = measure(torch, model, domains[level, 'test_ids'], device)
        row = dict(level=level, n=new['n'], source_accuracy=old['accuracy'], refined_accuracy=new['accuracy'],
                   gain_pp=100*(new['accuracy']-old['accuracy']), source_macro_f1=old['macro_f1'],
                   refined_macro_f1=new['macro_f1'], source_balanced_accuracy=old['balanced_accuracy'],
                   refined_balanced_accuracy=new['balanced_accuracy'], source_ce=old['ce'], refined_ce=new['ce'])
        rows.append(row)
        print(f'{level:4d} {new["n"]:5d} {100*old["accuracy"]:8.2f} {100*new["accuracy"]:9.2f}'
              f' {row["gain_pp"]:9.2f} {old["ce"]:9.4f} {new["ce"]:10.4f}', flush=True)
    print(f'Mean        {100*np.mean([r["source_accuracy"] for r in rows]):8.2f}'
          f' {100*np.mean([r["refined_accuracy"] for r in rows]):9.2f}'
          f' {np.mean([r["gain_pp"] for r in rows]):9.2f}', flush=True)
    path = a.output / 'Results/frozen_heldout_comparison.csv'
    write_csv(path, rows)
    print(f'Saved: {path}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'train', 'evaluate'))
    p.add_argument('--source-run', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--burst-dir', type=Path, required=True)
    p.add_argument('--levels', type=int, nargs='+')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--weight-decay', type=float, default=.001)
    p.add_argument('--label-smoothing', type=float, default=.05)
    p.add_argument('--selection', choices=('worst-loss', 'mean-loss'), default='worst-loss')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    a = p.parse_args()
    if a.command in ('audit', 'train') and a.source_run is None:
        p.error('--source-run is required for audit/train.')
    if a.command in ('train', 'evaluate') and a.output is None:
        p.error('--output is required for train/evaluate.')
    if (min(a.epochs, a.patience) < 1 or not np.isfinite(a.lr) or a.lr <= 0
            or not np.isfinite(a.weight_decay) or a.weight_decay < 0
            or not np.isfinite(a.label_smoothing) or not 0 <= a.label_smoothing < 1):
        p.error('Invalid epochs, patience or CNN optimization settings.')
    a.burst_dir = a.burst_dir.resolve()
    if a.source_run is not None:
        a.source_run = a.source_run.resolve()
    if a.output is not None:
        a.output = a.output.resolve()
    torch, official, device = runtime(a.device, a.seed)
    run = a.output if a.command == 'evaluate' else a.source_run
    model, artifact, seen = load_source(torch, official, run, a.burst_dir, device)
    requested = a.levels
    if requested is None and a.command == 'evaluate':
        requested = [v for v in (0, 1, 3, 5, 7, 10, 12, 15, 17, 20) if v not in seen]
    levels = validate_levels(requested, seen, evaluating=a.command == 'evaluate')
    if a.command == 'train':
        train(a, torch, official, device, model, artifact, levels)
    elif a.command == 'evaluate':
        evaluate(a, torch, official, device, model, artifact, levels)
    else:
        domains = load_domains(torch, official, a.burst_dir, artifact, levels, ('train_ids', 'val_ids'))
        print('Existing CNN audit; TRAIN and validation use eval mode; Gap = Train% - Val%.', flush=True)
        rows = domain_report(torch, model, domains, levels, device, 0, 'audit')
        path = (a.output if a.output is not None else a.source_run) / 'Results/frozen_audit.csv'
        write_csv(path, rows)
        print(f'Saved: {path}', flush=True)


if __name__ == '__main__':
    main()
