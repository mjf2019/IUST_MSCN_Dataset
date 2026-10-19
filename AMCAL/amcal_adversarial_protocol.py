"""The approved balanced-reference, 3-Context/7-action AMCAL on mapped attacks.

Each attack is trained independently on clean + ordinal level 2. The original
two-stage CNN/Context source training and structured policy engine are reused.
Existing unmapped legacy attack CSVs are never treated as new training domains.
"""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from baselines.adversarial_domains import (VERSION, LEVELS, digest, domain_name,
                                          export_domain, generate, strength)
from amcal_burst_frozen_generalization import validate_partitions


@contextmanager
def arguments(values):
    previous = sys.argv
    try:
        sys.argv = [previous[0]] + list(map(str, values))
        yield
    finally:
        sys.argv = previous


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_surrogate(a, torch, core, run):
    models = run / 'Models'
    artifact = joblib.load(models / 'protocol_preprocessing.joblib')
    partitions = validate_partitions(artifact)
    source_path = a.data_dir / a.source_file
    if digest(source_path) != artifact['source_sha256']:
        raise ValueError('Clean CSV differs from the saved clean CNN source.')
    frame = pd.read_csv(source_path)
    if list(frame.columns) != artifact['feature_names'] + [artifact['label_name']]:
        raise ValueError('Source schema differs from preprocessing.')
    groups = np.asarray(artifact['feature_groups'])
    for first, second in [('train_ids', 'val_ids'), ('train_ids', 'test_ids'), ('val_ids', 'test_ids')]:
        if set(groups[partitions[first]]) & set(groups[partitions[second]]):
            raise ValueError('Feature-identical source rows cross partitions.')
    checkpoint = torch.load(models / 'initial_cnn_model.pth', map_location='cpu', weights_only=False)
    if checkpoint.get('preprocessing_sha256') != digest(models / 'protocol_preprocessing.joblib'):
        raise ValueError('Clean CNN/preprocessing hash mismatch.')
    if any(v != 0 for v in checkpoint.get('training_levels', [])):
        raise ValueError('--clean-run must contain a clean-only CNN, not a multishift-refined CNN.')
    previous = run / 'two_stage_training.json'
    if previous.exists() and any(v != 0 for v in json.loads(previous.read_text()).get('cnn_refinement_domains', [])):
        raise ValueError('--clean-run was already refined on shifted domains; use the saved clean source CNN.')
    model = core.CNN1D(len(artifact['feature_names']), len(artifact['label_encoder'].classes_)).to(core.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, artifact, frame, checkpoint


def clean_domain(root, source_path, artifact):
    root.mkdir(parents=True, exist_ok=True)
    target = root / domain_name(0)
    if target.exists() and digest(target) != artifact['source_sha256']:
        raise ValueError('Cached clean source changed.')
    if not target.exists():shutil.copy2(source_path, target)
    pd.DataFrame({'source_row': np.arange(artifact['source_rows'])}).to_csv(
        target.with_name(target.stem + '_rows.csv'), index=False)


def make_attack(a, torch, core, model, artifact, source, root, level, ids, manifest):
    key = str(level)
    record = manifest.setdefault('domains', {}).get(key)
    path = root / domain_name(level)
    mapping = path.with_name(path.stem + '_rows.csv')
    if record is not None:
        if (digest(path) != record['csv_sha256'] or digest(mapping) != record['row_map_sha256']
                or record['rows'] != len(ids)):
            raise ValueError('Cached attack/map differs from its recorded provenance.')
        return
    features = source.iloc[ids, :-1]
    scaler = artifact['scaler']
    scaled = scaler.transform(features if hasattr(scaler, 'feature_names_in_') else features.to_numpy())
    if not np.isfinite(scaled).all():raise ValueError('Nonfinite source features.')
    labels = artifact['label_encoder'].transform(source.iloc[ids, -1])
    settings = manifest['attack_settings']
    radius = strength(a.attack, level)
    perturbed = generate(torch, model, scaled, labels, a.attack, radius, core.device,
                         manifest['generation_seed'] + 1009 * level, **settings)
    record = export_domain(root, level, source, ids, perturbed, artifact)
    displacement = perturbed.astype(np.float64) - scaled
    record.update(level=level, radius=radius, norm='Linf' if a.attack in ('FGSM', 'PGD') else 'L2',
                  max_linf=float(np.abs(displacement).max()),
                  max_l2=float(np.linalg.norm(displacement, axis=1).max()))
    clean_correct = attacked_correct = successes = 0
    with torch.no_grad():
        for start in range(0, len(ids), 128):
            truth = torch.as_tensor(labels[start:start+128], device=core.device)
            clean = torch.tensor(scaled[start:start+128], dtype=torch.float32, device=core.device).unsqueeze(1)
            attacked = torch.tensor(perturbed[start:start+128], dtype=torch.float32, device=core.device).unsqueeze(1)
            before = model(clean).argmax(1) == truth
            after = model(attacked).argmax(1) == truth
            clean_correct += int(before.sum().item())
            attacked_correct += int(after.sum().item())
            successes += int((before & ~after).sum().item())
    record.update(surrogate_clean_accuracy=clean_correct/len(ids),
                  surrogate_attacked_accuracy=attacked_correct/len(ids),
                  success_on_clean_correct=None if not clean_correct else successes/clean_correct)
    manifest['domains'][key] = record
    (root / 'attack_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'  Saved {a.attack} level {level}: surrogate accuracy {100*clean_correct/len(ids):.2f}%'
          f' -> {100*attacked_correct/len(ids):.2f}%; mapped rows {len(ids)}.', flush=True)


def runtime(a):
    import torch
    import amcal_burst_two_stage_protocol as two
    if a.device == 'cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable.')
    two.core.device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if a.device == 'auto' else a.device)
    two.core.set_seed(a.seed)
    return torch, two


def train(a):
    if a.clean_run is None:raise ValueError('Training requires --clean-run with a protocol-v3 clean CNN.')
    if a.output.exists():raise ValueError('Use a fresh --output directory; existing experiments are preserved.')
    torch, two = runtime(a)
    core = two.core
    model, artifact, source, checkpoint = load_surrogate(a, torch, core, a.clean_run)
    a.output.mkdir(parents=True)
    surrogate = a.output / 'Surrogate/Models'
    surrogate.mkdir(parents=True)
    for name in ('initial_cnn_model.pth', 'protocol_preprocessing.joblib'):
        shutil.copy2(a.clean_run / 'Models' / name, surrogate / name)
    roots = a.output / 'TrainingDomains'
    clean_domain(roots, a.data_dir / a.source_file, artifact)
    partitions = {key: np.asarray(artifact[key], dtype=np.int64).tolist()
                  for key in ('train_ids', 'val_ids', 'test_ids')}
    (roots / 'official_partitions.json').write_text(json.dumps(dict(
        train=partitions['train_ids'], validation=partitions['val_ids'], test=partitions['test_ids'])), encoding='utf-8')
    manifest = dict(version=VERSION, attack=a.attack, source_file=a.source_file,
                    source_sha256=artifact['source_sha256'], source_rows=len(source),
                    source_data_dir=str(a.data_dir), clean_run=str(a.clean_run), generation_seed=a.seed,
                    surrogate_sha256=digest(surrogate / 'initial_cnn_model.pth'),
                    preprocessing_sha256=digest(surrogate / 'protocol_preprocessing.joblib'),
                    ordinal_strengths={str(level): strength(a.attack, level) for level in range(6)},
                    training_levels=[0, 2], training_partition='source TRAIN and validation IDs only',
                    attack_settings={key: getattr(a, key) for key in ('pgd_steps', 'cw_steps', 'cw_search', 'deepfool_steps')},
                    training_settings={key: getattr(a, key) for key in ('seed', 'context_lr', 'budget_fraction', 'selection_cost', 'gain_weight', 'retention_weight', 'gamma', 'stage1_epochs', 'cnn_epochs', 'source_epochs', 'bank_epochs')},
                    threat_model='fixed clean source-CNN feature-space attacks; transfer after CNN refinement; not adaptive attacks on the final bank',
                    status='preparing', domains={})
    (a.output / 'adversarial_training.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # No TEST perturbation or TEST metric is constructed in training.
    ids = np.concatenate([artifact['train_ids'], artifact['val_ids']]).astype(np.int64)
    print(f'{a.attack}: train clean + ordinal level 2 (radius {strength(a.attack,2):g}); '
          f'TRAIN={len(artifact["train_ids"])} validation={len(artifact["val_ids"])} TEST={len(artifact["test_ids"])}.', flush=True)
    make_attack(a, torch, core, model, artifact, source, roots, 2, ids, manifest)
    original_train = core.train_initial_cnn
    calls = 0

    def reuse_clean(model_to_train, train_loader, val_loader, counts, epochs, patience):
        nonlocal calls
        calls += 1
        if calls == 1:
            current = joblib.load('Models/protocol_preprocessing.joblib')
            for split in ('train_ids', 'val_ids', 'test_ids'):
                if not np.array_equal(current[split], artifact[split]):raise ValueError('Source split changed during training.')
            for name in ('mean_', 'scale_'):
                if not np.allclose(getattr(current['scaler'], name), getattr(artifact['scaler'], name)):
                    raise ValueError('Source scaling changed during training.')
            if not np.array_equal(current['label_encoder'].classes_, artifact['label_encoder'].classes_):
                raise ValueError('Source class ordering changed during training.')
            model_to_train.load_state_dict(checkpoint['model_state_dict'])
            model_to_train.eval()
            print('Reused the existing clean CNN; clean Context/RL are trained before multishift refinement.', flush=True)
            return model_to_train
        return original_train(model_to_train, train_loader, val_loader, counts, epochs, patience)

    core.train_initial_cnn = reuse_clean
    try:
        with arguments(['train', '--burst-dir', roots, '--burst-mapping-dir', roots,
                        '--official-split-dir', roots, '--output', a.output / 'Source',
                        '--multishift-levels', 0, 2, '--stage1-rl-epochs', a.stage1_epochs,
                        '--cnn-multishift-epochs', a.cnn_epochs, '--cnn-multishift-patience', 10,
                        '--stage2-pretrain-epochs', 20, '--base-epochs', 1, '--base-patience', 1,
                        '--pretrain-epochs', 20, '--context-epochs', a.source_epochs,
                        '--context-patience', 10, '--train-online-lr', .0001,
                        '--budget-fraction', a.budget_fraction, '--selection-cost', a.selection_cost,
                        '--gain-weight', a.gain_weight, '--retention-weight', a.retention_weight,
                        '--routing-reward-mode', 'absolute', '--seed', a.seed, '--device', a.device]):
            two.main()
    finally:
        core.train_initial_cnn = original_train
    import amcal_burst_structured_bank_protocol as bank
    with arguments(['train', '--source-run', a.output / 'Source', '--burst-dir', roots,
                    '--output', a.output / 'Bank', '--levels', 0, 2,
                    '--balanced-reward-reference', '--epochs', a.bank_epochs, '--patience', 5,
                    '--router-warmup-episodes', 3, '--context-lr', a.context_lr,
                    '--budget-fraction', a.budget_fraction, '--selection-cost', a.selection_cost,
                    '--gain-weight', a.gain_weight, '--retention-weight', a.retention_weight,
                    '--gamma', a.gamma, '--query-schedule', 'policy', '--update-policy', 'learned',
                    '--train-epsilon', .2, '--eval-epsilon', 0, '--seed', a.seed, '--device', a.device]):
        bank.main()
    manifest['status'] = 'complete'
    (a.output / 'adversarial_training.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'{a.attack} complete: {a.output}. No TEST source rows were scored.', flush=True)


def summary_rows(attack, metrics):
    rows = []
    for m in metrics:
        level = m['level']
        same = m['same_level']['unqueried']
        rows.append(dict(Attack=attack, Level=level, Strength=strength(attack, level),
                         Domain='Seen' if level in (0, 2) else 'Unseen', Mode=m['mode'], N=m['samples'],
                         Frozen=100*m['frozen_accuracy'], Online=100*m['accuracy'], MacroF1=100*m['macro_f1'],
                         Labels=m['labels'], Cap=m['label_cap'], LabelPct=100*m['actual_label_fraction'],
                         CapUsedPct=m['budget_utilization_pct'], Oracle=100*m['oracle_all'],
                         RefDrop=m['reference']['drop_pp'], SameUnqueriedN=same['samples'],
                         SameUnqueriedFinal=None if same['final'] is None else 100*same['final'],
                         SameUnqueriedDrop=same['drop_pp'], OnlineSeconds=m['online_seconds'],
                         MsPerRow=1000*m['online_seconds']/m['samples']))
    return rows


def display(rows):
    print('\nAttack/domain summary (%; Gap/Drop in percentage points):', flush=True)
    print(' Attack   Level Radius Domain  Frozen     RL    Rnd    Gap LabRL LabRnd DropRL DropRnd', flush=True)
    for attack in dict.fromkeys(r['Attack'] for r in rows):
        own = [r for r in rows if r['Attack'] == attack]
        keys = list(dict.fromkeys((r['Level'], r['Strength'], r['Domain']) for r in own))
        for level, radius, domain in keys:
            pair = {r['Mode']: r for r in own if r['Level'] == level}
            if not {'learned', 'random'} <= pair.keys():continue
            rl, rnd = pair['learned'], pair['random']
            print(f'{attack:>9} {str(level):>5} {float(radius):6g} {domain:>6} {float(rl["Frozen"]):7.2f}'
                  f' {float(rl["Online"]):6.2f} {float(rnd["Online"]):6.2f}'
                  f' {float(rl["Online"])-float(rnd["Online"]):6.2f}'
                  f' {float(rl["Labels"]):5.0f} {float(rnd["Labels"]):6.0f}'
                  f' {float(rl["RefDrop"]):6.2f} {float(rnd["RefDrop"]):7.2f}', flush=True)
        unseen = {mode: [r for r in own if r['Mode'] == mode and r['Domain'] == 'Unseen']
                  for mode in ('learned', 'random')}
        if all(unseen.values()):
            mean = lambda mode, field: float(np.mean([float(r[field]) for r in unseen[mode]]))
            print(f'{attack:>9}  Mean      - Unseen {mean("learned","Frozen"):7.2f}'
                  f' {mean("learned","Online"):6.2f} {mean("random","Online"):6.2f}'
                  f' {mean("learned","Online")-mean("random","Online"):6.2f}'
                  f' {mean("learned","Labels"):5.1f} {mean("random","Labels"):6.1f}'
                  f' {mean("learned","RefDrop"):6.2f} {mean("random","RefDrop"):7.2f}', flush=True)
    print('Seen: heldout source rows at clean/training intensity. Unseen: heldout rows at other intensities.', flush=True)


def prepare_evaluation_domains(a):
    path = a.output / 'adversarial_training.json'
    manifest = json.loads(path.read_text())
    if manifest['version'] != VERSION or manifest['attack'] != a.attack or manifest['status'] != 'complete':
        raise ValueError('A complete matching adversarial training run is required.')
    settings = manifest['training_settings']
    for key in ('context_lr', 'budget_fraction', 'selection_cost', 'gain_weight', 'retention_weight', 'gamma'):
        if getattr(a, key) != settings[key]:raise ValueError('Evaluation differs from training: ' + key)
    torch, two = runtime(a)
    core = two.core
    model, artifact, source, _ = load_surrogate(a, torch, core, a.output / 'Surrogate')
    if digest(a.output / 'Surrogate/Models/initial_cnn_model.pth') != manifest['surrogate_sha256']:
        raise ValueError('Attack surrogate changed.')
    roots = a.output / 'EvaluationDomains'
    clean_domain(roots, a.data_dir / a.source_file, artifact)
    cache = roots / 'attack_manifest.json'
    if cache.exists():
        evaluation_manifest = json.loads(cache.read_text())
        for key in ('version', 'attack', 'surrogate_sha256', 'preprocessing_sha256', 'generation_seed', 'attack_settings'):
            if evaluation_manifest[key] != manifest[key]:raise ValueError('Evaluation attack settings/provenance changed.')
    else:
        evaluation_manifest = {**manifest, 'domains': {}, 'training_partition': 'heldout source TEST IDs only'}
    # Attack data are fixed across evaluation stream seeds.
    ids = np.asarray(artifact['test_ids'], dtype=np.int64)
    for level in a.levels:
        if level:make_attack(a, torch, core, model, artifact, source, roots, level, ids, evaluation_manifest)
    cache.write_text(json.dumps(evaluation_manifest, indent=2))
    return roots


def evaluate(a):
    roots = prepare_evaluation_domains(a)
    import amcal_burst_structured_bank_protocol as bank
    original = bank.evaluate
    original_report = bank.report_level
    metrics = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        metrics.extend(result)
        return result

    def compact(level, mode, m):
        print(f'{a.attack} | level {level} | radius {strength(a.attack,level):g} | {mode}'
              f' | online {100*m["accuracy"]:.2f}% | labels {m["labels"]}/{m["label_cap"]}'
              f' | oracle {100*m["oracle_all"]:.2f}% | reference drop {m["reference"]["drop_pp"]:+.2f} pp', flush=True)

    bank.evaluate, bank.report_level = capture, compact
    try:
        with arguments(['evaluate', '--burst-dir', roots, '--output', a.output / 'Bank',
                        '--levels', *a.levels, '--include-seen-levels', '--balanced-reward-reference',
                        '--context-lr', a.context_lr, '--budget-fraction', a.budget_fraction,
                        '--selection-cost', a.selection_cost, '--gain-weight', a.gain_weight,
                        '--retention-weight', a.retention_weight, '--gamma', a.gamma,
                        '--query-schedule', 'policy', '--update-policy', 'learned',
                        '--eval-epsilon', a.eval_epsilon, '--modes', 'learned', 'random',
                        '--seed', a.seed, '--device', a.device]):
            bank.main()
    finally:
        bank.evaluate, bank.report_level = original, original_report
    rows = summary_rows(a.attack, metrics)
    result = a.output / 'Results' / f'adversarial_summary_seed{a.seed}.csv'
    write_csv(result, rows)
    display(rows)
    print(f'Summary saved: {result}', flush=True)


def all_attacks(a):
    if a.command == 'train' and a.output.exists():raise ValueError('Use a fresh output root.')
    if a.command == 'train':a.output.mkdir(parents=True)
    for attack in LEVELS:
        command = [sys.executable, str(Path(__file__).resolve()), a.command,
                   '--attack', attack, '--data-dir', str(a.data_dir), '--source-file', a.source_file,
                   '--output', str(a.output / attack), '--seed', str(a.seed), '--device', a.device]
        if a.clean_run is not None:command += ['--clean-run', str(a.clean_run)]
        for key in ('context_lr', 'budget_fraction', 'selection_cost', 'gain_weight', 'retention_weight',
                    'gamma', 'eval_epsilon', 'stage1_epochs', 'cnn_epochs', 'source_epochs', 'bank_epochs',
                    'pgd_steps', 'cw_steps', 'cw_search', 'deepfool_steps'):
            command += ['--' + key.replace('_', '-'), str(getattr(a, key))]
        command += ['--levels', *map(str, a.levels)]
        subprocess.run(command, check=True)
    if a.command == 'evaluate':
        rows = []
        for attack in LEVELS:
            with (a.output / attack / 'Results' / f'adversarial_summary_seed{a.seed}.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows.extend(csv.DictReader(handle))
        target = a.output / 'Results' / f'all_attacks_seed{a.seed}.csv'
        write_csv(target, rows)
        display(rows)
        print(f'All-attack summary: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'evaluate'))
    p.add_argument('--attack', choices=('all', *LEVELS), default='all')
    p.add_argument('--data-dir', type=Path, default=Path('AMCAL/SDNCampus_TEST/Dataset'))
    p.add_argument('--source-file', default='SDNCampus_original.csv')
    p.add_argument('--clean-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--levels', type=int, nargs='+', default=list(range(6)), help='Evaluation levels; 0 is clean, 1..5 are ordered strengths. Training always uses 0/2.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--context-lr', type=float, default=.0005)
    p.add_argument('--budget-fraction', type=float, default=.2)
    p.add_argument('--selection-cost', type=float, default=.0001)
    p.add_argument('--gain-weight', type=float, default=1.)
    p.add_argument('--retention-weight', type=float, default=1.)
    p.add_argument('--gamma', type=float, default=.95)
    p.add_argument('--eval-epsilon', type=float, default=.05)
    p.add_argument('--stage1-epochs', type=int, default=12)
    p.add_argument('--cnn-epochs', type=int, default=30)
    p.add_argument('--source-epochs', type=int, default=30)
    p.add_argument('--bank-epochs', type=int, default=12)
    p.add_argument('--pgd-steps', type=int, default=20)
    p.add_argument('--cw-steps', type=int, default=100)
    p.add_argument('--cw-search', type=int, default=3)
    p.add_argument('--deepfool-steps', type=int, default=25)
    a = p.parse_args()
    for name in ('data_dir', 'clean_run', 'output'):
        if getattr(a, name) is not None:setattr(a, name, getattr(a, name).resolve())
    if a.command == 'train' and a.clean_run is None:p.error('--clean-run is required for training.')
    if not (a.data_dir / a.source_file).is_file():p.error('Clean source CSV not found: ' + str(a.data_dir / a.source_file))
    if a.command == 'train' and any(not (a.clean_run / 'Models' / name).is_file()
                                   for name in ('initial_cnn_model.pth', 'protocol_preprocessing.joblib')):
        p.error('--clean-run needs Models/initial_cnn_model.pth and Models/protocol_preprocessing.joblib.')
    if min(getattr(a, key) for key in ('stage1_epochs', 'cnn_epochs', 'source_epochs', 'bank_epochs', 'pgd_steps', 'cw_steps', 'cw_search', 'deepfool_steps')) < 1:
        p.error('Epoch/attack step counts must be positive.')
    if (not np.isfinite(a.context_lr) or a.context_lr <= 0 or not 0 <= a.budget_fraction <= 1
            or not 0 <= a.gamma <= 1 or not 0 <= a.eval_epsilon <= 1
            or any(not np.isfinite(v) or v < 0 for v in (a.selection_cost, a.gain_weight, a.retention_weight))):
        p.error('Invalid learning, budget or reward settings.')
    if not a.levels or len(set(a.levels)) != len(a.levels) or any(v not in range(6) for v in a.levels):
        p.error('Evaluation levels must be distinct integers 0..5.')
    if a.attack == 'all':all_attacks(a)
    elif a.command == 'train':train(a)
    else:evaluate(a)


if __name__ == '__main__':
    main()
