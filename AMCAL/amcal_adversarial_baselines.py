"""Existing seven baselines on exactly the mapped AMCAL clean/level-2 domains.

Requires completed AMCAL training per attack. Never trains on TEST source IDs.
The existing baseline implementations and hyperparameter defaults are reused.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import amcal_adversarial_protocol as protocol
from baselines.adversarial_domains import VERSION, LEVELS, digest, domain_name

METHODS = ('cnn', 'ral', 'af', 'dfe', 'arf', 'calmid', 'micfoal')


def validate_run(root, attack):
    manifest = json.loads((root / 'adversarial_training.json').read_text())
    if manifest['version'] != VERSION or manifest['attack'] != attack or manifest['status'] != 'complete':
        raise ValueError('Complete matching AMCAL training required: ' + str(root))
    domains = root / 'TrainingDomains'
    if digest(domains / domain_name(0)) != manifest['source_sha256']:
        raise ValueError('Clean training CSV changed.')
    for level, record in manifest['domains'].items():
        path = domains / domain_name(int(level))
        mapping = path.with_name(path.stem + '_rows.csv')
        if digest(path) != record['csv_sha256'] or digest(mapping) != record['row_map_sha256']:
            raise ValueError('AMCAL training attack/map changed.')
    return manifest


def command_for(a, attack):
    root = a.amcal_run / attack if a.attack == 'all' else a.amcal_run
    output = a.output / attack if a.attack == 'all' else a.output
    manifest = validate_run(root, attack)
    training = root / 'TrainingDomains'
    domains = training
    if a.command == 'evaluate':
        config = dict(manifest['training_settings'])
        config.update(output=root, attack=attack, data_dir=a.data_dir,
                      source_file=a.source_file, levels=a.levels, seed=a.seed, device=a.device)
        domains = protocol.prepare_evaluation_domains(SimpleNamespace(**config))
    command = [sys.executable, str(Path(__file__).with_name('amcal_burst_baselines.py')),
               a.command, '--burst-dir', str(domains), '--source-run', str(root / 'Bank'),
               '--output', str(output), '--train-levels', '0', '2',
               '--methods', *a.methods, '--seed', str(a.seed), '--device', a.device,
               '--budget-fraction', str(a.budget_fraction), '--jobs', str(a.jobs),
               '--dfe-epochs', str(a.dfe_epochs), '--af-epochs', str(a.af_epochs),
               '--patience', str(a.patience), '--batch-size', str(a.batch_size)]
    if a.command == 'evaluate':
        command += ['--training-domain-dir', str(training), '--include-seen-levels',
                    '--levels', *map(str, a.levels)]
    return command, output


def aggregate(a, attacks):
    selected = {}
    for attack in attacks:
        output = a.output / attack if a.attack == 'all' else a.output
        for path in sorted((output / 'Results').glob('*_metrics.json'), key=lambda item: item.stat().st_mtime):
            metrics = json.loads(path.read_text())
            if metrics.get('execution_config', {}).get('seed') != a.seed or metrics['execution_config'].get('budget') != a.budget_fraction:
                continue
            method = path.name.split('_per', 1)[0]
            level = metrics.get('level')
            if method not in a.methods or level not in a.levels:
                continue
            ref = metrics['reference']
            selected[(attack, method, level)] = dict(Attack=attack, Method=method, Level=level,
                             Strength=protocol.strength(attack, level),
                             Domain='Seen' if level in (0, 2) else 'Unseen',
                             Online=100*metrics['accuracy'], MacroF1=100*metrics['macro_f1'],
                             Labels=metrics['labels'], Cap=metrics['label_cap'],
                             LabelPct=100*metrics['actual_label_fraction'], RefDrop=ref['drop_pp'])
    rows = list(selected.values())
    if rows:
        target = a.output / 'Results' / f'all_baselines_seed{a.seed}.csv'
        protocol.write_csv(target, rows)
        print(f'All baseline results: {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('train', 'evaluate'))
    p.add_argument('--attack', choices=('all', *LEVELS), default='all')
    p.add_argument('--amcal-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data-dir', type=Path, default=Path('AMCAL/SDNCampus_TEST/Dataset'))
    p.add_argument('--source-file', default='SDNCampus_original.csv')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--levels', nargs='+', type=int, default=list(range(6)))
    p.add_argument('--budget-fraction', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--dfe-epochs', type=int, default=200)
    p.add_argument('--af-epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    if not a.methods or len(set(a.methods)) != len(a.methods):p.error('Distinct methods required.')
    if not a.levels or len(set(a.levels)) != len(a.levels) or any(v not in range(6) for v in a.levels):
        p.error('Distinct evaluation levels 0..5 required.')
    if not 0 <= a.budget_fraction <= 1 or min(a.dfe_epochs, a.af_epochs, a.patience) < 1 or a.batch_size < 2:
        p.error('Invalid budget/training settings.')
    for name in ('amcal_run', 'output', 'data_dir'):
        setattr(a, name, getattr(a, name).resolve())
    attacks = list(LEVELS) if a.attack == 'all' else [a.attack]
    # Validate all source runs before starting any lengthy baseline training.
    for attack in attacks:
        root = a.amcal_run / attack if a.attack == 'all' else a.amcal_run
        validate_run(root, attack)
        output = a.output / attack if a.attack == 'all' else a.output
        if a.command == 'train' and output.exists():
            p.error('Use a fresh baseline output: ' + str(output))
    for attack in attacks:
        command, output = command_for(a, attack)
        print(f'{attack} | {a.command} | clean + level 2 | methods: {", ".join(a.methods)}', flush=True)
        subprocess.run(command, check=True)
    if a.command == 'evaluate':aggregate(a, attacks)


if __name__ == '__main__':main()
