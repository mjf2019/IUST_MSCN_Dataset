"""Three-action transfer protocol on supplied Burst domains.

Row correspondence must be supplied by source_row maps, or explicitly assumed
for exploratory runs. Matching label sequences alone is not provenance proof.
The clean Burst level-0 file is split once; the same source IDs are kept apart
in every training/validation/test domain. No class/flow Context routing.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import amcal_three_action_protocol as core


def mapped_domain(path, artifact, mapping_dir=None, assume_aligned=False):
    frame = pd.read_csv(path)
    expected = artifact['feature_names'] + [artifact['label_name']]
    if list(frame.columns) != expected:
        raise ValueError(f'Burst schema mismatch: {path}')
    mapping_path = None if mapping_dir is None else mapping_dir / (path.stem + '_rows.csv')
    if mapping_path is not None and mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
        if 'source_row' not in mapping or len(mapping) != len(frame):
            raise ValueError('Each mapping needs one zero-based source_row per domain row.')
        raw = pd.to_numeric(mapping['source_row'], errors='raise').to_numpy()
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
            raise ValueError('Invalid source_row IDs.')
        ids = raw.astype(np.int64)
        alignment = 'explicit source_row map'
    elif assume_aligned and len(frame) == artifact['source_rows']:
        ids = np.arange(len(frame))
        alignment = 'ASSUMED original row order; NOT verified provenance'
    else:
        raise ValueError(f'Row provenance required for {path.name}: supply --burst-mapping-dir or explicitly --burst-assume-row-aligned for exploratory tests.')
    if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= artifact['source_rows']).any():
        raise ValueError('Source row IDs must be unique and within level-0 bounds.')
    if not np.array_equal(frame.iloc[:, -1].to_numpy(), artifact['source_labels'][ids]):
        raise ValueError(f'Label order contradicts source-row correspondence: {path}')
    return frame, ids, alignment


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--burst-dir', type=Path, required=True)
    parser.add_argument('--burst-mapping-dir', type=Path)
    parser.add_argument('--burst-assume-row-aligned', action='store_true')
    extra, remaining = parser.parse_known_args()
    root = extra.burst_dir.resolve()
    maps = extra.burst_mapping_dir.resolve() if extra.burst_mapping_dir else None
    if not extra.burst_assume_row_aligned and maps is None:
        parser.error('Provide source-row maps, or explicitly acknowledge exploratory alignment with --burst-assume-row-aligned.')
    def domain_path(level):
        if not float(level).is_integer() or level < 0:
            raise ValueError('Burst levels must be nonnegative integers.')
        return root / f'AdvBurstTrainedBurst_size_{int(level)}_Step_2000.csv'
    if extra.burst_assume_row_aligned:
        print('BURST: row order is ASSUMED across intensities; matching labels do not prove provenance.')
    calls = 0
    def build_domains(cnn, features, labels):
        nonlocal calls
        split = ('train_ids', 'val_ids')[calls]
        calls += 1
        artifact = joblib.load('Models/protocol_preprocessing.joblib')
        selected_ids = np.asarray(artifact[split])
        domains = {}
        manifest = []
        for level in core.TRAIN_PERTURBATIONS:
            path = domain_path(level)
            if level == 0:
                domains[level] = features.detach()
                manifest.append(dict(level=level, sha256=core.file_sha256(path), alignment='level-0 source IDs'))
                continue
            frame, ids, alignment = mapped_domain(path, artifact, maps, extra.burst_assume_row_aligned)
            lookup = {int(source_id): position for position, source_id in enumerate(ids)}
            if any(int(i) not in lookup for i in selected_ids):
                raise ValueError(f'{path.name} does not cover all {split} rows.')
            subset = frame.iloc[[lookup[int(i)] for i in selected_ids]]
            encoded = artifact['label_encoder'].transform(subset.iloc[:, -1].to_numpy())
            if not np.array_equal(encoded, labels.detach().cpu().numpy()):
                raise ValueError('Domain labels do not match clean partition.')
            x = artifact['scaler'].transform(subset.iloc[:, :-1].to_numpy()).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite Burst features.')
            domains[level] = torch.tensor(x, device=features.device).reshape_as(features)
            manifest.append(dict(level=level, sha256=core.file_sha256(path), alignment=alignment))
        report = Path('burst_training_domains.json')
        current = json.loads(report.read_text()) if report.exists() else {}
        current[split] = dict(rows=len(selected_ids), domains=manifest,
                             protocol='supplied Burst CSVs, not generated FGSM; fixed level-0 source partition')
        report.write_text(json.dumps(current, indent=2))
        print(f'BURST supplied domains for {split}: {list(domains)}; held-out source IDs excluded.')
        return domains
    core.build_training_domains = build_domains
    # Keep explicit domain provenance in selector metadata rather than FGSM wording.
    original_train = core.train_adversarial_system
    def train_with_metadata(*args, **kwargs):
        result = original_train(*args, **kwargs)
        path = Path('Models/adversarial_best_dqn.pth')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['training_shift_protocol'] = 'supplied Burst domains; fixed source-ID train/validation/test partition'
        checkpoint['burst_domain_manifest'] = json.loads(Path('burst_training_domains.json').read_text())
        torch.save(checkpoint, path)
        return result
    core.train_adversarial_system = train_with_metadata
    # Caller supplies command/output/etc.; force level-0 as the source, no oversampled files.
    if '--source-file' in remaining or '--data-dir' in remaining:
        parser.error('Use --burst-dir; source is the clean level-0 CSV.')
    sys.argv = [sys.argv[0]] + remaining + ['--data-dir', str(root), '--source-file', domain_path(0).name]
    if 'evaluate' in remaining and extra.burst_assume_row_aligned and '--row-map' not in remaining:
        sys.argv += ['--assume-row-aligned']
    core.cli()


if __name__ == '__main__':
    main()
