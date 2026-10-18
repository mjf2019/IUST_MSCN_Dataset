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
    parser.add_argument('--official-split-dir',type=Path)
    parser.add_argument('--original-split-dir', type=Path, help='Original oversampled train/validation files for CNN and Context pretraining.')
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
    original_split = extra.original_split_dir.resolve() if extra.original_split_dir else None
    episodic_loaders = None
    if original_split is not None:
        def original_load_data(dataset_path='Dataset', csv_filename=None):
            nonlocal episodic_loaders
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
            source = pd.read_csv(domain_path(0))
            train = pd.read_csv(original_split / 'oversampled_train_dataset.csv')
            validation = pd.read_csv(original_split / 'oversampled_validation_dataset.csv')
            for frame in (train, validation):
                if list(frame.columns) != list(source.columns):
                    raise ValueError('Original split schema differs from Burst level 0.')
            train_ids, val_ids, test_ids = core.split_source_rows(source, core.RUN_SEED)
            # Original supervised files are preserved. Exclude exact feature overlap
            # from all target development/holdout pools, rather than train on test rows.
            supervised_groups = set(core.feature_groups(train)) | set(core.feature_groups(validation))
            groups = core.feature_groups(source)
            pools = []
            for ids in (train_ids, val_ids, test_ids):
                pools.append(np.asarray([i for i in ids if groups[i] not in supervised_groups], dtype=np.int64))
            train_ids, val_ids, test_ids = pools
            if min(map(len, pools)) < 1:
                raise ValueError('Original supervised files overlap all rows of a Burst pool. Need independent source data; cannot silently reuse supervised rows as heldout.')
            encoder = LabelEncoder().fit(train.iloc[:, -1])
            scaler = StandardScaler().fit(train.iloc[:, :-1].to_numpy())
            ytrain = encoder.transform(train.iloc[:, -1].to_numpy())
            counts = np.bincount(ytrain, minlength=len(encoder.classes_))
            def dataset(frame):
                x = scaler.transform(frame.iloc[:, :-1].to_numpy()).astype(np.float32)
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite original/target features.')
                y = encoder.transform(frame.iloc[:, -1].to_numpy())
                return TensorDataset(torch.tensor(x, device=core.device).unsqueeze(1),
                                     torch.tensor(y, dtype=torch.long, device=core.device))
            core.save_preprocessing(source, str(root), domain_path(0).name, scaler, encoder,
                counts, train_ids, val_ids, test_ids)
            artifact_path = Path('Models/protocol_preprocessing.joblib')
            artifact = joblib.load(artifact_path)
            artifact['base_split_protocol'] = 'original oversampled train and validation; target source pools exclude exact-feature overlap with both'
            artifact['original_train_sha256'] = core.file_sha256(original_split / 'oversampled_train_dataset.csv')
            artifact['reference_csv'] = str(original_split / 'oversampled_validation_dataset.csv')
            artifact['reference_sha256'] = core.file_sha256(artifact['reference_csv'])
            joblib.dump(artifact, artifact_path)
            ds = dataset(train)
            sampler = WeightedRandomSampler(1.0 / counts[ytrain], len(ytrain))
            train_loader = DataLoader(ds, batch_size=64, sampler=sampler, drop_last=True)
            full_loader = DataLoader(ds, batch_size=64, sampler=sampler)
            validation_loader = DataLoader(dataset(validation), batch_size=64)
            # No heldout score is used in training or model selection.
            test_loader = DataLoader(dataset(source.iloc[test_ids]), batch_size=64)
            episodic_loaders = (DataLoader(dataset(source.iloc[train_ids]), batch_size=64),
                                DataLoader(dataset(source.iloc[val_ids]), batch_size=64))
            print(f'Original supervised train: {len(train)}; validation: {len(validation)}')
            print(f'Independent Burst RL pools: train={len(train_ids)}, validation={len(val_ids)}, heldout={len(test_ids)}')
            return train_loader, validation_loader, test_loader, full_loader, len(counts), source.shape[1]-1, counts
        core.load_data = original_load_data
    if extra.official_split_dir is not None:
        def official_load_data(dataset_path='Dataset',csv_filename=None):
            from sklearn.preprocessing import StandardScaler,LabelEncoder
            from torch.utils.data import TensorDataset,DataLoader
            source=pd.read_csv(domain_path(0))
            split_root=extra.official_split_dir.resolve()
            partitions=json.loads((split_root/'official_partitions.json').read_text())
            pools=[np.asarray(partitions[k],dtype=np.int64) for k in ('train','validation','test')]
            if sorted(np.concatenate(pools).tolist())!=list(range(len(source))):raise ValueError('Official partitions must cover source once')
            # Remove exact-feature cross-partition overlap; preserve original membership.
            groups=core.feature_groups(source);seen=set();filtered=[]
            for ids in pools:
                keep=np.array([i for i in ids if groups[i] not in seen],dtype=np.int64)
                filtered.append(keep);seen.update(groups[i] for i in ids)
            train_ids,val_ids,test_ids=filtered
            if min(map(len,filtered))==0:raise ValueError('Empty official partition after duplicate exclusion')
            train=source.iloc[train_ids];encoder=LabelEncoder().fit(train.iloc[:,-1]);scaler=StandardScaler().fit(train.iloc[:,:-1])
            counts=np.bincount(encoder.transform(train.iloc[:,-1]),minlength=len(encoder.classes_))
            core.save_preprocessing(source,str(root),domain_path(0).name,scaler,encoder,counts,train_ids,val_ids,test_ids)
            ref=Path('Models/official_reference_validation.csv');source.iloc[val_ids].to_csv(ref,index=False)
            ap=Path('Models/protocol_preprocessing.joblib');artifact=joblib.load(ap)
            artifact['reference_csv']=str(ref.resolve());artifact['reference_sha256']=core.file_sha256(ref)
            artifact['base_split_protocol']='official train/validation/test; exact-feature overlap removed from later partitions; no oversampling'
            joblib.dump(artifact,ap)
            manifest=Path('Models/protocol_manifest.json');m=json.loads(manifest.read_text());m['split_rule']=artifact['base_split_protocol'];manifest.write_text(json.dumps(m,indent=2))
            def loader(ids,shuffle=False):
                frame=source.iloc[ids];x=scaler.transform(frame.iloc[:,:-1]).astype(np.float32);y=encoder.transform(frame.iloc[:,-1])
                ds=TensorDataset(torch.tensor(x,device=core.device).unsqueeze(1),torch.tensor(y,dtype=torch.long,device=core.device))
                return DataLoader(ds,batch_size=64,shuffle=shuffle)
            print('Official split sizes after overlap removal:',list(map(len,filtered)))
            return loader(train_ids,True),loader(val_ids),loader(test_ids),loader(train_ids),len(counts),source.shape[1]-1,counts
        core.load_data=official_load_data
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
        if episodic_loaders is not None:
            kwargs["episode_loaders"] = episodic_loaders
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
