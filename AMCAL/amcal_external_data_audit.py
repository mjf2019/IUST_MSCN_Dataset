"""Read-only, full-CSV preflight for external AMCAL experiments; no PyTorch required."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ('DReLAB_TEST', 'ASNM_NPBO_TEST', 'ASNM_TUN_TEST')
IDENTIFIERS = {'id', 'srcip', 'dstip', 'srcmac', 'dstmac', 'srcaddr', 'dstaddr',
               'srcport', 'dstport', 'tsesstart', 'tsesend', 'starttime'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def excluded(column):
    name = str(column).strip().lower()
    return name.startswith(('label', 'unnamed:')) or name == 'n_actions' or name in IDENTIFIERS


def separator(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        sample = f.read(32768)
    return csv.Sniffer().sniff(sample, delimiters=',;\t').delimiter


def binary_label(values):
    out = values.astype(str).str.strip().str.lower().map(
        {'false': 0, 'true': 1, '0': 0, '1': 1, '0.0': 0, '1.0': 1})
    if out.isna().any(): raise ValueError('Unrecognized binary label values.')
    return out.astype(int)


def roles_and_targets(frame, dataset, path):
    if dataset == 'DReLAB_TEST':
        parts = [part.lower() for part in path.parts]
        if path.name.lower() == 'benign.csv': role = 'benign'
        elif 'malicious' in parts: role = 'direct'
        elif ('rf' in parts or 'wnd' in parts) and ('ddqn' in parts or 'sarsa' in parts): role = 'adversarial'
        else: raise ValueError('Unrecognized DReLAB task path; raw flow files are not task CSVs.')
        expected = 0 if role == 'benign' else 1
        if 'Label' not in frame: raise ValueError('DReLAB Label column missing.')
        labels = frame['Label'].astype(str).str.strip().str.lower()
        accepted = {'0', '0.0'} if expected == 0 else {'1', '1.0'}
        if role == 'adversarial': accepted.add('adversarial')
        if not labels.isin(accepted).all(): raise ValueError('DReLAB label conflicts with task path.')
        return pd.Series(role, index=frame.index), pd.Series(expected, index=frame.index)
    # NPBOv2 calls its ternary label "label"; TUN calls it "label_3".
    ternary = next((c for c in ('label_3', 'label') if c in frame), None)
    if ternary is None: raise ValueError('No ternary ASNM role label found.')
    codes = pd.to_numeric(frame[ternary], errors='raise')
    roles = codes.map({1: 'direct', 2: 'adversarial', 3: 'benign'})
    if roles.isna().any(): raise ValueError('ASNM ternary labels must be 1, 2, 3.')
    targets = (codes != 3).astype(int)
    if 'label_2' in frame and not binary_label(frame['label_2']).equals(targets):
        raise ValueError('ASNM binary and ternary labels disagree.')
    for col in ('label_poly', 'label_poly_o', 'label_poly_s'):
        if col in frame:
            prefix = pd.to_numeric(frame[col].astype(str).str.split('_').str[0], errors='coerce')
            if not prefix.eq(codes).all(): raise ValueError(f'{col} disagrees with ternary labels.')
    return roles, targets


def audit_file(path, dataset, repo_root, chunksize=10000):
    sep = separator(path)
    result = dict(file=path.relative_to(repo_root).as_posix(), separator=sep,
                  sha256=sha256(path), rows=0, roles={}, binary_classes={},
                  label_values={}, excluded_columns=[], candidate_features=[])
    role_counts, target_counts, labels = Counter(), Counter(), {}
    bad_numeric, missing, infinite = Counter(), Counter(), Counter()
    columns = None
    for frame in pd.read_csv(path, sep=sep, encoding='utf-8-sig', chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = list(frame.columns)
            result['columns'] = columns
            result['excluded_columns'] = [c for c in columns if excluded(c)]
            result['candidate_features'] = [c for c in columns if not excluded(c)]
        roles, targets = roles_and_targets(frame, dataset, path)
        role_counts.update(roles.tolist()); target_counts.update(map(str, targets.tolist()))
        result['rows'] += len(frame)
        for col in columns:
            if str(col).lower().startswith('label'):
                values = labels.setdefault(col, Counter())
                values.update(frame[col].astype(str).tolist())
        for col in result['candidate_features']:
            raw = frame[col]
            numeric = pd.to_numeric(raw, errors='coerce')
            bad_numeric[col] += int((numeric.isna() & raw.notna()).sum())
            missing[col] += int(numeric.isna().sum())
            infinite[col] += int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if not result['rows']: raise ValueError('Empty CSV.')
    result.update(roles=dict(role_counts), binary_classes=dict(target_counts),
        label_values={k: dict(v) for k, v in labels.items()},
        nonnumeric_values={k: v for k,v in bad_numeric.items() if v},
        missing_values={k: v for k,v in missing.items() if v},
        infinite_values={k: v for k,v in infinite.items() if v},
        candidate_feature_count=len(result['candidate_features']))
    result['numeric_feature_count'] = sum(not bad_numeric[c] for c in result['candidate_features'])
    return result


def audit(repo_root, chunksize=10000):
    report = dict(version=1, purpose='data readiness only; not training or test results',
        method='PairOnly; one Context; three actions; Context LR 0.0005; 20% label cap',
        limitations=['No source/attack identity alignment is inferred from CSV indices.',
            'Feature selection, imputation and scaling must fit TRAIN only.',
            'Full feature/identity leakage checks and split creation are still required before training.',
            'Candidate feature counts are not a trained feature selection.'], datasets=[])
    for dataset in DATASETS:
        root = repo_root/'AMCAL'/dataset/'Dataset_official_v1'
        record = dict(name=dataset, files=[], skipped=[], errors=[])
        paths = sorted(root.rglob('*.csv'))
        for path in paths:
            if dataset == 'DReLAB_TEST' and path.name.lower() == 'raw_labelled_flow.csv':
                record['skipped'].append(path.relative_to(repo_root).as_posix()); continue
            try: record['files'].append(audit_file(path, dataset, repo_root, chunksize))
            except Exception as error:
                record['errors'].append(dict(file=path.relative_to(repo_root).as_posix(), error=str(error)))
        if not paths: record['errors'].append(dict(error='No CSV files found.'))
        total = Counter()
        for item in record['files']: total.update(item['roles'])
        record['role_rows'] = dict(total)
        record['ready_for_split_design'] = bool(record['files']) and not record['errors']
        report['datasets'].append(record)
        print(f"{dataset}: {len(record['files'])} audited CSVs; roles={dict(total)}; errors={len(record['errors'])}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    p.add_argument('--chunksize', type=int, default=10000)
    a = p.parse_args()
    if a.chunksize < 1: p.error('--chunksize must be positive.')
    root = a.repo_root.resolve()
    output = a.output or root/'AMCAL'/'external_datasets_audit.json'
    if output.suffix.lower() != '.json': p.error('--output must be a JSON report.')
    if any(part.lower() == 'dataset_official_v1' for part in output.resolve().parts):
        p.error('Report output cannot be inside raw dataset folders.')
    report = audit(root, a.chunksize)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf-8')
    print('Report:', output.resolve(), flush=True)
    if any(record['errors'] for record in report['datasets']): raise SystemExit(2)


if __name__ == '__main__': main()
