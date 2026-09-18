"""Reproducible train/test runner. Run --help for paper scenarios and diagnostics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from cdr_mlc import CDRMLC, Config, BASE_FEATURES, SENSITIVE, evaluate


def read_captures(paths, stream_column=None):
    frames, manifest = [], []
    for i, path in enumerate(paths):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Dataset not found: {path}')
        df = pd.read_csv(path)
        if stream_column:
            if stream_column not in df or df[stream_column].isna().any():
                raise ValueError(f'{path}: invalid stream column {stream_column}')
            # File boundary ALWAYS resets; user may additionally supply capture IDs.
            keys = pd.factorize(df[stream_column], sort=False)[0]
            df['_capture_id'] = [f'{i}:{k}' for k in keys]
        else:
            df['_capture_id'] = str(i)
        # Source level annotation is ONLY for reporting, never features/routing.
        if path.stem in ('level_1', 'level_2', 'level_3'):
            df['_source_level'] = int(path.stem[-1])
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        manifest.append({'path': str(path), 'sha256': h.hexdigest(), 'rows': len(df)})
        frames.append(df)
    return pd.concat(frames, ignore_index=True), manifest


def data_audit(df, target):
    result = {'rows': len(df), 'classes': df[target].value_counts().to_dict(),
              'class_runs': int(df[target].ne(df[target].shift()).sum()),
              'ignored_columns': sorted(set(df.columns) - set(BASE_FEATURES) - {target, '_capture_id'}),
              'sensitive_feature_summary': df[list(SENSITIVE)].describe().to_dict()}
    if result['class_runs'] <= df[target].nunique() * df['_capture_id'].nunique():
        result['order_warning'] = 'Rows appear class-blocked. File order is NOT verified capture chronology.'
    return result


def run_experiment(train_paths, test_paths, output, config=None, target='label',
                   stream_column=None, level_column=None, silhouette_samples=2000,
                   compare_agglomerative=False, allow_shuffled=False):
    if not train_paths or not test_paths:
        raise ValueError('Both train and test paths are required')
    resolved_train = {Path(p).resolve() for p in train_paths}
    if resolved_train.intersection(Path(p).resolve() for p in test_paths):
        raise ValueError('The same file cannot be both training and test data')
    for p in [*train_paths, *test_paths]:
        name = Path(p).name.lower()
        if 'shuffle' in name and 'notshuffle' not in name and not allow_shuffled:
            raise ValueError(f'{p}: shuffled rows are not a chronological stream. Use raw ordered captures; '
                             '--allow-shuffled is an explicitly non-temporal diagnostic only')
    if stream_column and CDRMLC._label_metadata(stream_column):
        raise ValueError('Do not construct stream boundaries from labels')
    train, train_manifest = read_captures(train_paths, stream_column)
    test, test_manifest = read_captures(test_paths, stream_column)
    if target not in train or target not in test:
        raise ValueError(f'Missing target: {target}')
    if stream_column == target:
        raise ValueError('Target cannot define stream boundaries')
    model = CDRMLC(config, stream_column='_capture_id').fit(train, target)
    metrics, predictions, clusters = evaluate(model, test, target)
    audits = {'train': data_audit(train, target), 'test': data_audit(test, target)}
    for split in audits:
        if 'order_warning' in audits[split]:
            warnings.warn(f"{split}: {audits[split]['order_warning']}")
    # Exact-feature matches can be legitimate identical flows; report, don't silently remove.
    train_hash = pd.util.hash_pandas_object(train[list(BASE_FEATURES)], index=False)
    test_hash = pd.util.hash_pandas_object(test[list(BASE_FEATURES)], index=False)
    audits['test_rows_matching_train_features'] = int(test_hash.isin(train_hash).sum())
    audits['overlap_note'] = 'Feature equality is an audit flag, not proof of identical captures.'
    def diagnostic(df):
        level = level_column or ('_source_level' if '_source_level' in df and df['_source_level'].notna().all() else None)
        return model.diagnostics(df, target, level, silhouette_samples, compare_agglomerative)
    report = {'config': asdict(model.config), 'metrics': metrics,
              'training_diagnostics': diagnostic(train), 'test_diagnostics': diagnostic(test),
              'data_audit': audits, 'train_files': train_manifest, 'test_files': test_manifest,
              'classification_features': list(model.classification_features),
              'base_feature_count': len(BASE_FEATURES),
              'missing_classes_per_expert': model.missing_classes_,
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                           'numpy': np.__version__, 'pandas': pd.__version__},
              'protocol': 'Fit preprocessing, MBK, RF on training rows only; one routed RF per test row',
              'limitations': [
                  'Cluster IDs are arbitrary geometric groups, not certified low/medium/high congestion.',
                  'Input chronology must be validated from capture metadata; CSV order alone is insufficient.',
                  'Paper Table 5 metrics are not assumed or reproduced by construction.',
                  'Single-level training cannot establish coverage of all three physical congestion levels.',
              ]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2, default=lambda x: x.item(), allow_nan=False), encoding='utf-8')
    pd.DataFrame({'row': np.arange(len(test)), 'capture_id': test['_capture_id'],
                  'true_label': test[target], 'prediction': predictions, 'cluster': clusters}).to_csv(output / 'predictions.csv', index=False)
    model.training_class_counts_.to_csv(output / 'train_cluster_class_counts.csv')
    joblib.dump(model, output / 'model.joblib')
    print(json.dumps({'accuracy': metrics['accuracy'], 'train_clusters': report['training_diagnostics']['cluster_counts'],
                      'test_clusters': report['test_diagnostics']['cluster_counts'],
                      'train_silhouette': report['training_diagnostics']['silhouette'],
                      'test_silhouette': report['test_diagnostics']['silhouette'],
                      'report': str(output / 'report.json')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['1', '2', '3', '4', '5', 'all'])
    parser.add_argument('--short-dir', type=Path, default=Path(__file__).parent / 'DATASETS/CDR-MLC/scale_1/Short')
    parser.add_argument('--long-file', type=Path, default=Path(__file__).parent / 'DATASETS/CDR-MLC/scale_1/Long/CDR-MLC-notShuffle.csv')
    parser.add_argument('--train', nargs='+')
    parser.add_argument('--test', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('results/cdr_mlc'))
    parser.add_argument('--target', default='label')
    parser.add_argument('--stream-column', help='Real capture/session identifier, available before classification')
    parser.add_argument('--level-column', help='Known congestion annotation, used only in diagnostics')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-scale', action='store_true', help='Ablation: disable training-only trend standardization')
    parser.add_argument('--std-ddof', type=int, choices=[0, 1], default=1)
    parser.add_argument('--silhouette-samples', type=int, default=2000)
    parser.add_argument('--compare-agglomerative', action='store_true')
    parser.add_argument('--allow-shuffled', action='store_true', help='Non-temporal diagnostic only')
    args = parser.parse_args()
    if args.scenario and (args.train or args.test):
        parser.error('Choose scenario OR explicit train/test files')
    if not args.scenario and not (args.train and args.test):
        parser.error('Specify --scenario or both --train and --test')
    config = Config(random_state=args.seed, scale_trends=not args.no_scale, std_ddof=args.std_ddof)
    shared = dict(config=config, target=args.target, stream_column=args.stream_column,
                  level_column=args.level_column, silhouette_samples=args.silhouette_samples,
                  compare_agglomerative=args.compare_agglomerative, allow_shuffled=args.allow_shuffled)
    levels = [args.short_dir / f'level_{i}.csv' for i in (1, 2, 3)]
    scenarios = {'1': ([levels[0]], [levels[1]]), '2': ([levels[0]], [levels[2]]),
                 '3': ([levels[1]], [levels[2]]), '4': (levels, [args.long_file]),
                 '5': ([args.long_file], levels)}
    try:
        if args.scenario:
            for number in (scenarios if args.scenario == 'all' else [args.scenario]):
                train_paths, test_paths = scenarios[number]
                run_experiment(train_paths, test_paths, args.output / f'scenario_{number}', **shared)
        else:
            run_experiment(args.train, args.test, args.output, **shared)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
