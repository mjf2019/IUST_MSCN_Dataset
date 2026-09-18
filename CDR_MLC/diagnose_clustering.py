"""Fit-only clustering audit across known levels; NOT a held-out accuracy test."""
import argparse
import json
from pathlib import Path

from cdr_mlc import CDRMLC, Config
from run_cdr_mlc import read_captures, data_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--target', default='label')
    p.add_argument('--level-column')
    p.add_argument('--stream-column')
    p.add_argument('--output', type=Path, default=Path('results/cluster_audit.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-size', type=int, default=2000)
    args = p.parse_args()
    if args.stream_column and (CDRMLC._label_metadata(args.stream_column) or args.stream_column == args.target):
        p.error('Use real capture identifiers, never labels, for stream boundaries')
    if any('shuffle' in Path(f).name.lower() and 'notshuffle' not in Path(f).name.lower() for f in args.files):
        p.error('Use unshuffled captures to inspect temporal trend clustering')
    df, manifest = read_captures(args.files, args.stream_column)
    level = args.level_column or ('_source_level' if '_source_level' in df else None)
    model = CDRMLC(Config(random_state=args.seed), stream_column='_capture_id').fit(df, args.target)
    report = {'scope': 'IN-SAMPLE clustering diagnostic only. No held-out classification score.',
              'files': manifest, 'data_audit': data_audit(df, args.target),
              'diagnostics': model.diagnostics(df, args.target, level, args.sample_size, True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda x: x.item(), allow_nan=False), encoding='utf-8')
    d = report['diagnostics']
    print(json.dumps({k: d[k] for k in ('cluster_counts', 'silhouette', 'agglomerative_silhouette',
                                       'cluster_by_level', 'level_adjusted_rand', 'class_adjusted_rand') if k in d}, indent=2))
    print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
