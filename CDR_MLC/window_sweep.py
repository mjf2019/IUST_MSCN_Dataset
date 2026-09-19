"""Per-application, three-level CDR-MLC timing-window diagnostic.

Application-conditioned exploratory analysis, NOT deployable application routing.
Temporal 60/20/20 splits; fit on training, compare windows on validation only.
No window crosses a file, partition, invalid timing row or purged-flow gap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Set defaults before numerical imports; reproducible and avoids oversubscription.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             balanced_accuracy_score, silhouette_score)
from threadpoolctl import threadpool_limits

TIMING = ['TcpRtt', 'SynAck', 'AckDat']
STATS = ['mean', 'max', 'median', 'min', 'std']
LEVELS = ['Low', 'Medium', 'High']
SERVICES = {'HTTP': ('192.168.2.122', 8080), 'SFTP': ('192.168.2.120', 22),
            'SMTP': ('192.168.2.120', 8025), 'SSH': ('192.168.2.120', 22),
            'Video': ('192.168.2.121', 5000)}
CLIENT = '192.168.1.111'
DEFAULT_WINDOWS = [1, 3, 5, 10, 20, 50, 100]
DEFAULT_SEEDS = [0, 21, 42, 84, 123]


def load_capture(path, application, level):
    """Same service endpoint filter as the current paper notebook; preserve provenance."""
    d = pd.read_csv(path, low_memory=False, on_bad_lines='error')
    d.columns = d.columns.str.strip()
    required = ['StartTime', 'SrcAddr', 'DstAddr', 'Proto', 'Sport', 'Dport', *TIMING]
    if set(required) - set(d):
        raise ValueError(f'{path.name}: missing fields {set(required) - set(d)}')
    for c in d.select_dtypes('object'):
        d[c] = d[c].str.strip()
    d['source_row'] = np.arange(len(d)) + 2
    server, port = SERVICES[application]
    sport = pd.to_numeric(d.Sport, errors='coerce')
    dport = pd.to_numeric(d.Dport, errors='coerce')
    tcp = d.Proto.eq('tcp')
    forward = tcp & d.SrcAddr.eq(CLIENT) & d.DstAddr.eq(server) & dport.eq(port)
    reverse = tcp & d.SrcAddr.eq(server) & d.DstAddr.eq(CLIENT) & sport.eq(port)
    if reverse.any():
        raise ValueError(f'{path.name}: reverse records need explicit canonicalization')
    audit = {'file': path.name, 'application': application, 'level': level,
             'raw_records': len(d), 'service_records': int(forward.sum()),
             'excluded_nonservice': int((~forward).sum()),
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    d = d.loc[forward, required + ['source_row']].copy()
    if d.empty:
        raise ValueError(f'{path.name}: no service records')
    d['timestamp'] = pd.to_datetime(d.StartTime, format='%Y/%m/%d %H:%M:%S.%f', errors='raise')
    audit['timestamp_inversions_in_export'] = int(d.timestamp.diff().lt(pd.Timedelta(0)).sum())
    d = d.sort_values(['timestamp', 'source_row'], kind='stable').reset_index(drop=True)
    d[TIMING] = d[TIMING].apply(pd.to_numeric, errors='coerce')
    d['timing_valid'] = np.isfinite(d[TIMING]).all(axis=1) & d[TIMING].ge(0).all(axis=1)
    d['application'], d['level'], d['capture'] = application, level, path.stem
    # Bidirectional keys; within a file TCP tuple reuse is purged conservatively.
    def key(row):
        ends = sorted([(row.SrcAddr, int(row.Sport)), (row.DstAddr, int(row.Dport))])
        return (row.Proto, *ends)
    d['flow_key'] = [key(r) for r in d.itertuples()]
    audit.update(invalid_timing=int((~d.timing_valid).sum()),
                 all_zero_timing=int(d[TIMING].eq(0).all(axis=1).sum()),
                 first_timestamp=str(d.timestamp.iloc[0]), last_timestamp=str(d.timestamp.iloc[-1]))
    return d, audit


def prepare_partitions(d, max_window, purge_mode="five-tuple"):
    """Purge reused TCP tuples across splits before selecting common window endpoints."""
    if purge_mode not in ('none', 'five-tuple'):
        raise ValueError('Unknown purge mode')
    a, b = int(.6 * len(d)), int(.8 * len(d))
    reserved_keys = set(d.flow_key.iloc[b:])
    validation_keys = set(d.flow_key.iloc[a:b])
    result, audit = {}, {'capture': d.capture.iloc[0], 'total': len(d), 'reserved_rows': len(d)-b}
    for name, raw, forbidden in [('train', d.iloc[:a], validation_keys | reserved_keys),
                                  ('validation', d.iloc[a:b], reserved_keys)]:
        g = raw.copy().reset_index(drop=True)
        purged = g.flow_key.isin(forbidden) if purge_mode == 'five-tuple' else pd.Series(False, index=g.index)
        eligible = g.timing_valid & ~purged
        # Rolling over the UNCOMPRESSED sequence means windows never bridge removed records.
        g['eligible'] = eligible
        g['common_endpoint'] = eligible.astype(int).rolling(max_window, min_periods=max_window).sum().eq(max_window)
        result[name] = g
        audit.update({f'{name}_raw': len(g), f'{name}_purged_tuple_rows': int(purged.sum()),
                      f'{name}_eligible_rows': int(eligible.sum()),
                      f'{name}_common_endpoints': int(g.common_endpoint.sum())})
    # Purging uses tuple membership only, never held-out timing values or outcomes.
    overlap = set(result['train'].loc[result['train'].eligible, 'flow_key']) & set(result['validation'].loc[result['validation'].eligible, 'flow_key'])
    audit['train_validation_shared_tuples'] = len(overlap)
    if purge_mode == 'five-tuple':
        assert not overlap
    return result, audit


def window_matrix(g, window):
    """All five stats for each of three timings, same eligible endpoints for every W."""
    if window < 1:
        raise ValueError('window must be >= 1')
    values = g[TIMING].where(g.eligible, np.nan)
    out = {}
    for f in TIMING:
        r = values[f].rolling(window, min_periods=window)
        for s in STATS:
            out[f'{f}_{s}'] = r.std(ddof=0) if s == 'std' else getattr(r, s)()
    x = pd.DataFrame(out).loc[g.common_endpoint]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Invalid window or common-endpoint mask')
    return x.to_numpy(dtype=float)


def training_mapping(levels, clusters):
    """One-to-one Hungarian mapping using TRAINING contingency only, equal level weights."""
    table = np.zeros((3, 3), dtype=int)
    np.add.at(table, (np.asarray(levels, dtype=int), np.asarray(clusters, dtype=int)), 1)
    normalized = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    rows, cols = linear_sum_assignment(-normalized)
    mapping = np.empty(3, dtype=int)
    mapping[cols] = rows
    return mapping, table


def fixed_sample(n, size):
    return np.sort(np.random.default_rng(20260918).choice(n, min(n, size), replace=False))


def cluster_metrics(z, truth, clusters, mapping, sample):
    mapped = mapping[clusters]
    labels = clusters[sample]
    silhouette = float(silhouette_score(z[sample], labels)) if 1 < len(np.unique(labels)) < len(labels) else np.nan
    return {'ari': float(adjusted_rand_score(truth, clusters)),
            'ami': float(adjusted_mutual_info_score(truth, clusters)),
            'mapped_balanced_accuracy': float(balanced_accuracy_score(truth, mapped)),
            'silhouette': silhouette,
            'occupied_clusters': len(np.unique(clusters)),
            **{f'recall_{LEVELS[i].lower()}': float(np.mean(mapped[truth == i] == i)) for i in range(3)}}


def run(data_dir, out, windows=DEFAULT_WINDOWS, seeds=DEFAULT_SEEDS, sample_size=1000, source_commit=None, purge_mode="none"):
    windows, seeds = sorted(set(windows)), list(dict.fromkeys(seeds))
    if not windows or min(windows) < 1 or not seeds or sample_size < 3:
        raise ValueError('Need positive windows, seeds and sample_size >= 3')
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    frames, inputs, splits, prepared = {}, [], [], {}
    for app in SERVICES:
        for level in LEVELS:
            frame, audit = load_capture(data_dir/f'{app}_{level}.flow', app, level)
            parts, split = prepare_partitions(frame, max(windows), purge_mode)
            for name, g in parts.items():
                if g.common_endpoint.sum() < 10:
                    raise ValueError(f'{app}/{level}/{name}: fewer than 10 common endpoints; '
                                     'inspect split audit, do not silently bridge purged gaps')
            frames[app, level], prepared[app, level] = frame, parts
            inputs.append(audit); splits.append(split)
    pd.DataFrame(inputs).to_csv(out/'input_audit.csv', index=False)
    pd.DataFrame(splits).to_csv(out/'split_audit.csv', index=False)
    # Exact endpoint provenance makes cross-window comparisons auditable.
    endpoint_rows = []
    for (app, level), parts in prepared.items():
        for name, g in parts.items():
            v = g.loc[g.common_endpoint, ['source_row', 'timestamp']].copy()
            v['application'], v['level'], v['partition'] = app, level, name
            endpoint_rows.append(v)
    pd.concat(endpoint_rows).to_csv(out/'common_endpoints.csv', index=False)
    records, contingency, durations, center_rows = [], [], [], []
    with threadpool_limits(limits=1):
        for app in SERVICES:
            truth = {name: np.concatenate([np.full(int(prepared[app, lev][name].common_endpoint.sum()), j)
                     for j, lev in enumerate(LEVELS)]) for name in ('train', 'validation')}
            samples = {name: fixed_sample(len(y), sample_size) for name, y in truth.items()}
            for window in windows:
                matrices = {name: np.vstack([window_matrix(prepared[app, lev][name], window) for lev in LEVELS])
                            for name in ('train', 'validation')}
                scaler = StandardScaler().fit(matrices['train'])
                z = {name: scaler.transform(x) for name, x in matrices.items()}
                for level in LEVELS:
                    for name in ('train', 'validation'):
                        g = prepared[app, level][name]
                        seconds = (g.timestamp - g.timestamp.shift(window-1)).dt.total_seconds()[g.common_endpoint]
                        durations.append({'application': app, 'level': level, 'partition': name, 'window': window,
                                          'median_span_seconds': float(seconds.median()),
                                          'p95_span_seconds': float(seconds.quantile(.95))})
                for seed in seeds:
                    # Label/level is absent from model inputs and fit; no rebalancing/reassignment.
                    mbk = MiniBatchKMeans(n_clusters=3, init='k-means++', batch_size=1024,
                                          n_init=10, max_iter=100, reassignment_ratio=.01,
                                          random_state=seed).fit(z['train'])
                    train_cluster = mbk.predict(z['train'])
                    if len(np.unique(train_cluster)) != 3:
                        raise ValueError(f'{app}/W{window}/seed{seed}: collapsed training clusters')
                    mapping, train_table = training_mapping(truth['train'], train_cluster)
                    for name in ('train', 'validation'):
                        cluster = train_cluster if name == 'train' else mbk.predict(z[name])
                        m = cluster_metrics(z[name], truth[name], cluster, mapping, samples[name])
                        records.append({'application': app, 'window': window, 'seed': seed,
                                        'partition': name, 'n': len(cluster), 'silhouette_n': len(samples[name]), **m})
                        table = np.zeros((3, 3), dtype=int)
                        np.add.at(table, (truth[name], cluster), 1)
                        for level in range(3):
                            for cid in range(3):
                                contingency.append({'application': app, 'window': window, 'seed': seed,
                                                    'partition': name, 'level': LEVELS[level], 'cluster': cid,
                                                    'training_mapped_level': LEVELS[mapping[cid]],
                                                    'count': int(table[level, cid])})
                    for cid, center in enumerate(scaler.inverse_transform(mbk.cluster_centers_)):
                        center_rows.append({'application': app, 'window': window, 'seed': seed, 'cluster': cid,
                                            'training_mapped_level': LEVELS[mapping[cid]],
                                            **dict(zip([f'{f}_{s}' for f in TIMING for s in STATS], center))})
                last = pd.DataFrame(records)
                sub = last[(last.application == app) & (last.window == window) & (last.partition == 'validation')]
                print(f"{app:5s} W={window:3d} validation ARI={sub.ari.mean():.4f} "
                      f"AMI={sub.ami.mean():.4f} balanced={sub.mapped_balanced_accuracy.mean():.4f}", flush=True)
            pd.DataFrame(records).to_csv(out/'metrics_by_seed.csv', index=False)
    metrics = pd.DataFrame(records)
    fields = ['ari', 'ami', 'mapped_balanced_accuracy', 'silhouette',
              'recall_low', 'recall_medium', 'recall_high']
    summary = metrics.groupby(['application', 'window', 'partition'], sort=False)[fields].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out/'window_summary.csv', index=False)
    pd.DataFrame(contingency).to_csv(out/'contingency_by_seed.csv', index=False)
    pd.DataFrame(durations).to_csv(out/'window_time_spans.csv', index=False)
    pd.DataFrame(center_rows).to_csv(out/'centroids_raw_units.csv', index=False)
    val = summary[summary.partition == 'validation']
    # Explicitly exploratory validation selection, not a final held-out performance claim.
    best = val.sort_values(['ari_mean', 'window'], ascending=[False, True]).groupby('application', sort=False).head(1)
    best = best.set_index('application').reindex(SERVICES).reset_index()
    best.to_csv(out/'best_validation_windows.csv', index=False)
    config = {'source_commit': source_commit, 'purge_mode': purge_mode, 'windows': windows, 'seeds': seeds,
              'raw_records': int(sum(a['raw_records'] for a in inputs)),
              'service_records': int(sum(a['service_records'] for a in inputs)),
              'timing_features': TIMING, 'statistics': STATS, 'std_ddof': 0,
              'split': {'train': .6, 'validation': .2, 'reserved_not_evaluated': .2},
              'common_endpoints': f'All {max(windows)} preceding records including current must be eligible; same rows for all W',
              'purge': ('Training excludes tuples in validation/reserved; validation excludes reserved tuples; gaps invalidate windows' if purge_mode == 'five-tuple' else 'No tuple purge: within-capture temporal validation permits related TCP records across partitions; not independent-flow generalization'),
              'silhouette_sample_size': sample_size, 'silhouette_sampling_seed': 20260918,
              'clusterer': {'name': 'MiniBatchKMeans', 'n_clusters': 3, 'batch_size': 1024,
                            'n_init': 10, 'max_iter': 100, 'reassignment_ratio': .01},
              'scaling': 'StandardScaler fitted separately on training only for each app/window',
              'selection': 'Highest mean validation ARI across seeds; smallest W breaks ties. Exploratory, NOT test accuracy.',
              'mapping': 'Training-only Hungarian mapping, equal weights for Low/Medium/High',
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                           'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'limitations': [('Shared TCP tuples across partitions remain; scores may benefit from within-connection dependence.' if purge_mode == 'none' else 'Conservative tuple purge can also remove distinct connections reusing an ephemeral port; surviving windows form a selected subset.'),
                              'Only one capture per application/level: level is confounded with run conditions.',
                              'Application identity conditions this diagnostic; do not use unknown labels to route deployment.',
                              'Reserved tail not scored; these captures have prior exploratory use and are not globally pristine.',
                              'Overlapping windows are dependent; seed std is optimizer variability, not statistical confidence.',
                              'Row-count windows have different time spans across applications/captures.',
                              'No cross-level boundary transitions tested; windows reset at capture boundaries.'],
              'elapsed_seconds': time.time()-start}
    (out/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    make_plots(summary, best, pd.DataFrame(contingency), out, seeds, purge_mode)
    write_report(best, val, config, out)
    return summary


def make_plots(summary, best, contingency, out, seeds, purge_mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    colors = ['#2563eb', '#d97706', '#059669', '#dc2626', '#7c3aed']
    for app, color in zip(SERVICES, colors):
        v = summary[(summary.application == app) & (summary.partition == 'validation')]
        for ax, metric, title in zip(axes, ['ari', 'mapped_balanced_accuracy'],
                                      ['Agreement with congestion level (ARI)', 'Balanced accuracy (training-only cluster mapping)']):
            x, mean, sd = v.window.to_numpy(), v[f'{metric}_mean'].to_numpy(), v[f'{metric}_std'].fillna(0).to_numpy()
            ax.plot(x, mean, 'o-', label=app, color=color, linewidth=2)
            ax.fill_between(x, mean-sd, mean+sd, alpha=.12, color=color)
            ax.set_xscale('log'); ax.set_xticks(x, labels=x); ax.set_xlabel('Trailing window (records)')
            ax.set_title(title, fontsize=12); ax.grid(alpha=.15)
    axes[0].axhline(0, ls='--', color='gray', linewidth=1)
    axes[1].axhline(1/3, ls='--', color='gray', linewidth=1)
    axes[0].set_ylim(-.05, 1.05); axes[1].set_ylim(0, 1.05)
    axes[0].legend(loc='best', fontsize=9)
    protocol = 'flow overlap allowed' if purge_mode == 'none' else 'five-tuple purged'
    fig.suptitle(f'Three-level clustering within each application — {protocol}\nTemporal validation: mean ± 1 seed SD; identical rows across windows', fontsize=13)
    fig.savefig(out/'window_comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), layout='constrained')
    seed = 42 if 42 in seeds else seeds[0]
    for ax, app in zip(axes, SERVICES):
        w = int(best.set_index('application').loc[app, 'window'])
        t = contingency[(contingency.application == app) & (contingency.window == w) &
                        (contingency.seed == seed) & (contingency.partition == 'validation')]
        matrix = t.pivot_table(index='level', columns='training_mapped_level', values='count', aggfunc='sum').reindex(index=LEVELS, columns=LEVELS).fillna(0).to_numpy()
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center', color='white' if matrix[i,j]>.55 else '#172554')
        ax.set_xticks(range(3), LEVELS, rotation=45, ha='right');ax.set_yticks(range(3), LEVELS)
        ax.set_title(f'{app} · W={w}'); ax.set_xlabel('Training-mapped cluster')
    axes[0].set_ylabel('Collected congestion level')
    fig.suptitle(f'Validation-selected windows · representative seed {seed} · row-normalized\nSelection is exploratory; reserved tail was not evaluated', fontsize=13)
    fig.savefig(out/'best_window_contingency.png', dpi=180); plt.close(fig)


def write_report(best, val, config, out):
    text = ['# Per-application congestion clustering: window sweep', '',
            f"Source dataset commit: `{config['source_commit']}`.",
            f"Protocol: **{config['purge_mode']}**. {config['service_records']:,} service records from {config['raw_records']:,} raw records, 15 captures.", '',
            'Exploratory diagnostic: true application identity defines five separate analyses. Three levels are present in training and validation for each application.', '',
            '## Mean validation ARI by window', '',
            '| Window | HTTP | SFTP | SMTP | SSH | Video |', '|---|---:|---:|---:|---:|---:|']
    for w in config['windows']:
        values = val[val.window == w].set_index('application')
        text.append('| '+str(w)+' | '+' | '.join(f"{values.loc[a,'ari_mean']:.4f}" for a in SERVICES)+' |')
    text += ['', '## Exploratory best windows', '',
             '| Application | Window | W=3 ARI | Selected ARI ± seed SD | AMI | Mapped balanced accuracy |',
             '|---|---:|---:|---:|---:|---:|']
    for r in best.itertuples():
        baseline = val[(val.application == r.application) & (val.window == 3)]
        baseline_text = f'{baseline.ari_mean.iloc[0]:.4f}' if len(baseline) else 'not run'
        text.append(f'| {r.application} | {r.window} | {baseline_text} | {r.ari_mean:.4f} ± {r.ari_std:.4f} | {r.ami_mean:.4f} | {r.mapped_balanced_accuracy_mean:.4f} |')
    text += ['', '![Window comparison](window_comparison.png)', '', '![Contingency](best_window_contingency.png)', '',
             '## Protocol', '',
             '- First 60% / next 20% / final 20% of each timestamp-sorted capture; final tail not evaluated.',
             '- '+config['purge'],
             f"- Common endpoints require {max(config['windows'])} consecutive eligible rows. Every W uses exactly the same training/validation endpoints.",
             '- Five statistics (mean, max, median, min, population std) for TcpRtt, SynAck, AckDat. Zeros retained; invalid values break windows.',
             '- StandardScaler and MBK fit on training only. No application/congestion labels are model inputs.',
             '- Primary metric: ARI. AMI and silhouette also reported. Balanced accuracy uses a one-to-one mapping frozen from training only.',
             '- Best W maximizes mean validation ARI; the reserved partition is not used to verify that selection.',
             '- Hyperparameters, SHA-256 hashes, sample counts, exclusions and window spans are in companion CSV/JSON files.', '',
             '## Limits', '']
    text += ['- '+s for s in config['limitations']]
    text += ['', 'Full instructions: [WINDOW_SWEEP.md](../../../WINDOW_SWEEP.md).', '']
    (out/'REPORT.md').write_text('\n'.join(text), encoding='utf-8')


def audit_eligibility(data_dir, output, windows=DEFAULT_WINDOWS):
    """Preflight only: quantify why long purged windows may be unavailable."""
    rows = []
    for app in SERVICES:
        for level in LEVELS:
            d, _ = load_capture(Path(data_dir)/f'{app}_{level}.flow', app, level)
            for w in windows:
                _, audit = prepare_partitions(d, w, 'five-tuple')
                rows.append({'application': app, 'level': level, 'window': w, **audit})
    result = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    p.add_argument('--data-dir', type=Path, default=root/'DATASETS/CDR-MLC/New_Version')
    p.add_argument('--output', type=Path, default=root/'outputs/window_sweep')
    p.add_argument('--windows', nargs='+', type=int, default=DEFAULT_WINDOWS)
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--silhouette-samples', type=int, default=1000)
    p.add_argument('--purge-mode', choices=['none', 'five-tuple'], default='none',
                   help='none: dependent within-capture diagnostic; five-tuple: conservative flow-disjoint check')
    p.add_argument('--audit-eligibility-only', action='store_true', help='Write purged endpoint counts, without fitting models')
    p.add_argument('--source-commit', default=None, help='Input data revision, recorded for provenance')
    args = p.parse_args()
    if args.audit_eligibility_only:
        audit_eligibility(args.data_dir, args.output/'purged_eligibility_all_windows.csv', args.windows)
    else:
        run(args.data_dir, args.output, args.windows, args.seeds, args.silhouette_samples, args.source_commit, args.purge_mode)


if __name__ == '__main__':
    main()
