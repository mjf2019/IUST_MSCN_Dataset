"""CDR-MLC: paper Algorithms 1/2, with label-free routing and causal TTFE.

Unspecified paper choices (scaling, warm-up, ddof, MBK parameters) are explicit.
See README.md for the distinction between algorithm fidelity and reproduction.
"""
from collections import deque
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report,
                             confusion_matrix, normalized_mutual_info_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

SENSITIVE = ('AckDat', 'TcpRtt', 'SynAck')
STATS = ('max', 'min', 'median', 'mean', 'std')
TREND_COLUMNS = tuple(f'{f}_{s}' for f in SENSITIVE for s in STATS)
# 36 numeric columns in repository CSVs minus IdleTime = 35 paper inputs.
BASE_FEATURES = (
    'pRetran', 'SrcRetra', 'PCRatio', 'SrcWin', 'SrcLoss', 'DstRate',
    'SrcLoad', 'Load', 'DstLoad', 'TcpRtt', 'Sum', 'AckDat', 'dTtl',
    'Min', 'pLoss', 'DstLoss', 'Loss', 'StdDev', 'Rate', 'SrcRate',
    'Dur', 'SrcPkts', 'SrcGap', 'DstBytes', 'DstGap', 'sTtl', 'DstWin',
    'TotPkts', 'DstPkts', 'Mean', 'SrcBytes', 'TotBytes', 'dMeanPktSz',
    'DstRetra', 'SynAck',
)


@dataclass(frozen=True)
class Config:
    window_size: int = 3
    n_clusters: int = 3
    n_estimators: int = 20
    random_state: int = 42
    batch_size: int = 1024
    n_init: int = 10
    max_iter: int = 100
    scale_trends: bool = True
    std_ddof: int = 1
    n_jobs: int = -1

    def __post_init__(self):
        for name in ('window_size', 'n_clusters', 'n_estimators', 'batch_size', 'n_init', 'max_iter'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.std_ddof not in (0, 1):
            raise ValueError('std_ddof must be 0 or 1')


def numeric_frame(df, columns):
    if not isinstance(df, pd.DataFrame) or df.columns.duplicated().any():
        raise ValueError('Input must be a DataFrame with unique column names')
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required features: {missing}')
    try:
        out = df.loc[:, list(columns)].apply(pd.to_numeric, errors='raise').astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Model features must be numeric') from exc
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Features contain NaN/Inf; clean upstream without using test statistics')
    return out


def stream_keys(df, stream_column):
    if stream_column is None:
        return np.zeros(len(df), dtype=int)
    if stream_column not in df or df[stream_column].isna().any():
        raise ValueError(f'Missing/invalid stream column: {stream_column}')
    return df[stream_column].to_numpy()


def trend_features(df, config=None, stream_column=None):
    """15 trailing-window features, current row included; no future rows.

    Start with partial windows, std=0 for one row. Reset on each contiguous
    capture boundary. Never group or reset using class/congestion labels.
    Input order is authoritative; caller must supply chronological captures.
    """
    config = config or Config()
    raw = numeric_frame(df, SENSITIVE).reset_index(drop=True)
    keys = stream_keys(df, stream_column)
    if not len(raw):
        return pd.DataFrame(columns=TREND_COLUMNS, index=df.index, dtype=float)
    boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(raw)]
    parts = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        roll = raw.iloc[start:end].rolling(config.window_size, min_periods=1)
        measures = {'max': roll.max(), 'min': roll.min(), 'median': roll.median(),
                    'mean': roll.mean(), 'std': roll.std(ddof=config.std_ddof).fillna(0)}
        parts.append(pd.DataFrame({f'{f}_{s}': measures[s][f] for f in SENSITIVE for s in STATS}))
    out = pd.concat(parts)
    out.index = df.index
    return out


class CDRMLC:
    """One MBK router and one RF per geometric cluster. No label at inference."""
    def __init__(self, config=None, feature_columns=None, stream_column=None):
        self.config = config or Config()
        self.feature_columns = tuple(BASE_FEATURES if feature_columns is None else feature_columns)
        self.stream_column = stream_column
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError('Duplicate feature names')
        if not set(SENSITIVE).issubset(self.feature_columns):
            raise ValueError(f'All three sensitive features are required: {SENSITIVE}')
        forbidden = [c for c in self.feature_columns if self._metadata(c)]
        if forbidden or (stream_column is not None and stream_column in self.feature_columns):
            raise ValueError(f'Metadata/label columns cannot be model features: {forbidden}')
        if stream_column and self._label_metadata(stream_column):
            raise ValueError('Stream boundaries must not be derived from class or congestion labels')
        self.classification_features = tuple(c for c in self.feature_columns if c not in SENSITIVE)
        if not self.classification_features:
            raise ValueError('At least one classification feature is required')

    @staticmethod
    def _label_metadata(name):
        name = name.lower()
        return any(token in name for token in ('label', 'target', 'class', 'congestion', 'cluster', 'level'))

    @classmethod
    def _metadata(cls, name):
        return cls._label_metadata(name) or name.lower().startswith(('index', 'unnamed:')) or name in (
            'IdleTime', 'capture_id', 'stream_id', 'timestamp', 'StartTime', 'LastTime', 'level')

    def fit(self, df, target_column='label'):
        if target_column in self.feature_columns or self.stream_column == target_column:
            raise ValueError('Target cannot be a feature or stream identifier')
        if target_column not in df or df[target_column].isna().any():
            raise ValueError(f'Missing/invalid target column: {target_column}')
        if len(df) < self.config.n_clusters:
            raise ValueError('Fewer training rows than clusters')
        numeric_frame(df, self.feature_columns)
        y = df[target_column].to_numpy()
        trends = trend_features(df, self.config, self.stream_column)
        self.scaler_ = StandardScaler() if self.config.scale_trends else None
        z = self.scaler_.fit_transform(trends) if self.scaler_ is not None else trends.to_numpy()
        if np.unique(z, axis=0).shape[0] < self.config.n_clusters:
            raise ValueError('Fewer distinct trend vectors than clusters; cannot train three experts')
        self.router_ = MiniBatchKMeans(
            n_clusters=self.config.n_clusters, random_state=self.config.random_state,
            n_init=self.config.n_init, batch_size=self.config.batch_size,
            max_iter=self.config.max_iter, reassignment_ratio=0.0,
        ).fit(z)
        # EXACTLY the same assignment rule used at inference (no class balancing).
        clusters = self.router_.predict(z)
        if len(np.unique(clusters)) != self.config.n_clusters:
            raise ValueError('MBK has an empty cluster. Inspect data/parameters; no hidden fallback used')
        x = numeric_frame(df, self.classification_features)
        self.experts_ = {}
        self.classes_ = np.unique(y)
        self.training_clusters_ = clusters
        self.target_column_ = target_column
        self.training_cluster_counts_ = np.bincount(clusters, minlength=self.config.n_clusters)
        self.training_class_counts_ = pd.crosstab(
            pd.Series(clusters, name='cluster'), pd.Series(y, name='class'))
        self.missing_classes_ = {}
        for cid in range(self.config.n_clusters):
            mask = clusters == cid
            self.missing_classes_[cid] = sorted(set(self.classes_) - set(y[mask]))
            if self.missing_classes_[cid]:
                warnings.warn(f'Cluster {cid} lacks classes {self.missing_classes_[cid]}; '
                              'no samples were moved across clusters', UserWarning)
            self.experts_[cid] = RandomForestClassifier(
                n_estimators=self.config.n_estimators, random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            ).fit(x.loc[mask], y[mask])
        distances = self.router_.transform(z)
        self.train_distance_q99_ = np.array([
            np.quantile(distances[clusters == cid, cid], .99)
            for cid in range(self.config.n_clusters)])
        return self

    def _check_fitted(self):
        if not hasattr(self, 'experts_'):
            raise ValueError('Call fit before prediction')

    def _transform(self, trends):
        return self.scaler_.transform(trends) if self.scaler_ is not None else trends.to_numpy()

    def _predict_with_trends(self, df, trends):
        self._check_fitted()
        x = numeric_frame(df, self.classification_features)
        if not len(df):
            return np.array([], dtype=self.classes_.dtype), np.array([], dtype=int)
        clusters = self.router_.predict(self._transform(trends))
        predictions = np.empty(len(df), dtype=self.classes_.dtype)
        for cid, expert in self.experts_.items():
            mask = clusters == cid
            if mask.any():
                predictions[mask] = expert.predict(x.loc[mask])
        return predictions, clusters

    def predict_with_clusters(self, df):
        # Stateless batch = fresh independent stream; reset between test captures.
        return self._predict_with_trends(df, trend_features(df, self.config, self.stream_column))

    def predict(self, df):
        return self.predict_with_clusters(df)[0]

    def stream(self):
        self._check_fitted()
        return StreamPredictor(self)

    def diagnostics(self, df, target_column=None, level_column=None,
                    sample_size=2000, compare_agglomerative=False):
        self._check_fitted()
        if sample_size < 2 or not len(df):
            raise ValueError('Diagnostics need nonempty data and sample_size >= 2')
        z = self._transform(trend_features(df, self.config, self.stream_column))
        ids = self.router_.predict(z)
        rng = np.random.default_rng(self.config.random_state)
        take = np.sort(rng.choice(len(z), min(sample_size, len(z)), replace=False))
        zs, cs = z[take], ids[take]
        def silhouette(a, b):
            return float(silhouette_score(a, b)) if 1 < len(np.unique(b)) < len(b) else None
        dist = self.router_.transform(z)
        nearest = dist[np.arange(len(ids)), ids]
        report = {
            'n_samples': len(df), 'cluster_counts': np.bincount(ids, minlength=self.config.n_clusters).tolist(),
            'silhouette_sample_size': len(take), 'silhouette': silhouette(zs, cs),
            'silhouette_scope': 'fixed random sample; MBK labels from fitted training router',
            'distance_above_train_q99_fraction': float(np.mean(nearest > self.train_distance_q99_[ids])),
            'distance_note': 'Heuristic distribution-shift diagnostic, not a calibrated rejection threshold',
            'centers_in_trend_units': (self.scaler_.inverse_transform(self.router_.cluster_centers_)
                                      if self.scaler_ is not None else self.router_.cluster_centers_).tolist(),
            'trend_columns': list(TREND_COLUMNS),
        }
        for col, name in ((target_column, 'class'), (level_column, 'level')):
            if col is not None:
                if col not in df or df[col].isna().any():
                    raise ValueError(f'Missing/invalid diagnostic column: {col}')
                values = df[col].astype(str).to_numpy()
                table = pd.crosstab(pd.Series(ids, name='cluster'), pd.Series(values, name=name))
                report[f'cluster_by_{name}'] = {str(i): {str(k): int(v) for k, v in row.items()}
                                                    for i, row in table.iterrows()}
                report[f'{name}_adjusted_rand'] = float(adjusted_rand_score(values, ids))
                report[f'{name}_normalized_mutual_info'] = float(normalized_mutual_info_score(values, ids))
        if compare_agglomerative and len(take) > self.config.n_clusters:
            ag = AgglomerativeClustering(n_clusters=self.config.n_clusters).fit_predict(zs)
            report['agglomerative_silhouette'] = silhouette(zs, ag)
            report['agglomerative_scope'] = 'Ward clustering fitted on the same sampled rows; diagnostic only'
        return report


class StreamPredictor:
    """Stateful unlabeled inference; feed observations in chronological order."""
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.model.config.window_size)
        self.last_stream = None
        self.started = False

    def predict_one(self, sample):
        frame = pd.DataFrame([sample])
        # Validate before changing stream state.
        numeric_frame(frame, self.model.feature_columns)
        key = stream_keys(frame, self.model.stream_column)[0]
        if self.started and key != self.last_stream:
            self.history.clear()
        self.history.append({f: sample[f] for f in SENSITIVE})
        self.last_stream, self.started = key, True
        trends = trend_features(pd.DataFrame(self.history), self.model.config).iloc[[-1]]
        pred, cluster = self.model._predict_with_trends(frame, trends)
        return {'prediction': pred[0], 'cluster_id': int(cluster[0])}


def evaluate(model, test_df, target_column='label'):
    if target_column not in test_df or test_df[target_column].isna().any():
        raise ValueError(f'Missing/invalid evaluation target: {target_column}')
    # Enforce physically label-free prediction, even during evaluation.
    predictions, clusters = model.predict_with_clusters(test_df.drop(columns=[target_column]))
    true = test_df[target_column].to_numpy()
    labels = np.unique(np.concatenate([model.classes_, true]))
    return {
        'n_samples': len(test_df), 'accuracy': float(accuracy_score(true, predictions)),
        'classification_report': classification_report(true, predictions, labels=labels,
                                                       output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(true, predictions, labels=labels).tolist(),
        'confusion_matrix_labels': labels.tolist(),
        'unseen_test_classes': sorted(set(true) - set(model.classes_)),
    }, predictions, clusters


def universal_clustering_pipeline(df, target_column='label', external_test_df=None,
                                  test_size=.2, random_state=42, n_clusters=3,
                                  window_size=3, clustering_stats=None,
                                  use_at_least_one_rule=False, ensure_label_coverage=False,
                                  preserve_order=True, feature_columns=None,
                                  stream_column=None, **kwargs):
    """Notebook migration wrapper. Unsafe legacy options fail explicitly."""
    if use_at_least_one_rule or ensure_label_coverage or not preserve_order:
        raise ValueError('Oracle routing, label-based cluster reassignment and shuffled splits are disabled')
    if kwargs:
        raise TypeError(f'Unsupported legacy options: {sorted(kwargs)}')
    if clustering_stats is not None and set(clustering_stats) != set(STATS):
        raise ValueError('Paper requires all five statistics (15 trend features)')
    if external_test_df is None:
        if not 0 < test_size < 1:
            raise ValueError('test_size must be between 0 and 1')
        cut = int(len(df) * (1 - test_size))
        if not 0 < cut < len(df):
            raise ValueError('Split leaves an empty train or test set')
        train_df, test_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    else:
        train_df, test_df = df.copy(), external_test_df.copy()
    model = CDRMLC(Config(window_size=window_size, n_clusters=n_clusters,
                         random_state=random_state), feature_columns, stream_column).fit(train_df, target_column)
    metrics, predictions, clusters = evaluate(model, test_df, target_column)
    return {'model': model, 'test_results': metrics, 'predictions': predictions,
            'test_clusters': clusters, 'train_df': train_df, 'test_df': test_df,
            'classifiers': model.experts_, 'kmeans_model': model.router_, 'scaler': model.scaler_,
            'classification_features': list(model.classification_features),
            'window_size': window_size, 'clustering_stats': list(STATS)}


def create_deployment_pipeline(training_results):
    return training_results['model'].stream().predict_one


def run_pipeline_from_two_files(train_file, test_file, **kwargs):
    return universal_clustering_pipeline(pd.read_csv(train_file), external_test_df=pd.read_csv(test_file), **kwargs)


def run_pipeline_from_file(file_path, **kwargs):
    return universal_clustering_pipeline(pd.read_csv(file_path), **kwargs)
