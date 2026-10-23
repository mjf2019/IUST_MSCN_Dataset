"""Observer-only paired endpoint metrics; never exposes observer labels to fitting."""
import numpy as np

VERSION = 1
METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')


def classification(y, prediction, classes=range(6)):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or y.ndim != 1:
        raise ValueError('Endpoint predictions and truth must be aligned vectors.')
    labels = [int(c) for c in classes]
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    index = {int(c): i for i, c in enumerate(labels)}
    for t, p in zip(y, prediction):
        if int(t) not in index or int(p) not in index:
            raise ValueError('Endpoint class outside declared source vocabulary.')
        cm[index[int(t)], index[int(p)]] += 1
    if not len(y):
        return dict(samples=0, classes=labels, confusion_matrix=cm.tolist(),
                    **{k: None for k in METRICS})
    tp = np.diag(cm).astype(float)
    actual, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2*tp, actual+predicted, out=np.zeros_like(tp), where=(actual+predicted) != 0)
    return dict(samples=len(y), classes=labels, confusion_matrix=cm.tolist(),
                accuracy=float(tp.sum()/len(y)), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()))


def paired(before, after, y, mask, classes=range(6)):
    before, after, y = np.asarray(before), np.asarray(after), np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    if before.shape != y.shape or after.shape != y.shape or mask.shape != y.shape:
        raise ValueError('Paired endpoints must refer to exactly the same rows.')
    first = classification(y[mask], before[mask], classes)
    last = classification(y[mask], after[mask], classes)
    gain = {k: None if first[k] is None else 100*(last[k]-first[k]) for k in METRICS}
    drop = {k: None if gain[k] is None else -gain[k] for k in METRICS}
    return dict(version=VERSION, initial=first, final=last, gain_pp=gain, drop_pp=drop,
                scope='same masked rows at both endpoints; observer-only; macro scores use source class vocabulary')


def install_baseline(module, classes=range(6)):
    original = module.summary_pair
    if getattr(original, '_endpoint_metrics', False): return
    def summary(before, after, y, mask):
        result = original(before, after, y, mask)
        result['metric_pairs'] = paired(before, after, y, mask, classes)
        return result
    summary._endpoint_metrics = True
    module.summary_pair = summary


def install_amcal(legacy, classes=range(6)):
    original = legacy.retention_summary
    if getattr(original, '_endpoint_metrics', False): return
    def summary(before, after, y, mask):
        result = original(before, after, y, mask)
        result['metric_pairs'] = paired(before[0], after[0], y, mask, classes)
        return result
    summary._endpoint_metrics = True
    legacy.retention_summary = summary
