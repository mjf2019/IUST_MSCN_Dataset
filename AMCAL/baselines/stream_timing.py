"""Synchronized wall-time accounting for causal stream evaluation only."""
from contextlib import contextmanager
import time
import torch


class StreamTimer:
    def __init__(self, device, enabled=True):
        self.device = torch.device(device)
        self.enabled = enabled
        self.seconds = dict(cache=0., prediction=0., selection=0., update=0.)
        self.wall_start = None
        self.wall_seconds = 0.

    def synchronize(self):
        if self.enabled and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] += time.perf_counter() - start

    def start(self):
        if self.enabled:
            self.synchronize()
            self.wall_start = time.perf_counter()

    def stop(self):
        if self.enabled:
            self.synchronize()
            self.wall_seconds += time.perf_counter() - self.wall_start

    def metrics(self, samples, labels):
        if not self.enabled:
            return {}
        cache = self.seconds['cache']
        total = self.wall_seconds + cache
        prediction = self.seconds['prediction'] + cache
        selection, update = self.seconds['selection'], self.seconds['update']
        return dict(online_seconds=total, prediction_seconds=prediction,
                    selection_seconds=selection, update_seconds=update,
                    cache_seconds=cache,
                    other_seconds=max(0., total-prediction-selection-update),
                    online_ms_per_sample=1000*total/samples if samples else 0.,
                    samples_per_second=samples/total if total else 0.,
                    update_ms_per_label=1000*update/labels if labels else 0.,
                    timing_device=str(self.device),
                    timing_protocol='synchronized wall time; prediction includes frozen-cache construction; '
                    'update includes Context/classifier, reward probes and selector learning; excludes model loading, '
                    'initialization, endpoint retention observers, file I/O and human label latency; '
                    'other includes stream bookkeeping, oracle scoring and progress output')


def timing_row(name, level, metrics):
    return dict(Method=name, PER=level, **{
        'Pred(s)': metrics['prediction_seconds'], 'Select(s)': metrics['selection_seconds'],
        'Update(s)': metrics['update_seconds'], 'Other(s)': metrics['other_seconds'],
        'Online(s)': metrics['online_seconds'], 'Cache(s)': metrics['cache_seconds'],
        'ms/row': metrics['online_ms_per_sample'], 'rows/s': metrics['samples_per_second'],
        'ms/label': metrics['update_ms_per_label']})


def timing_report(metrics):
    return (f'  Online time: {metrics["online_seconds"]:.3f}s | prediction '
            f'{metrics["prediction_seconds"]:.3f}s | selection {metrics["selection_seconds"]:.3f}s | '
            f'update {metrics["update_seconds"]:.3f}s | {metrics["online_ms_per_sample"]:.3f} ms/row')
