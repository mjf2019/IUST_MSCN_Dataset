"""Replay paid query positions to compare Context against last-label-per-flow.

Reads existing legacy evaluation traces. No models, GPU, or additional labels.
The class-as-flow proxy must be explicitly recorded by the source diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def score_trace(frame):
    last_label = {}
    truth, baseline, model = [], [], []
    queried = 0
    fallback = 0
    for expected_step, row in enumerate(frame.to_dict('records'), 1):
        if int(row['step']) != expected_step:
            raise ValueError('Trace must be complete and in original stream order.')
        # ID is permitted by the experiment; its numeric value is never used
        # as a prediction until a paid label has actually been observed.
        flow_id = int(row['true_label'])
        label = int(row['true_label'])
        if flow_id in last_label:
            prediction = last_label[flow_id]
        else:
            prediction = int(row['base_prediction'])
            fallback += 1
        baseline.append(prediction)
        truth.append(label)
        model.append(int(row['prediction_before_update']))
        paid = str(row['updated']).lower()
        if paid not in ('true', 'false', '1', '0'):
            raise ValueError('Invalid updated flag.')
        if paid in ('true', '1'):
            queried += 1
            last_label[flow_id] = label  # Only AFTER predicting this sample.
        if int(row['labels_used']) != queried:
            raise ValueError('Paid label count does not match trace.')
    if not truth:
        raise ValueError('Empty trace.')
    def metrics(pred):
        return dict(accuracy=float(accuracy_score(truth, pred)),
                    macro_f1=float(f1_score(truth, pred, average='macro', zero_division=0)),
                    weighted_f1=float(f1_score(truth, pred, average='weighted', zero_division=0)))
    a, b = metrics(model), metrics(baseline)
    return dict(samples=len(truth), labels_used=queried, flows_with_paid_label=len(last_label),
                cnn_fallback_samples=fallback, amcal_before_feedback=a,
                last_label_before_feedback=b,
                accuracy_delta_amcal_minus_baseline=a['accuracy']-b['accuracy']), baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Legacy result directory, or a run directory to find its latest matching evaluation.')
    args = parser.parse_args()
    directory = args.results
    if not list(directory.glob('legacy_PER_*_trace.csv')):
        candidates = [p for p in (directory / 'Results').glob('*')
                      if p.is_dir() and 'trainmetadata' in p.name
                      and 'class_context_proxy' in p.name
                      and '_budget_per_class' not in p.name
                      and list(p.glob('legacy_PER_*_trace.csv'))]
        if not candidates:
            parser.error('No global-budget training-metadata proxy traces found; pass the exact result directory.')
        directory = max(candidates, key=lambda p: max(f.stat().st_mtime for f in p.glob('legacy_PER_*_trace.csv')))
    print('Source evaluation:', directory)
    reports = []
    paths = sorted(directory.glob('legacy_PER_*_trace.csv'), key=lambda p: int(p.name.split('_')[2]))
    for path in paths:
        level = int(path.name.split('_')[2])
        diagnostic_path = directory / f'legacy_PER_{level}_diagnostics.json'
        diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
        if not diagnostic.get('oracle_context_routing'):
            raise ValueError('This baseline requires recorded class-as-flow proxy evaluation.')
        frame = pd.read_csv(path)
        report, predictions = score_trace(frame)
        if report['labels_used'] != diagnostic['labels_used']:
            raise ValueError('Diagnostic and trace label counts differ.')
        report.update(perturbation_level=level,
                      source_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      budget_mode=diagnostic['budget_mode'],
                      protocol='same RL-paid query positions; CNN until first paid flow label; predict before feedback; observer only')
        reports.append(report)
        pd.DataFrame(dict(step=frame['step'], source_row=frame['source_row'],
                          baseline_prediction=predictions)).to_csv(directory / f'flow_label_baseline_PER_{level}_predictions.csv', index=False)
        print(f"PER {level:2d}: AMCAL={100*report['amcal_before_feedback']['accuracy']:.2f}% | "
              f"last-label={100*report['last_label_before_feedback']['accuracy']:.2f}% | "
              f"delta={100*report['accuracy_delta_amcal_minus_baseline']:+.2f} pp | labels={report['labels_used']}")
    mean = {name: float(np.mean([r[name]['accuracy'] for r in reports]))
            for name in ('amcal_before_feedback', 'last_label_before_feedback')}
    print(f"Mean accuracy: AMCAL={100*mean['amcal_before_feedback']:.2f}% | last-label={100*mean['last_label_before_feedback']:.2f}%")
    output = directory / 'flow_label_baseline_summary.json'
    output.write_text(json.dumps(dict(levels=reports, mean_accuracy=mean), indent=2), encoding='utf-8')
    print('Report:', output)


if __name__ == '__main__':
    main()
