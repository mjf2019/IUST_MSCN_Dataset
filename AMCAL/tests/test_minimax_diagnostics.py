"""Diagnostic arithmetic and optional Torch-only audit smoke tests."""
import csv
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('minimax_diagnostics_std', ROOT/'enhancements'/'minimax_diagnostics.py')
diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)


def row(step, score, probability=.5):
    return dict(step=step, score=score, gap=score+.05, weighted_gain=0., weighted_harm=0.,
        request_cost=.05, query_probability=probability, actor_probability_after=probability,
        actor_logit_before=0., actor_grad_norm=.1, query_legal=True,
        queried=step % 2 == 0, target_status='measured')


class DiagnosticArithmeticTests(unittest.TestCase):
    def test_zero_values_are_retained_in_distributions(self):
        result = diag.distribution([None, '', 0, 1, -1])
        self.assertEqual(result['n'], 3); self.assertEqual(result['mean'], 0)
        self.assertEqual(result['p50'], 0)

    def test_target_signs_and_components(self):
        result = diag.summarize_targets([row(1, .2, .001), row(2, -.3, .01), row(3, 0)])
        self.assertEqual(result['positive'], 1); self.assertEqual(result['negative'], 1)
        self.assertEqual(result['zero'], 1)
        self.assertEqual(result['positive_targets_with_p_below_0p01'], 1)
        self.assertAlmostEqual(result['positive_percent'], 100/3)

    def test_inconsistent_or_illegal_target_is_rejected(self):
        for key, value in [('score', 10), ('query_legal', False)]:
            sample = row(1, .2); sample[key] = value
            with self.assertRaises(ValueError): diag.summarize_targets([sample])

    def test_balanced_accuracy_and_tie_auc(self):
        result = diag.binary_report([1, -1, -1, -1], [.5]*4)
        self.assertEqual(result['balanced_accuracy'], .5); self.assertEqual(result['auc'], .5)
        self.assertEqual(diag.binary_report([1, -1], [.8, .2])['auc'], 1)
        self.assertIsNone(diag.binary_report([-1], [.1])['auc'])

    def test_routing_mistakes_and_mapping_are_distinct(self):
        rows = [dict(route=1, truth=0, prediction=1, frozen_prediction=0, context_prediction=1,
            frozen_route_score=0., context_route_score=1.)]
        result = diag.routing_report(rows)
        self.assertEqual(result['mapping_errors'], 0)
        self.assertEqual(result['missed_correct_route'], 1)
        self.assertEqual(result['informative_selection_accuracy'], 0)
        rows[0]['prediction'] = 0
        self.assertEqual(diag.routing_report(rows)['mapping_errors'], 1)

    def test_recorder_writes_each_row_and_keeps_sign_prevalence_separate(self):
        class State:
            def detach(self): return self
            def cpu(self): return self
            def reshape(self, *args): return self
            def clone(self): return self
        with tempfile.TemporaryDirectory() as root:
            recorder = diag.TrainingRecorder(Path(root), 42, 2, {})
            recorder.save_cache = lambda: None
            recorder.start_stage(1, 0)
            for i, score in enumerate([1., -1., -1., -1.], 1): recorder.observe(State(), row(i, score))
            with contextlib.redirect_stdout(io.StringIO()):
                recorder.finish_stage(); recorder.finalize()
            with (Path(root)/'train_epoch1_level0.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            self.assertEqual(recorder.seen['negative'], 3)
            self.assertEqual(len(recorder.samples['negative']), 1)
            self.assertEqual(recorder.stages[0]['positive_percent'], 25.)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires the existing PyTorch experiment environment')
class ActorAuditTests(unittest.TestCase):
    def test_actual_gradient_contract_and_tiny_signed_fit(self):
        sys.path.insert(0, str(ROOT))
        try:
            from enhancements.minimax_diagnostics import actor_audit
            from enhancements.minimax_query import QueryMaximizer
            torch.manual_seed(42)
            initial = QueryMaximizer(2)
            samples = []
            for sign in (1., -1.):
                for i in range(16): samples.append(dict(state=torch.tensor([sign, i/16]), score=sign))
            with tempfile.TemporaryDirectory() as root:
                path = Path(root)/'cache.pt'
                torch.save(dict(version=diag.DIAGNOSTIC_VERSION, samples=samples, initial_query=initial.state_dict(),
                    seen={'positive':16, 'negative':16}, scope='synthetic TRAIN capacity test'), path)
                report = actor_audit(path, Path(root)/'report.json', 42, 100, .01, 'cpu')
                self.assertTrue(report['gradient_contract']['sign_ok'])
                self.assertTrue(report['gradient_contract']['target_detached'])
                self.assertGreater(report['original_max_objective']['final']['balanced_accuracy'], .9)
                self.assertGreater(report['bce_sign_diagnostic_only']['final']['balanced_accuracy'], .9)
        finally: sys.path.pop(0)

    def test_one_sided_targets_do_not_claim_failed_learning(self):
        sys.path.insert(0, str(ROOT))
        try:
            from enhancements.minimax_diagnostics import actor_audit
            from enhancements.minimax_query import QueryMaximizer
            with tempfile.TemporaryDirectory() as root:
                path = Path(root)/'cache.pt'
                torch.save(dict(version=diag.DIAGNOSTIC_VERSION, samples=[dict(state=torch.zeros(2), score=-1.)],
                    initial_query=QueryMaximizer(2).state_dict(), seen={'negative':1}, scope='TRAIN'), path)
                report = actor_audit(path, Path(root)/'report.json', 42, 10, .001, 'cpu')
                self.assertEqual(report['status'], 'insufficient_sign_diversity')
                self.assertNotIn('bce_sign_diagnostic_only', report)
        finally: sys.path.pop(0)


if __name__ == '__main__': unittest.main()
