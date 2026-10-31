"""User-run exact schedule, causal update and reset checks; no dataset needed."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_periodic as runner
from test_two_action_ddqn import Fixture


class PeriodicQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.engine.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def fixture(self):
        fixture = Fixture(); fixture.a.dataset = 'IAT'
        return fixture

    def test_one_skip_and_two_skips_start_at_first_row_and_stop_at_twenty_percent(self):
        for skip, last in [(1, 119), (2, 178)]:
            schedule = runner.PeriodicQuery(skip); used = 0; positions = []
            for step in range(1, 301):
                action, q = schedule.choose(object(), used < 60, 0., object())
                self.assertIsNone(q)
                if action: positions.append(step); used += 1
            self.assertEqual(positions, list(range(1, 301, skip+1))[:60])
            self.assertEqual(positions[-1], last)
            self.assertEqual(schedule.queries, 60); self.assertEqual(schedule.position, 300)
            self.assertEqual(schedule.steps, 0); self.assertEqual(schedule.updates, 0)

    def test_invalid_stride_exploration_and_query_learning_are_rejected(self):
        for skip in (0, -1, 3):
            with self.assertRaises(ValueError): runner.PeriodicQuery(skip)
        schedule = runner.PeriodicQuery(1)
        with self.assertRaises(ValueError): schedule.choose(None, True, .1, None)
        with self.assertRaises(RuntimeError): schedule.remember(None)
        action, _ = schedule.choose(None, False, 0., None)
        self.assertEqual(action, 0)

    def test_real_update_environment_uses_the_exact_periodic_mask_without_query_learning(self):
        fixture = self.fixture()
        for skip in (1, 2):
            schedule = runner.PeriodicQuery(skip)
            result, trace = runner.engine.run_episode(fixture, schedule, fixture.domain, fixture.reference,
                42, training=False, mode='DDQN')
            positions = runner.validate_periodic_trace(trace, skip, result['cap'])
            self.assertEqual(result['labels'], int(len(fixture.domain[1])*.2))
            self.assertEqual(positions, list(range(1, len(trace)+1, skip+1))[:result['cap']])
            self.assertEqual(result['td_updates'], 0)
            bad = [dict(row) for row in trace]; bad[0]['queried'] = False
            with self.assertRaises(RuntimeError): runner.validate_periodic_trace(bad, skip, result['cap'])

    def test_other_control_cannot_contaminate_context_memory_or_order(self):
        fixture = self.fixture()
        cnn_hash = runner.engine.weight_hash(fixture.cnn.state_dict())
        initial_hash = runner.engine.weight_hash(fixture.template.state_dict())
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            roots = [Path(temporary)/str(i) for i in range(3)]
            first = runner.evaluate_level(fixture, 1, 5, fixture.domain, fixture.reference, roots[0], dict(test=True))
            runner.evaluate_level(fixture, 2, 5, fixture.domain, fixture.reference, roots[1], dict(test=True))
            repeated = runner.evaluate_level(fixture, 1, 5, fixture.domain, fixture.reference, roots[2], dict(test=True))
            self.assertEqual(first, repeated)
            self.assertEqual((roots[0]/'trace.csv').read_bytes(), (roots[2]/'trace.csv').read_bytes())
        self.assertEqual(cnn_hash, runner.engine.weight_hash(fixture.cnn.state_dict()))
        self.assertEqual(initial_hash, runner.engine.weight_hash(fixture.template.state_dict()))

    def test_console_and_saved_metadata_distinguish_controls_and_reuse_completed_results(self):
        fixture = self.fixture(); output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            root = Path(temporary)/'case'
            expected = runner.evaluate_level(fixture, 2, 7, fixture.domain, fixture.reference, root,
                dict(test=True), 8, 14)
            metrics = json.loads((root/'metrics.json').read_text())
            self.assertIn('disabled', metrics['Q_learning'])
            self.assertEqual(metrics['query_control'], 'PeriodicSkip2')
            with patch.object(runner.engine, 'run_episode', side_effect=AssertionError('Unexpected rerun')):
                actual = runner.evaluate_level(fixture, 2, 7, fixture.domain, fixture.reference, root,
                    dict(test=True), 8, 14)
                self.assertEqual(actual, expected)
            metrics['labels'] = 0; (root/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.evaluate_level(fixture, 2, 7, fixture.domain, fixture.reference, root, dict(test=True))
        for token in ('DATASET=IAT', 'METHOD=PeriodicSkip2', 'BUDGET=20%', 'LEVEL=7',
                'PATTERN=query1/skip2', 'LAST_QUERY_STEP=', 'DONE', 'REUSED'):
            self.assertIn(token, output.getvalue())


if __name__ == '__main__': unittest.main()
