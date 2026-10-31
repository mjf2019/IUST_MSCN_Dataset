"""User-run budget, reset, provenance and caching checks; no dataset required."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_budget_sweep as runner
from test_two_action_ddqn import Fixture


class BudgetSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.engine.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def fixture(self):
        fixture = Fixture(); fixture.a.dataset = 'IAT'; fixture.threshold = 0.
        agent = fixture.learner()
        with torch.no_grad():
            for p in agent.policy.parameters(): p.zero_()
            agent.policy.network[-1].bias.copy_(torch.tensor([0., 1.]))
        agent.target.load_state_dict(agent.policy.state_dict())
        return fixture, agent

    def paper_checkpoint(self):
        saved = dict(selected_epoch=2, protocol=dict(candidate='paper', utility=None,
            protocol=dict(budget_fraction=.2, context_lr=.00025, replay_weight=.1)))
        return saved, dict(selected_epoch=2, protocol=deepcopy(saved['protocol']))

    def test_only_requested_no_drift_paper_checkpoint_is_accepted(self):
        saved, completion = self.paper_checkpoint()
        self.assertEqual(runner.check_paper_protocol(saved, completion)['budget_fraction'], .2)
        for mutation in ('utility', 'synthetic', 'budget', 'epoch'):
            model, done = self.paper_checkpoint()
            if mutation == 'utility': model['protocol']['utility'] = dict(cost=0.)
            elif mutation == 'synthetic': model['protocol']['protocol']['synthetic'] = dict(seed=42)
            elif mutation == 'budget': model['protocol']['protocol']['budget_fraction'] = .8
            else: model['selected_epoch'] = 3
            if mutation != 'epoch': done['protocol'] = deepcopy(model['protocol'])
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError): runner.check_paper_protocol(model, done)

    def test_all_six_caps_apply_to_both_policies_without_query_training(self):
        fixture, agent = self.fixture()
        weights = runner.engine.weight_hash(agent.policy.state_dict())
        cnn = runner.engine.weight_hash(fixture.cnn.state_dict())
        initial_context = runner.engine.weight_hash(fixture.template.state_dict())
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            for budget in runner.BUDGETS:
                for method in ('DDQN', 'Uncertainty'):
                    root = Path(temporary)/str(budget)/method
                    row = runner.evaluate_case(fixture, agent, method, budget, 5,
                        fixture.domain, fixture.reference, root, dict(test=True))
                    result = json.loads((root/'metrics.json').read_text())
                    self.assertEqual(result['cap'], int(len(fixture.domain[1])*budget/100))
                    self.assertEqual(result['labels'], result['cap'])
                    self.assertEqual(row['budget_percent'], budget)
                    self.assertEqual(row['method'], runner.METHOD_NAMES[method])
                    self.assertEqual(result['td_updates'], 0)
        self.assertEqual(agent.steps, 0); self.assertEqual(agent.updates, 0)
        self.assertEqual(weights, runner.engine.weight_hash(agent.policy.state_dict()))
        self.assertEqual(cnn, runner.engine.weight_hash(fixture.cnn.state_dict()))
        self.assertEqual(initial_context, runner.engine.weight_hash(fixture.template.state_dict()))

    def test_previous_budget_cannot_contaminate_next_context_or_memory(self):
        fixture, agent = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            roots = [Path(temporary)/str(i) for i in range(3)]
            first = runner.evaluate_case(fixture, agent, 'DDQN', 30, 5,
                fixture.domain, fixture.reference, roots[0], dict(test=True))
            runner.evaluate_case(fixture, agent, 'DDQN', 80, 5,
                fixture.domain, fixture.reference, roots[1], dict(test=True))
            repeated = runner.evaluate_case(fixture, agent, 'DDQN', 30, 5,
                fixture.domain, fixture.reference, roots[2], dict(test=True))
            self.assertEqual(first, repeated)
            self.assertEqual((roots[0]/'trace.csv').read_bytes(), (roots[2]/'trace.csv').read_bytes())

    def test_console_identifies_dataset_method_budget_level_and_actual_consumption(self):
        fixture, agent = self.fixture(); output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            runner.evaluate_case(fixture, agent, 'Uncertainty', 40, 7,
                fixture.domain, fixture.reference, Path(temporary)/'case', dict(test=True), 3, 84)
        text = output.getvalue()
        for token in ('[3/84]', 'DATASET=IAT', 'METHOD=Uncertainty', 'TEST_BUDGET=40%',
                'LEVEL=7', 'START', 'DONE', 'LABELS=', 'FINAL_TARGET_F1=', 'ORIGINAL_DOMAIN_FORGETTING='):
            self.assertIn(token, text)

    def test_verified_completed_case_is_reused_and_tampering_is_rejected(self):
        fixture, agent = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)/'case'
            expected = runner.evaluate_case(fixture, agent, 'DDQN', 30, 5,
                fixture.domain, fixture.reference, root, dict(test=True))
            with patch.object(runner.engine, 'run_episode', side_effect=AssertionError('Unexpected rerun')):
                reused = runner.evaluate_case(fixture, agent, 'DDQN', 30, 5,
                    fixture.domain, fixture.reference, root, dict(test=True))
                self.assertEqual(expected, reused)
            path = root/'metrics.json'; changed = json.loads(path.read_text()); changed['labels'] = 0
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.evaluate_case(fixture, agent, 'DDQN', 30, 5,
                    fixture.domain, fixture.reference, root, dict(test=True))


if __name__ == '__main__': unittest.main()
