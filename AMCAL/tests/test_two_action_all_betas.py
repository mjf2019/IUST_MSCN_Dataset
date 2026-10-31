"""User-run tests for all-beta coverage, parent reuse and frozen TEST policies."""
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
import amcal_two_action_all_betas as runner
from test_two_action_ddqn import Fixture


class AllBetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.base.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def parent(self):
        return dict(dataset='IAT', seed=42, betas=[0.,10.,100.], budget_fraction=.2,
            context_lr=.00025, replay_weight=.1, pilot_rows=300,
            reward='query: original paper reward + beta * same-domain gain')

    def fixture(self, query=True):
        fixture = Fixture(); fixture.a.dataset = 'IAT'; fixture.a.beta = 0.
        agent = fixture.learner()
        with torch.no_grad():
            for p in agent.policy.parameters(): p.zero_()
            agent.policy.network[-1].bias.copy_(torch.tensor([0.,1.] if query else [1.,0.]))
        agent.target.load_state_dict(agent.policy.state_dict())
        return fixture, agent

    def test_all_five_betas_use_existing_models_or_only_missing_extra_models(self):
        protocol = self.parent(); runner.validate_parent(protocol, 'IAT', 42)
        parent, extra = Path('/parent'), Path('/extra')
        reused, trained = [], []
        for beta in runner.BETAS:
            folder, reuse = runner.candidate_location(parent, extra, protocol, beta)
            (reused if reuse else trained).append(beta)
            self.assertEqual(folder, (parent if reuse else extra)/'Candidates'/runner.base.beta_name(beta))
        self.assertEqual(reused, [0.,10.,100.]); self.assertEqual(trained, [15.,20.])
        self.assertEqual(runner.LEVELS, [5,7,10,12,15,17,20])
        with self.assertRaises(ValueError): runner.candidate_location(parent, extra, protocol, 30.)

    def test_incompatible_parent_experiments_are_rejected(self):
        for key, value in [('dataset','PS'),('seed',7),('budget_fraction',.3),
                ('reward','utility only'),('betas',[0.,10.]),('synthetic',{})]:
            protocol = self.parent(); protocol[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): runner.validate_parent(protocol, 'IAT', 42)

    def test_each_beta_is_tested_even_if_it_makes_zero_queries(self):
        reports = []
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            for beta in runner.BETAS:
                fixture, agent = self.fixture(query=beta != 100.)
                before = runner.base.weight_hash(agent.policy.state_dict())
                row = runner.evaluate_case(fixture, agent, beta, 5, fixture.domain, fixture.reference,
                    Path(temporary)/str(beta), dict(test=True), 'model-sha')
                reports.append(row)
                self.assertEqual(row['labels'], 0 if beta == 100. else int(.2*len(fixture.domain[1])))
                self.assertEqual(before, runner.base.weight_hash(agent.policy.state_dict()))
                self.assertEqual(agent.steps, 0); self.assertEqual(agent.updates, 0)
            summary = runner.summarize(reports)
        self.assertEqual([row['beta'] for row in summary], runner.BETAS)
        self.assertEqual(len(summary), 5)

    def test_each_beta_level_resets_context_and_memory_and_console_is_explicit(self):
        fixture, agent = self.fixture(); output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            roots = [Path(temporary)/str(i) for i in range(3)]
            first = runner.evaluate_case(fixture, agent, 15., 5, fixture.domain, fixture.reference, roots[0], {}, 'sha')
            runner.evaluate_case(fixture, agent, 20., 7, fixture.domain, fixture.reference, roots[1], {}, 'sha')
            repeated = runner.evaluate_case(fixture, agent, 15., 5, fixture.domain, fixture.reference, roots[2], {}, 'sha')
            self.assertEqual(first, repeated)
            self.assertEqual((roots[0]/'trace.csv').read_bytes(), (roots[2]/'trace.csv').read_bytes())
        for token in ('DATASET=IAT','BETA=15','BETA=20','LEVEL=5','LEVEL=7','BUDGET=20%',
                'RESET=Context/Adam/memory','FINAL_TARGET_F1='):
            self.assertIn(token, output.getvalue())

    def test_completed_results_are_reused_and_tampering_is_detected(self):
        fixture, agent = self.fixture()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)/'case'
            first = runner.evaluate_case(fixture, agent, 10., 5, fixture.domain, fixture.reference, root, {}, 'sha')
            with patch.object(runner.base, 'run_episode', side_effect=AssertionError('Unexpected rerun')):
                repeated = runner.evaluate_case(fixture, agent, 10., 5, fixture.domain, fixture.reference, root, {}, 'sha')
            self.assertEqual(first, repeated)
            result = json.loads((root/'metrics.json').read_text()); result['labels'] = 0
            (root/'metrics.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.evaluate_case(fixture, agent, 10., 5, fixture.domain, fixture.reference, root, {}, 'sha')


if __name__ == '__main__': unittest.main()
