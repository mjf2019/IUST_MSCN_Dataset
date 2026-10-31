"""User-run CPU checks; no real dataset or package installation is required."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_ddqn as runner
from enhancements.competitive_selector import CompetitiveSelector, original_paper_reward
from enhancements.config import EnhancementConfig
from enhancements.agent import EnhancementAgent
from enhancements.two_action_ddqn import QueryDDQN


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = self.conv_block2 = self.conv_block3 = nn.Identity()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 2))

    def forward(self, x): return self.dense_layers(self.flatten(x))


class TinyContext(nn.Module):
    def __init__(self):
        super().__init__(); self.layer = nn.Linear(8, 8)
        nn.init.eye_(self.layer.weight); nn.init.zeros_(self.layer.bias)

    def forward(self, x): return self.layer(x.squeeze(1)).unsqueeze(1)


class Fixture:
    def __init__(self, output=None):
        self.spec = dict(mean=[0.]*22, scale=[1.]*22)
        self.actor = CompetitiveSelector(self.spec).eval().requires_grad_(False)
        with torch.no_grad(): self.actor.actor[-1].bias.fill_(.2)
        self.actor_saved = dict(state=deepcopy(self.actor.state_dict()))
        self.cnn = TinyCNN().eval().requires_grad_(False)
        self.template = TinyContext().eval()
        config = EnhancementConfig.preset('accuracy')
        self.a = SimpleNamespace(classes=['0', '1'], config=config, seed=42, epochs=2, output=output,
            budget_fraction=.2, context_lr=.00025, replay_weight=.1,
            competitive_selector=True, original_paper_reward=True, competitive_spec=self.spec,
            collect_query_targets_only=True, budget_actor_mode=True, eval_routing='policy',
            eval_query_schedule='policy', train_query_schedule='uniform', cost=0.,
            query_lr=.0001, horizon=2, reference_size=4, gain_weight=1., forgetting_weight=1.,
            progress_every=100, actor_experiment=None)
        self.threshold = .5
        self.anchors = torch.randn(16, 1, 8)+5, np.arange(16)%2, np.arange(100, 116)
        self.reference = torch.randn(12, 1, 8)+10, np.arange(12)%2, np.arange(200, 212)
        self.domain = torch.randn(32, 1, 8), np.arange(32)%2, np.arange(32)
        self.train = {level:self.domain for level in runner.TRAIN}

    def context(self):
        self.last_context = deepcopy(self.template).eval()
        return self.last_context

    def memory(self, seed):
        return runner.pilot.Replay(self.anchors, seed+61, self.a.config, 2)

    def split(self, split, levels): return {level:self.domain for level in levels}, self.reference

    def learner(self, **kwargs):
        return QueryDDQN(self.spec, self.actor_saved['state'], 'cpu', 42, **kwargs)


class TwoActionDDQNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def constant(self, network, values):
        for parameter in network.parameters(): parameter.data.zero_()
        network.network[-1].bias.data.copy_(torch.tensor(values, dtype=torch.float32))

    def test_original_paper_reward_table_and_zero_skip_payoff(self):
        self.assertEqual(original_paper_reward(1, 1, 0), 3.)
        self.assertEqual(original_paper_reward(1, 0, 0), 2.)
        self.assertEqual(original_paper_reward(0, 1, 0), -2.)
        self.assertEqual(original_paper_reward(0, 0, 0), 0.)

    def test_parent_encoder_and_train_normalizer_preserved_but_returns_start_at_zero(self):
        fixture = Fixture(); agent = fixture.learner()
        torch.testing.assert_close(agent.policy.network[0].weight, fixture.actor.actor[0].weight)
        torch.testing.assert_close(agent.policy.mean, fixture.actor.mean)
        torch.testing.assert_close(agent.policy.scale, fixture.actor.scale)
        torch.testing.assert_close(agent.policy(torch.ones(1, 22)), torch.zeros(1, 2))

    def test_double_dqn_selects_online_action_and_uses_target_value(self):
        fixture = Fixture(); agent = fixture.learner(batch_size=1, learn_every=1)
        self.constant(agent.policy, [1., 2.]); self.constant(agent.target, [10., 3.])
        result = agent.remember(torch.zeros(22), 1, 0., torch.zeros(22), False, True)
        self.assertAlmostEqual(result['target'], .95*3., places=5)
        self.assertAlmostEqual(result['loss'], .5*(2.-.95*3.)**2, places=5)

    def test_exhausted_budget_masks_query_in_both_action_and_bootstrap(self):
        fixture = Fixture(); agent = fixture.learner(batch_size=1, learn_every=1)
        self.constant(agent.policy, [1., 100.]); self.constant(agent.target, [10., 300.])
        action, _ = agent.choose(torch.zeros(22), False, 1., np.random.default_rng(42))
        self.assertEqual(action, 0)
        result = agent.remember(torch.zeros(22), 0, 0., torch.zeros(22), False, False)
        self.assertAlmostEqual(result['target'], .95*10., places=5)

    def test_terminal_transition_has_no_bootstrap(self):
        fixture = Fixture(); agent = fixture.learner(batch_size=1, learn_every=1)
        self.constant(agent.policy, [1., 2.]); self.constant(agent.target, [100., 300.])
        result = agent.remember(torch.zeros(22), 1, 0., torch.zeros(22), True, False)
        self.assertEqual(result['target'], 0.)
        self.assertAlmostEqual(result['loss'], 1.5, places=6)

    def test_snapshot_restores_optimizer_target_replay_and_rng(self):
        fixture = Fixture(); agent = fixture.learner(batch_size=2, learn_every=1)
        for i in range(6): agent.remember(torch.ones(22)*i, 1, 1., torch.zeros(22), True, False)
        saved = agent.snapshot(); other = fixture.learner(batch_size=2, learn_every=1)
        other.restore(saved)
        self.assertAlmostEqual(agent.learn()['loss'], other.learn()['loss'], places=7)
        for key, value in agent.policy.state_dict().items():
            torch.testing.assert_close(value, other.policy.state_dict()[key], rtol=0, atol=0)

    def test_actor_control_exactly_matches_existing_two_action_context_updates(self):
        fixture = Fixture(); agent = fixture.learner()
        new, trace = runner.run_episode(fixture, agent, fixture.domain, fixture.reference, 42, mode='ActorPaper')
        actual_context = deepcopy(fixture.last_context.state_dict())
        context = fixture.context()
        legacy_agent = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, fixture.a.config, slots=1)
        actor = deepcopy(fixture.actor)
        with redirect_stdout(io.StringIO()):
            live, old, old_trace = runner.pilot.stream(fixture.a,
                (fixture.cnn, context, legacy_agent, actor), fixture.domain, fixture.anchors,
                None, fixture.reference, 42, target_endpoints=False)
        self.assertEqual(new['labels'], old['labels'])
        self.assertEqual([r['queried'] for r in trace], [r['queried'] for r in old_trace])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in old_trace])
        for key, value in actual_context.items(): torch.testing.assert_close(value, live[1].state_dict()[key], rtol=0, atol=0)

    def test_real_transitions_only_and_frozen_q_during_test(self):
        fixture = Fixture(); agent = fixture.learner()
        self.constant(agent.policy, [0., 1.]); agent.target.load_state_dict(agent.policy.state_dict())
        before = deepcopy(agent.policy.state_dict())
        result, trace = runner.run_episode(fixture, agent, fixture.domain, None, 42, training=True, epsilon=0.)
        self.assertEqual(len(agent.memory), len(fixture.domain[1]))
        self.assertTrue(agent.memory[-1][4]); self.assertFalse(agent.memory[-1][5])
        self.assertLessEqual(result['labels'], result['cap'])
        for row, transition in zip(trace, agent.memory):
            self.assertEqual(row['action'], transition[1]); self.assertEqual(row['reward'], transition[2])
            if not row['queried']: self.assertEqual(row['reward'], 0.)
        trained = deepcopy(agent.policy.state_dict()); updates = agent.updates
        runner.run_episode(fixture, agent, fixture.domain, fixture.reference, 42)
        self.assertEqual(updates, agent.updates)
        for key, value in trained.items(): torch.testing.assert_close(value, agent.policy.state_dict()[key], rtol=0, atol=0)

    def test_unqueried_truth_and_current_query_truth_cannot_leak_into_decisions(self):
        fixture = Fixture(); agent = fixture.learner()
        self.constant(agent.policy, [0., 1.])
        _, trace = runner.run_episode(fixture, agent, fixture.domain, fixture.reference, 42)
        paid = {row['source_row'] for row in trace if row['queried']}
        changed = fixture.domain[1].copy()
        for i in range(len(changed)):
            if i not in paid: changed[i] = 1-changed[i]
        _, other = runner.run_episode(fixture, agent, (fixture.domain[0], changed, fixture.domain[2]), fixture.reference, 42)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in other])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in other])
        first = trace[0]['source_row']; changed = fixture.domain[1].copy(); changed[first] = 1-changed[first]
        _, other = runner.run_episode(fixture, agent, (fixture.domain[0], changed, fixture.domain[2]), fixture.reference, 42)
        self.assertEqual(trace[0]['action'], other[0]['action'])
        self.assertEqual(trace[0]['prediction'], other[0]['prediction'])

    def test_training_resume_and_completed_model_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary)/'output'); agent = fixture.learner()
            evaluate = runner.evaluate
            interrupted = [False]
            def validation(*args, **kwargs):
                if args[4].name == 'VAL_epoch_02' and not interrupted[0]:
                    interrupted[0] = True; raise RuntimeError('Simulated interruption')
                return evaluate(*args, **kwargs)
            with redirect_stdout(io.StringIO()), patch.object(runner, 'evaluate', side_effect=validation):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    runner.train(fixture, agent, dict(version=1))
                resumed = fixture.learner()
                runner.train(fixture, resumed, dict(version=1))
                steps = resumed.steps
                runner.train(fixture, resumed, dict(version=1))
                self.assertEqual(steps, resumed.steps)
            complete = json.loads((fixture.a.output/'training_complete.json').read_text())
            self.assertIn(complete['selected_epoch'], [1, 2])
            self.assertEqual(len(json.loads((fixture.a.output/'training.json').read_text())['epochs']), 2)


if __name__ == '__main__': unittest.main()
