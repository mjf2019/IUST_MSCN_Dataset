"""User-run reward, causality and matched-environment checks; no real data needed."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_utility as runner
import amcal_two_action_ddqn as original
from enhancements.query_utility_reward import FixedProbeReward, prepare_train_probes, signed_update_reward, keys
from test_two_action_ddqn import Fixture


class UtilityRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def constant(self, agent, values):
        with torch.no_grad():
            for p in agent.policy.parameters(): p.zero_()
            agent.policy.network[-1].bias.copy_(torch.tensor(values))
        agent.target.load_state_dict(agent.policy.state_dict())

    def setup_reward(self, fixture, cost=.001):
        generalization = torch.randn(12, 1, 8)+20, np.arange(12)%2, np.arange(400, 412)
        fixture.utility_reward = FixedProbeReward([generalization], fixture.reference, 1., cost)
        fixture.train_anchors = fixture.anchors
        fixture.reward_probe_keys = keys(generalization[0])
        return fixture.utility_reward

    def test_signed_gain_harm_and_cost_are_exact(self):
        reward, parts = signed_update_reward((1., .5), (.8, .6), .5, .01)
        self.assertAlmostEqual(reward, .14)
        self.assertAlmostEqual(parts['probe_gain'], .2)
        self.assertAlmostEqual(parts['weighted_probe_harm'], .05)

    def test_harmful_update_is_negative_and_retention_recovery_is_not_penalized(self):
        reward, _ = signed_update_reward((1., .5), (1.2, .6), 1., .01)
        self.assertAlmostEqual(reward, -.31)
        reward, parts = signed_update_reward((1., .5), (.8, .4), 1., 0.)
        self.assertAlmostEqual(reward, .2); self.assertEqual(parts['probe_harm'], 0.)

    def test_nonfinite_measurements_and_negative_parameters_are_rejected(self):
        for before, after, weight, cost in [((float('nan'), 1.), (1., 1.), 1., 0.),
                ((1., 1.), (1., 1.), -1., 0.), ((1., 1.), (1., 1.), 1., -.1)]:
            with self.assertRaises(ValueError): signed_update_reward(before, after, weight, cost)

    def test_macro_loss_does_not_weight_frequent_classes_more(self):
        fixture = Fixture(); context = fixture.context()
        x = torch.randn(8, 1, 8); y = np.array([0]*7+[1])
        measured = FixedProbeReward.macro_loss(fixture.cnn, context, (x, y))
        losses = F.cross_entropy(fixture.cnn(context(x)), torch.tensor(y), reduction='none')
        torch.testing.assert_close(measured, (losses[:7].mean()+losses[7])/2)

    def test_probe_observation_does_not_mutate_models_rng_or_gradients(self):
        fixture = Fixture(); probe = self.setup_reward(fixture); context = fixture.context()
        before_context = deepcopy(context.state_dict()); before_cnn = deepcopy(fixture.cnn.state_dict())
        rng = torch.get_rng_state().clone()
        before = probe.measure(fixture.cnn, context)
        reward, _ = probe.finish(fixture.cnn, context, before)
        self.assertAlmostEqual(reward, -.001)
        torch.testing.assert_close(rng, torch.get_rng_state())
        for k, v in before_context.items(): torch.testing.assert_close(v, context.state_dict()[k], rtol=0, atol=0)
        for k, v in before_cnn.items(): torch.testing.assert_close(v, fixture.cnn.state_dict()[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in [*context.parameters(), *fixture.cnn.parameters()]))

    def test_probe_feature_cannot_enter_current_paid_or_replay_update(self):
        fixture = Fixture(); probe = self.setup_reward(fixture)
        current = (fixture.domain[0][:1], torch.tensor(fixture.domain[1][:1]))
        forbidden = probe.generalization[0][0][:1], torch.tensor([0])
        probe.assert_disjoint_update(current, [], None)
        for query, paid, replay in [(forbidden, [], None), (current, [forbidden], None), (current, [], forbidden)]:
            with self.assertRaisesRegex(ValueError, 'entered the actual update'):
                probe.assert_disjoint_update(query, paid, replay)

    def test_only_paid_train_updates_measure_reward_and_test_freezes_q(self):
        fixture = Fixture(); probe = self.setup_reward(fixture); agent = fixture.learner()
        self.constant(agent, [0., 1.])
        with redirect_stdout(io.StringIO()):
            result, trace = runner.run_episode(fixture, agent, fixture.domain, None, 42, training=True)
        self.assertEqual(probe.calls, 2*result['labels'])
        self.assertEqual(result['labels'], result['cap'])
        for row, transition in zip(trace, agent.memory):
            self.assertEqual(row['reward'], transition[2])
            if row['queried']:
                self.assertAlmostEqual(row['reward'], row['probe_gain']-row['weighted_probe_harm']-row['request_cost'])
            else:
                self.assertEqual(row['reward'], 0.); self.assertIsNone(row['probe_gain'])
        before = deepcopy(agent.policy.state_dict()); calls, updates = probe.calls, agent.updates
        runner.run_episode(fixture, agent, fixture.domain, fixture.reference, 42)
        self.assertEqual(probe.calls, calls); self.assertEqual(agent.updates, updates)
        for k, v in before.items(): torch.testing.assert_close(v, agent.policy.state_dict()[k], rtol=0, atol=0)

    def test_skip_has_zero_reward_without_probe_access(self):
        fixture = Fixture(); probe = self.setup_reward(fixture); agent = fixture.learner()
        self.constant(agent, [1., 0.])
        with redirect_stdout(io.StringIO()):
            result, trace = runner.run_episode(fixture, agent, fixture.domain, None, 42, training=True)
        self.assertEqual(result['labels'], 0); self.assertEqual(probe.calls, 0)
        self.assertTrue(all(row['reward'] == 0 for row in trace))

    def test_parent_actor_evaluation_matches_original_environment_exactly(self):
        fixture = Fixture(); agent = fixture.learner()
        first, trace = original.run_episode(fixture, agent, fixture.domain, fixture.reference, 42, mode='ActorPaper')
        state = deepcopy(fixture.last_context.state_dict())
        self.setup_reward(fixture)
        second, other = runner.run_episode(fixture, agent, fixture.domain, fixture.reference, 42, mode='ActorPaper')
        self.assertEqual(first['online'], second['online']); self.assertEqual(first['labels'], second['labels'])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in other])
        self.assertEqual([r['queried'] for r in trace], [r['queried'] for r in other])
        self.assertEqual(fixture.utility_reward.calls, 0)
        for k, v in state.items(): torch.testing.assert_close(v, fixture.last_context.state_dict()[k], rtol=0, atol=0)

    def test_train_probe_reservation_preserves_streams_and_excludes_ids_and_duplicates(self):
        fixture = Fixture(); fixture.a.burst_dir = Path('/unused'); fixture.artifact = {}
        pools = {level:(torch.randn(80, 1, 8)+20+level*20, np.arange(80)%2,
            np.arange(1000+level*1000, 1080+level*1000)) for level in runner.TRAIN}
        fixture.anchors = pools[0]
        before = {level:runner.domain_hash(d) for level, d in fixture.train.items()}
        with patch.object(runner.pilot.base.legacy, 'read_domain', side_effect=lambda burst, artifact, level, split:pools[level]):
            probes, retention, anchors, manifest = prepare_train_probes(fixture, runner.pilot, 6, 6)
        forbidden_ids = {int(i) for d in [*probes, retention] for i in d[2]}
        forbidden_keys = set().union(*(keys(d[0]) for d in [*probes, retention]))
        self.assertFalse(set(map(int, anchors[2])).intersection(forbidden_ids))
        self.assertFalse(keys(anchors[0]).intersection(forbidden_keys))
        self.assertEqual(before, {level:runner.domain_hash(d) for level, d in fixture.train.items()})
        self.assertEqual(manifest['excluded_anchor_rows'], 6)
        for d in fixture.train.values(): self.assertFalse(keys(d[0]).intersection(forbidden_keys))
        with patch.object(runner.pilot.base.legacy, 'read_domain', return_value=fixture.domain):
            with self.assertRaisesRegex(ValueError, 'no validation/test'):
                prepare_train_probes(fixture, runner.pilot, 6, 6)

    def test_validation_selects_lower_consumption_only_with_accuracy_retention_constraints(self):
        reference = dict(AC=80., positive_forgetting=1., labels_percent=20.)
        rows = [dict(candidate='good', AC=79.8, labels_percent=10., positive_forgetting=1.2),
                dict(candidate='bad_accuracy', AC=78., labels_percent=1., positive_forgetting=0.),
                dict(candidate='bad_retention', AC=80., labels_percent=2., positive_forgetting=3.)]
        result = runner.select_on_validation(rows, reference)
        self.assertEqual(result['selected'], 'good'); self.assertTrue(result['validation_goal_met'])
        self.assertAlmostEqual(result['label_saving_pp'], 10.)
        unchanged = runner.select_on_validation([dict(candidate='all_budget', AC=80.,
            positive_forgetting=1., labels_percent=20.)], reference)
        self.assertTrue(unchanged['accuracy_retention_constraints_met'])
        self.assertFalse(unchanged['validation_goal_met'])
        failed = runner.select_on_validation(rows[1:], reference)
        self.assertFalse(failed['validation_goal_met']); self.assertEqual(failed['eligible'], [])


if __name__ == '__main__': unittest.main()
