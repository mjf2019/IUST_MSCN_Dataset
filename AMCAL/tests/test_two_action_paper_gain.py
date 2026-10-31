"""User-run additive reward contracts; no real data/checkpoints required."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_paper_gain as runner
import amcal_two_action_utility as original
from enhancements.paper_gain_reward import SameDomainGain, additive_reward
from enhancements.query_utility_reward import keys
from test_two_action_ddqn import Fixture


class PaperGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def experiment(self, beta):
        fixture = Fixture(); fixture.a.dataset = 'IAT'
        experiment = runner.PaperGainExperiment.__new__(runner.PaperGainExperiment)
        experiment.__dict__.update(fixture.__dict__)
        experiment.train = {level:(fixture.domain[0].clone(), fixture.domain[1].copy(), fixture.domain[2].copy())
            for level in runner.TRAIN}
        probes = [(torch.randn(12, 1, 8)+20+20*level, np.arange(12)%2,
            np.arange(400+level*100, 412+level*100)) for level in runner.TRAIN]
        experiment.train_anchors = fixture.anchors
        experiment.reward_probe_keys = set().union(*(keys(d[0]) for d in probes))
        experiment.set_gain_reward(probes, fixture.reference, beta)
        return experiment, fixture, probes

    def test_exact_paper_plus_signed_gain_without_cost_or_harm_penalty(self):
        self.assertAlmostEqual(additive_reward(3., .02, 10.)[0], 3.2)
        self.assertAlmostEqual(additive_reward(3., -.02, 10.)[0], 2.8)
        self.assertAlmostEqual(additive_reward(-2., .02, 100.)[0], 0.)
        for reward in (-2., 0., 2., 3.):
            self.assertEqual(additive_reward(reward, -.7, 0.)[0], reward)
        for paper, gain, beta in [(0., 0., -1.), (float('nan'), 0., 1.), (0., float('inf'), 1.)]:
            with self.assertRaises(ValueError): additive_reward(paper, gain, beta)

    def test_probe_uses_plain_mean_ce_and_never_measures_retention_loss(self):
        experiment, fixture, probes = self.experiment(10.)
        context = experiment.context(); probe = experiment.reward_for_domain(experiment.train[0])
        x, y = probes[0][:2]
        expected = float(F.cross_entropy(experiment.cnn(context(x)), torch.tensor(y)))
        before = deepcopy(context.state_dict()); rng = torch.get_rng_state().clone()
        with patch.object(SameDomainGain, 'macro_loss', side_effect=AssertionError('Unexpected macro/reference loss')):
            measured = probe.measure(experiment.cnn, context)
            reward, parts = probe.finish(experiment.cnn, context, measured, 3.)
        self.assertAlmostEqual(measured, expected); self.assertEqual(reward, 3.)
        self.assertEqual(parts['probe_gain'], 0.); self.assertNotIn('probe_harm', parts)
        self.assertNotIn('request_cost', parts)
        self.assertEqual(probe.calls, 2)
        torch.testing.assert_close(rng, torch.get_rng_state())
        for key, value in before.items(): torch.testing.assert_close(value, context.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in context.parameters()))

    def test_each_train_domain_uses_only_its_own_probe(self):
        experiment, fixture, probes = self.experiment(10.)
        # Rebuilding streams must also rebuild the identity-to-probe mapping.
        experiment.train = {level:(fixture.domain[0].clone(), fixture.domain[1].copy(), fixture.domain[2].copy())
            for level in runner.TRAIN}
        experiment.set_gain_reward(probes, fixture.reference, 10.)
        for level, expected in zip(runner.TRAIN, probes):
            actual = experiment.reward_for_domain(experiment.train[level])
            self.assertEqual(len(actual.generalization), 1)
            self.assertIs(actual.generalization[0], expected)
        with self.assertRaises(ValueError): experiment.reward_for_domain(fixture.domain)
        with self.assertRaises(ValueError): experiment.set_gain_reward(probes[:2], fixture.reference, 10.)
        experiment.set_gain_reward(probes, fixture.reference, 0.)
        self.assertTrue(all(experiment.reward_for_domain(d) is None for d in experiment.train.values()))

    def test_reward_matches_trace_and_probes_are_train_only(self):
        experiment, fixture, probes = self.experiment(10.)
        agent = fixture.learner()
        with torch.no_grad():
            for p in agent.policy.parameters(): p.zero_()
            agent.policy.network[-1].bias.copy_(torch.tensor([0., 1.]))
        agent.target.load_state_dict(agent.policy.state_dict())
        domain = experiment.train[0]; probe = experiment.reward_for_domain(domain)
        with redirect_stdout(io.StringIO()):
            result, trace = runner.run_episode(experiment, agent, domain, None, 42, training=True)
        self.assertEqual(probe.calls, 2*result['labels'])
        for row, transition in zip(trace, agent.memory):
            self.assertEqual(row['reward'], transition[2])
            if row['queried']:
                self.assertAlmostEqual(row['reward'], row['paper_reward_component']+10.*row['probe_gain'])
            else: self.assertEqual(row['reward'], 0.)
        calls, updates = probe.calls, agent.updates; weights = deepcopy(agent.policy.state_dict())
        runner.run_episode(experiment, agent, domain, fixture.reference, 42)
        self.assertEqual(calls, probe.calls); self.assertEqual(updates, agent.updates)
        for key, value in weights.items(): torch.testing.assert_close(value, agent.policy.state_dict()[key], rtol=0, atol=0)

    def test_probe_cannot_enter_current_paid_or_replay_update(self):
        experiment, fixture, probes = self.experiment(10.)
        probe = experiment.reward_for_domain(experiment.train[0])
        current_probe = probe.generalization[0][0][:1], torch.tensor([0])
        with self.assertRaises(ValueError): probe.assert_disjoint_update(current_probe, [], None)
        with self.assertRaises(ValueError): probe.assert_disjoint_update((fixture.domain[0][:1], torch.tensor([0])), [current_probe], None)
        with self.assertRaises(ValueError): probe.assert_disjoint_update((fixture.domain[0][:1], torch.tensor([0])), [], current_probe)

    def test_beta_zero_update_path_matches_original_paper_environment(self):
        experiment, fixture, probes = self.experiment(0.)
        first, trace = original.run_episode(experiment, fixture.learner(), fixture.domain, fixture.reference, 42, mode='ActorPaper')
        second, other = runner.run_episode(experiment, fixture.learner(), fixture.domain, fixture.reference, 42, mode='ActorPaper')
        self.assertEqual(first['online'], second['online']); self.assertEqual(first['labels'], second['labels'])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in other])
        self.assertEqual([r['queried'] for r in trace], [r['queried'] for r in other])

    def test_val_selection_can_retain_baseline_and_rejects_no_query_collapse(self):
        baseline = dict(beta=0., AC=70., positive_forgetting=0., labels_percent=20.)
        unchanged = runner.select_beta([baseline, dict(beta=100., AC=72., positive_forgetting=0., labels_percent=0.)])
        self.assertEqual(unchanged['selected_beta'], 0.)
        self.assertFalse(unchanged['validation_score_improved'])
        better = runner.select_beta([baseline, dict(beta=10., AC=71., positive_forgetting=0., labels_percent=18.)])
        self.assertEqual(better['selected_beta'], 10.); self.assertTrue(better['validation_score_improved'])
        self.assertAlmostEqual(better['labels_saved_pp'], 2.)
        with self.assertRaises(ValueError): runner.select_beta([dict(beta=10., AC=71., positive_forgetting=0., labels_percent=18.)])


if __name__ == '__main__': unittest.main()
