"""User-run synthetic-domain contracts; no real data or network access required."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_two_action_drift_utility as runner
from enhancements.query_drift import DriftSpec, fit_train_scale, transform_domain, episode_spec, choose_validation_candidate
from test_two_action_ddqn import Fixture


class SyntheticUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.pilot.base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def experiment(self):
        fixture = Fixture()
        experiment = runner.DriftExperiment.__new__(runner.DriftExperiment)
        experiment.__dict__.update(fixture.__dict__)
        probes = [(torch.randn(12, 1, 8)+20+20*level, np.arange(12)%2,
            np.arange(400+level*100, 412+level*100)) for level in runner.TRAIN]
        experiment.enable_synthetic(probes, fixture.reference, fixture.anchors)
        return experiment, fixture

    def test_scale_uses_only_train_features_and_ignores_labels(self):
        fixture = Fixture(); changed = (fixture.domain[0], 1-fixture.domain[1], fixture.domain[2])
        scale = fit_train_scale([fixture.domain])
        torch.testing.assert_close(scale, fit_train_scale([changed]))
        torch.testing.assert_close(scale, fixture.domain[0].double().std(0, unbiased=False).float())

    def test_transform_is_label_blind_preserves_ids_and_does_not_mutate_inputs(self):
        fixture = Fixture(); data = fixture.domain
        original = data[0].clone(); scale = fit_train_scale([data]); spec = episode_spec(42, 'TRAIN', 0)
        first = transform_domain(data, scale, spec)
        second = transform_domain((data[0], 1-data[1], data[2]), scale, spec)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(data[0], original, rtol=0, atol=0)
        np.testing.assert_array_equal(first[1], data[1]); np.testing.assert_array_equal(first[2], data[2])
        self.assertFalse(torch.equal(first[0], original))

    def test_row_noise_is_independent_of_batch_order_and_probe_membership(self):
        fixture = Fixture(); data = fixture.domain; scale = fit_train_scale([data]); spec = episode_spec(42, 'TRAIN', 1)
        first = transform_domain(data, scale, spec)
        at = np.array([9, 3, 7])
        subset = data[0][at], data[1][at], data[2][at]
        second = transform_domain(subset, scale, spec)
        torch.testing.assert_close(first[0][at], second[0], rtol=0, atol=0)
        torch.testing.assert_close(transform_domain(data, scale, DriftSpec(1, 0.))[0], data[0], rtol=0, atol=0)

    def test_train_epochs_and_val_have_distinct_predeclared_drift(self):
        train = [episode_spec(42, 'TRAIN', level, epoch).seed for epoch in (1, 2, 3) for level in runner.TRAIN]
        validation = [episode_spec(42, 'VAL', level).seed for level in runner.TRAIN]
        self.assertEqual(len(train), len(set(train)))
        self.assertFalse(set(train).intersection(validation))
        with self.assertRaisesRegex(ValueError, 'TEST'): episode_spec(42, 'TEST', 5)
        with self.assertRaises(ValueError): DriftSpec(1, 4.).validated()

    def test_reward_probe_uses_same_episode_transform_as_stream(self):
        experiment, fixture = self.experiment()
        experiment.reward_parameters = dict(forgetting_weight=.5, cost=0.)
        experiment.set_train_epoch(2)
        for level in runner.TRAIN:
            spec = episode_spec(42, 'TRAIN', level, 2)
            expected = transform_domain(experiment.base_probes[level], experiment.feature_scale, spec)
            probe = experiment.reward_for_domain(experiment.train[level])
            torch.testing.assert_close(expected[0], probe.generalization[0][0], rtol=0, atol=0)
            torch.testing.assert_close(probe.retention[0], fixture.reference[0], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'same synthetic episode'):
            experiment.reward_for_domain(fixture.domain)

    def test_val_is_shifted_and_test_rows_and_audit_are_exactly_unchanged(self):
        experiment, fixture = self.experiment()
        original = fixture.domain[0].clone()
        with patch.object(runner.Experiment, 'split', return_value=({0:fixture.domain}, fixture.reference)):
            validation, audit = experiment.split('val_ids', [0])
            test, test_audit = experiment.split('test_ids', [0])
        self.assertFalse(torch.equal(validation[0][0], original))
        self.assertIs(test[0], fixture.domain); self.assertIs(test_audit, fixture.reference)
        self.assertIs(audit, fixture.reference)
        torch.testing.assert_close(fixture.domain[0], original, rtol=0, atol=0)

    def test_probe_is_train_only_and_no_test_query_learning_occurs(self):
        experiment, fixture = self.experiment()
        experiment.reward_parameters = dict(forgetting_weight=.5, cost=0.)
        experiment.set_train_epoch(1)
        agent = fixture.learner()
        with torch.no_grad():
            for p in agent.policy.parameters(): p.zero_()
            agent.policy.network[-1].bias.copy_(torch.tensor([0., 1.]))
        agent.target.load_state_dict(agent.policy.state_dict())
        domain = experiment.train[0]; probe = experiment.reward_for_domain(domain)
        with redirect_stdout(io.StringIO()):
            result, trace = runner.run_episode(experiment, agent, domain, None, 42, training=True)
        self.assertEqual(probe.calls, result['labels']*2)
        self.assertEqual(result['labels'], result['cap'])
        calls, updates = probe.calls, agent.updates
        weights = deepcopy(agent.policy.state_dict())
        runner.run_episode(experiment, agent, fixture.domain, fixture.reference, 42)
        self.assertEqual(probe.calls, calls); self.assertEqual(agent.updates, updates)
        for k, v in weights.items(): torch.testing.assert_close(v, agent.policy.state_dict()[k], rtol=0, atol=0)

    def test_each_training_epoch_refreshes_drift_with_matching_probe(self):
        experiment, fixture = self.experiment()
        experiment.reward_parameters = dict(forgetting_weight=.5, cost=0.)
        with tempfile.TemporaryDirectory() as temporary:
            experiment.a.output = Path(temporary)/'candidate'
            def split(split, levels): return {level:fixture.domain for level in levels}, fixture.reference
            original_set_epoch = experiment.set_train_epoch
            with patch.object(runner.Experiment, 'split', side_effect=split), \
                    patch.object(experiment, 'set_train_epoch', wraps=original_set_epoch) as update, \
                    redirect_stdout(io.StringIO()):
                runner.train(experiment, fixture.learner(), dict(version=1))
            self.assertEqual([call.args[0] for call in update.call_args_list], [1, 2])
            self.assertTrue((experiment.a.output/'training_complete.json').is_file())

    def test_zero_request_policy_cannot_be_reported_as_success(self):
        reference = dict(AC=80., positive_forgetting=1., labels_percent=20.)
        no_query = dict(AC=75., positive_forgetting=0., labels_percent=0.)
        rows = [dict(candidate='skip', AC=75., positive_forgetting=0., labels_percent=0.),
            dict(candidate='useful', AC=79.8, positive_forgetting=1., labels_percent=10.)]
        selected = choose_validation_candidate(rows, reference, no_query)
        self.assertEqual(selected['selected'], 'useful'); self.assertTrue(selected['validation_goal_met'])
        failed = choose_validation_candidate(rows[:1], reference, no_query)
        self.assertFalse(failed['validation_goal_met']); self.assertEqual(failed['eligible'], [])
        invalid = choose_validation_candidate(rows, reference, dict(AC=81., labels_percent=0.))
        self.assertFalse(invalid['development_scenario_valid']); self.assertFalse(invalid['validation_goal_met'])


if __name__ == '__main__': unittest.main()
