"""User-run CPU contracts for the clean-only learned-beta experiment."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amcal_clean_meta_beta as runner
from test_two_action_ddqn import Fixture
from enhancements.meta_beta_ddqn import MetaBetaDDQN, ComponentReward


def domain(count=700, offset=0):
    x = torch.randn(count, 1, 8)+offset/1000
    return x, np.arange(count)%2, np.arange(offset, offset+count)


class CleanFixture(runner.CleanExperiment):
    def __init__(self, output):
        original = Fixture(output)
        self.a = original.a
        self.a.dataset = 'SDNCampus'; self.a.warmup_epochs = self.a.epochs = 1
        self.a.beta = 0.; self.a.protocol = 'Online'
        self.cnn = original.cnn; self.template = original.template
        self.full_train = domain(700, 1000)
        self.parts = runner.reserve_clean(self.full_train, 42, 2, meta_rows=160)
        self.train = self.parts['inner']; self.reference = self.parts['reference']
        self.anchors = self.full_train; self.train_anchors = self.train
        self.reward_probe_keys = set().union(*(set(runner.api.feature_keys(self.parts[k][0]))
            for k in ('gain','meta_gain')))
        self.gain_by_domain = {}
        self.validation = domain(100, 4000)
        self.val_stream, self.val_reference, _ = runner.api.reference_partition(self.validation, 42, .2)


class CleanMetaBetaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.api.base.legacy.core.device = torch.device('cpu')

    def setUp(self):
        torch.manual_seed(42)

    def test_only_matching_clean_original_source_is_accepted(self):
        recipe = dict(dataset='IUST', seed=42, training_levels=[0],
                      validation_levels=[0], architecture='original CNN')
        runner.validate_recipe(recipe, 'IUST', 42)
        for change in (dict(training_levels=[0,1,3]), dict(validation_levels=[1]),
                       dict(dataset='IoT'), dict(seed=43), dict(architecture='Light CNN')):
            with self.assertRaises(ValueError):
                runner.validate_recipe(dict(recipe, **change), 'IUST', 42)

    def test_reserves_are_pairwise_id_and_feature_disjoint(self):
        data = domain()
        # Feature duplicates are removed as whole groups from later roles.
        data[0][1] = data[0][0]
        parts = runner.reserve_clean(data, 42, 2)
        names = list(parts)
        for i, name in enumerate(names):
            runner.check_classes(parts[name], 2, name)
            for other in names[:i]:
                self.assertFalse(set(parts[name][2]) & set(parts[other][2]))
                self.assertFalse(set(runner.api.feature_keys(parts[name][0])) &
                                 set(runner.api.feature_keys(parts[other][0])))

    def test_inner_train_is_every_remaining_row_not_a_pilot_cap(self):
        data = domain(1200)
        parts = runner.reserve_clean(data, 42, 2)
        held = [d for name, d in parts.items() if name != 'inner']
        expected = runner.without(data, held)
        self.assertEqual(runner.base.domain_hash(parts['inner']), runner.base.domain_hash(expected))
        self.assertGreater(len(parts['inner'][1]), 300)

    def test_insufficient_clean_reserves_fail_without_using_other_domains(self):
        with self.assertRaises(ValueError):
            runner.reserve_clean(domain(80), 42, 2)

    def test_uniform_collection_spreads_exact_budget_and_respects_cap(self):
        for n in (160, 300, 619, 1195):
            collector = runner.UniformCollector(n)
            actions = [collector.choose(torch.zeros(22), True, 0., None)[0] for _ in range(n)]
            self.assertEqual(sum(actions), int(.2*n))
            self.assertEqual(actions[-1], 1)
            self.assertEqual(collector.updates, 0)
        collector = runner.UniformCollector(5)
        for _ in range(5):
            self.assertEqual(collector.choose(torch.zeros(22), False, 0., None)[0], 0)

    def test_meta_records_only_observed_paid_reward_components(self):
        collector = runner.UniformCollector(300, components=True)
        for i in range(300):
            state = torch.zeros(22)
            action, _ = collector.choose(state, True, 0., None)
            reward = ComponentReward(2., .03, 4.) if action else 0.
            collector.remember(state, action, reward, state, i == 299, False)
        self.assertEqual(len(collector.rows), 60)
        self.assertEqual(collector.rows[0][1:], (2., .03))
        with self.assertRaises(ValueError):
            collector.remember(torch.zeros(22), 1, 1., torch.zeros(22), True, False)

    def test_normalizer_handles_constant_features_without_test_statistics(self):
        spec = runner.normalization([torch.tensor([1., 2.]), torch.tensor([3., 2.])])
        self.assertEqual(spec['mean'], [2., 2.])
        self.assertEqual(spec['scale'], [1., 1.])

    def test_paper_warmup_removes_gain_oracle_only(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = CleanFixture(Path(temp)/'run')
            observer = runner.PaperOnlyExperiment(exp)
            self.assertIs(observer.cnn, exp.cnn)
            self.assertIsNone(observer.reward_for_domain(exp.train))
            with self.assertRaises(ValueError):
                exp.reward_for_domain(exp.validation)

    def test_clean_training_checkpoint_reuse_and_no_target_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            exp = CleanFixture(Path(temp)/'run'); protocol = dict(version=1, training_levels=[0])
            before = (runner.base.weight_hash(exp.cnn.state_dict()),
                      runner.base.weight_hash(exp.template.state_dict()))
            with redirect_stdout(io.StringIO()), patch.object(runner, 'attack_cases',
                    side_effect=AssertionError('Target CSV opened during TRAIN')):
                runner.train(exp, protocol)
                checkpoint = runner.api.digest(exp.a.output/'Models/learned_beta.pth')
                runner.train(exp, protocol)
                self.assertEqual(checkpoint, runner.api.digest(exp.a.output/'Models/learned_beta.pth'))
            selected, done = runner.load_selected(exp, protocol)
            self.assertTrue(done['selected_beta'] > 0)
            self.assertEqual(selected.updates, 0)
            self.assertEqual(selected.steps, 0)
            self.assertEqual(before, (runner.base.weight_hash(exp.cnn.state_dict()),
                                     runner.base.weight_hash(exp.template.state_dict())))

    def test_each_test_episode_resets_context_and_freezes_q_beta(self):
        exp = Fixture(); exp.a.beta = 1.
        agent = MetaBetaDDQN(exp.spec, None, 'cpu', 42)
        with torch.no_grad():
            agent.policy.network[-1].bias.copy_(torch.tensor([0., 1.]))
        before = deepcopy(agent.policy.state_dict()); beta = agent.beta_value()
        first, trace1 = runner.base.run_episode(exp, agent, exp.domain, exp.reference, 42)
        second, trace2 = runner.base.run_episode(exp, agent, exp.domain, exp.reference, 42)
        self.assertEqual(first['labels'], int(.2*len(exp.domain[1])))
        self.assertEqual([r['prediction'] for r in trace1], [r['prediction'] for r in trace2])
        self.assertEqual(first['online'], second['online'])
        self.assertEqual((agent.steps, agent.updates, agent.beta_value()), (0, 0, beta))
        for key, value in before.items():
            torch.testing.assert_close(value, agent.policy.state_dict()[key], rtol=0, atol=0)

    def test_summary_does_not_call_seen_stream_accuracy_independent_final(self):
        exp = Fixture(); agent = exp.learner()
        result, _ = runner.base.run_episode(exp, agent, exp.domain, exp.reference, 42)
        row = runner.summary_row('IUST', 'Online', dict(attack='CW', strength=.1, level=1), result, 1.)
        self.assertNotIn('target_final_AC', row)
        self.assertEqual(list(k for k in row if k in runner.METRICS), list(runner.METRICS))
        for key in runner.METRICS:
            self.assertIn('original_forgetting_'+key, row)

    def test_target_recipe_has_all_twelve_supplied_intensities(self):
        self.assertEqual(set(runner.DATASETS), {'SDNCampus','ISCX-TOR','IUST','IoT'})
        self.assertEqual(sum(len(s) for s in runner.ATTACKS.values()), 12)
        self.assertEqual(tuple(runner.ATTACKS['CW']), (.1, 1., 5.))
        self.assertEqual(tuple(runner.ATTACKS['FGSM']), (.01, .1, .5))


if __name__ == '__main__':
    unittest.main()
