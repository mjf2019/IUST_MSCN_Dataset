"""Random action ablation: no selector execution, persistent Context adaptation."""
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack
import unittest
import numpy as np
import torch
from test_final_enhancement import FinalEnhancementTests
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode


class PairOnlyRandomTests(unittest.TestCase):
    def test_random_is_independent_of_selector_and_preserves_rl_weights(self):
        FinalEnhancementTests.setUpClass()
        fixture = FinalEnhancementTests()
        fixture.setUp()
        cnn, bank, initial, a, domain, anchors, _ = fixture.fixture(query='policy', policy='learned')
        bank = bank[:1]
        agent = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, slots=1)
        policy = deepcopy(agent.policy.state_dict())
        target = deepcopy(agent.target.state_dict())
        continuation = {}
        with ExitStack() as stack:
            for obj, name in ((agent.policy, 'forward_for_action'), (agent.policy, 'route_logits'),
                              (agent.target, 'forward'), (agent, 'observe_routes'), (agent, 'learn_td')):
                stack.enter_context(patch.object(obj, name, side_effect=AssertionError('RL was called')))
            for _ in range(2):
                metrics, trace = episode(a, cnn, bank, initial, agent, domain, anchors,
                    mode='random', reference=anchors, same_level=True, continuation=continuation)
                self.assertGreater(metrics['labels'], 0)
                self.assertLessEqual(metrics['labels'], metrics['label_cap'])
                self.assertEqual(metrics['route_updates'], 0)
                self.assertEqual(metrics['td_updates'], 0)
                self.assertIsNone(metrics['route_brier_on_full_rows'])
                self.assertTrue(all(row['scores'] is None for row in trace))
        for name, value in policy.items(): self.assertTrue(torch.equal(value, agent.policy.state_dict()[name]))
        for name, value in target.items(): self.assertTrue(torch.equal(value, agent.target.state_dict()[name]))
        self.assertTrue(any(not torch.equal(value, bank[0].state_dict()[name]) for name, value in initial.items()))
        self.assertIn('memory', continuation)

        # An arbitrary selector cannot alter independent random predictions/actions.
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, slots=1)
        with torch.no_grad():
            for p in other.policy.parameters(): p.fill_(100.)
        first, t1 = episode(a, cnn, bank, initial, agent, domain, anchors, mode='random')
        second, t2 = episode(a, cnn, bank, initial, other, domain, anchors, mode='random')
        self.assertEqual(first['queried_source_ids'], second['queried_source_ids'])
        self.assertEqual([(r['action'],r['prediction']) for r in t1], [(r['action'],r['prediction']) for r in t2])
        a.budget_fraction = 0.
        metrics, _ = episode(a, cnn, bank, initial, other, domain, anchors, mode='random')
        self.assertEqual(metrics['labels'], 0)
        for name, value in initial.items(): self.assertTrue(torch.equal(value, bank[0].state_dict()[name]))


if __name__ == '__main__': unittest.main()
