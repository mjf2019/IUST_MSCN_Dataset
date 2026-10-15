"""Run without torch: python AMCAL/ANT_Burst_TEST/test_three_action_reward.py."""
import unittest
from amcal_three_action_reward import routing_reward, prediction_route, available_actions


class RewardContract(unittest.TestCase):
    def test_route_quality(self):
        self.assertEqual(routing_reward(0, 2, 1, 2), 1)
        self.assertEqual(routing_reward(1, 2, 1, 2), -1)
        self.assertEqual(routing_reward(0, 1, 2, 2), -1)
        self.assertEqual(routing_reward(1, 1, 2, 2), 1)

    def test_gain_cost_and_preupdate_quality(self):
        # Context initially wrong: quality remains -1 even after CE improves.
        self.assertAlmostEqual(routing_reward(2, 2, 1, 2, 3, 1), -1 + .5 - .2)
        self.assertAlmostEqual(routing_reward(2, 2, 2, 2, 1, 1), .8)
        self.assertAlmostEqual(routing_reward(2, 2, 2, 2, 1, 3), .3)
        self.assertAlmostEqual(routing_reward(2, 2, 2, 2, 0, 0), .8)

    def test_weight_and_cost(self):
        a=routing_reward(2, 0, 0, 0, 3, 1, selection_cost=.4, gain_weight=2)
        self.assertAlmostEqual(a, 1.6)

    def test_budget_mask_and_routes(self):
        self.assertEqual(available_actions(False), (0, 1))
        self.assertEqual(available_actions(True), (0, 1, 2))
        self.assertEqual(prediction_route(0), 'frozen')
        self.assertEqual(prediction_route(1), 'context')
        self.assertEqual(prediction_route(2), 'context')

    def test_invalid(self):
        with self.assertRaises(ValueError): routing_reward(3, 0, 0, 0)
        with self.assertRaises(ValueError): routing_reward(2, 0, 0, 0)
        with self.assertRaises(ValueError): routing_reward(2, 0, 0, 0, float('nan'), 1)


if __name__ == '__main__':
    unittest.main()
