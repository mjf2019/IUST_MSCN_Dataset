"""Reference sampling and reward math, independent of GPU/data availability."""
import unittest
import numpy as np
from AMCAL.baselines.balanced_reference import balanced_indices, macro_transfer


class BalancedReferenceTests(unittest.TestCase):
    def test_balanced_unique_and_eligible(self):
        labels=np.repeat(np.arange(3),[90,8,2])
        eligible=np.arange(1,100)
        ids=balanced_indices(labels,eligible,12,np.random.default_rng(42))
        self.assertEqual(len(ids),12)
        self.assertEqual(len(set(ids)),12)
        self.assertNotIn(0,ids)
        self.assertEqual(np.bincount(labels[ids],minlength=3).tolist(),[5,5,2])

    def test_empty_history_and_small_pool(self):
        self.assertEqual(len(balanced_indices([],[],32,np.random.default_rng(1))),0)
        self.assertEqual(set(balanced_indices([0,1,1],[0,2],32,np.random.default_rng(1))),{0,2})

    def test_seed_reproducibility(self):
        labels=np.arange(100)%6
        a=balanced_indices(labels,np.arange(100),32,np.random.default_rng(42))
        b=balanced_indices(labels,np.arange(100),32,np.random.default_rng(42))
        np.testing.assert_array_equal(a,b)
        counts=np.bincount(labels[a],minlength=6)
        self.assertLessEqual(counts.max()-counts.min(),1)

    def test_macro_damage_cannot_be_hidden_by_majority(self):
        labels=np.array([0]*9+[1])
        before=np.ones(10)
        after=np.array([.5]*9+[2.])
        utility,gain,harm=macro_transfer(before,after,labels,1,1,.0001)
        self.assertAlmostEqual(gain,1/6,places=7)
        self.assertAlmostEqual(harm,1/6,places=7)
        self.assertAlmostEqual(utility,-.0001,places=7)
        self.assertGreater(np.mean((before-after)/(before+after+1e-8)),.2)

    def test_no_reference_and_negative_cost(self):
        self.assertEqual(macro_transfer(None,None,None,1,1,.0001),(-.0001,0.,0.))
        self.assertEqual(macro_transfer(None,None,None,1,1,-.005),(.005,0.,0.))

    def test_reward_weights_and_shape(self):
        reward,gain,harm=macro_transfer([1,1],[.5,2],[0,1],2,3,.01)
        self.assertAlmostEqual(reward,2*gain-3*harm-.01)
        with self.assertRaises(ValueError):macro_transfer([1,2],[1],[0,1],1,1,0)


if __name__=='__main__':unittest.main()
