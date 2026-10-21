from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancements.utility_normalization import fit_scale, relative_utility, validate_calibration
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode
import test_final_enhancement as fixtures


class UtilityNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_train_fit_matches_relative_component_units_and_rejects_wrong_provenance(self):
        u = np.tile([-.001, 0., .001], (20,1))
        p = np.tile([.2, .5, .8], (20,1))
        value = fit_scale(u, p)
        self.assertAlmostEqual(value['scale'], 600.)
        value.update(checkpoint_sha256='abc', levels=[0,1,3])
        self.assertEqual(validate_calibration(value, 'abc', [0,1,3])['scope'], 'TRAIN')
        for changed in ({'scope':'TEST'}, {'checkpoint_sha256':'bad'}, {'levels':[5]}, {'scale':float('nan')}):
            with self.assertRaises(ValueError): validate_calibration({**value, **changed}, 'abc', [0,1,3])
        with self.assertRaises(ValueError): fit_scale([[float('nan')]*3], [[.5]*3])
        self.assertTrue(np.isfinite(fit_scale(np.zeros((5,3)), np.ones((5,3)))['scale']))

    def test_centering_preserves_mean_and_clips_only_large_differences(self):
        x = torch.tensor([[.02,.02,.02],[.5,.5+1e-9,.5-1e-9],[2.,-1.,0.]],dtype=torch.float64)
        result = relative_utility(x,1000.,clip=1.)
        self.assertTrue(torch.allclose(result.mean(1), x.mean(1)))
        self.assertTrue(torch.allclose(result[0],x[0]))
        self.assertLess(float((result[1]-result[1].mean()).abs().max()), 2e-6)
        self.assertLessEqual(float((result-result.mean(1,keepdim=True)).abs().max()), 1.000000001)
        shifted = relative_utility(x+10,1000.,clip=1.)
        self.assertTrue(torch.allclose(shifted,result+10))

    def test_normalized_actor_changes_destination_but_bootstrap_uses_raw_target_units(self):
        *_, agent = fixtures.FinalEnhancementTests().fixture()
        state = torch.zeros(1,139); state[:,-3]=1; state[:,-2]=1
        raw = torch.tensor([[0.,3.1,3.,3.,3.0997,3.0003,3.]])
        u = torch.tensor([[-.0003,.0003,0.]])
        agent.policy.utility_normalization = dict(scale=1000.,weight=1.,clip=1.)
        before = deepcopy(agent.policy.state_dict())
        with patch.object(agent.policy,'forward_components',return_value=(raw,torch.zeros(1,4),u)), \
             patch.object(agent.target,'forward',return_value=torch.arange(10.,17.)[None]):
            selected, _, critic, _ = agent.policy.forward_for_action(state)
            self.assertEqual(int(raw[:,4:].argmax())+4,4)
            self.assertEqual(int(selected[:,4:].argmax())+4,5)
            self.assertTrue(torch.equal(critic,raw))
            self.assertTrue(torch.equal(selected[:,:4],raw[:,:4]))
            self.assertEqual(float(agent._next_values(state)),15.)
        for k,v in before.items(): self.assertTrue(torch.equal(v,agent.policy.state_dict()[k]))

    def test_normalized_stream_does_not_learn_from_unqueried_labels(self):
        f = fixtures.FinalEnhancementTests(); f.setUp()
        cnn,bank,initial,a,domain,anchors,agent = f.fixture(policy='learned')
        agent.policy.utility_normalization = dict(scale=1000.,weight=1.,clip=1.)
        start = deepcopy(agent.policy.state_dict())
        m,trace = episode(a,cnn,bank,initial,agent,domain,anchors)
        after = deepcopy(agent.policy.state_dict()); paid=set(m['queried_source_ids'])
        labels=domain[1].copy()
        for i,source in enumerate(domain[2]):
            if source not in paid: labels[i]=1-labels[i]
        other = EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start)
        other.policy.utility_normalization = dict(scale=1000.,weight=1.,clip=1.)
        _,repeated=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors)
        self.assertEqual(m['labels'],m['label_cap'])
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in repeated])
        self.assertEqual([r['reward'] for r in trace],[r['reward'] for r in repeated])
        for k,v in after.items(): self.assertTrue(torch.equal(v,other.policy.state_dict()[k]))


if __name__=='__main__': unittest.main()
