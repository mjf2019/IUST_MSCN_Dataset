"""User-run CPU contracts for q75 as a query gate, not a reward or training change."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.q75_query_guide import Q75Decision,run_guided_case
from enhancements.independent_target_holdout import base
from test_two_action_ddqn import Fixture


class Q75GuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); base.pilot.base.legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def state(self,entropy):
        value=torch.zeros(22); value[9]=entropy
        return value

    def test_guide_requires_both_entropy_and_ddqn_with_budget(self):
        learner=SimpleNamespace(choose=Mock(return_value=(1,np.array([0.,1.]))))
        guide=Q75Decision(learner,2,.5,'Q75GuidedDDQN'); rng=np.random.default_rng(42)
        self.assertEqual(guide.choose(self.state(.4),True,0.,rng)[0],0)
        self.assertEqual(guide.choose(self.state(.5),True,0.,rng)[0],1)
        self.assertEqual(guide.choose(self.state(.9),False,0.,rng)[0],0)
        learner.choose.return_value=(0,np.array([1.,0.]))
        self.assertEqual(guide.choose(self.state(.9),True,0.,rng)[0],0)
        self.assertEqual(guide.threshold,.5)

    def test_q75_only_never_consults_q_or_ddqn_action(self):
        learner=SimpleNamespace(choose=Mock(side_effect=AssertionError('q75-only accessed DDQN')))
        guide=Q75Decision(learner,2,.5,'Q75Only'); rng=np.random.default_rng(42)
        self.assertEqual(guide.choose(self.state(.4),True,0.,rng)[0],0)
        self.assertEqual(guide.choose(self.state(.9),True,0.,rng)[0],1)
        self.assertEqual(guide.choose(self.state(.9),False,0.,rng)[0],0)
        learner.choose.assert_not_called()
        self.assertIsNone(guide.decisions[0]['guide_ddqn_action'])

    def test_invalid_threshold_state_and_exploration_are_rejected(self):
        learner=SimpleNamespace(choose=Mock(return_value=(1,[0.,1.])))
        for threshold in (-1.,1.1,float('nan')):
            with self.assertRaises(ValueError): Q75Decision(learner,2,threshold,'Q75GuidedDDQN')
        guide=Q75Decision(learner,2,.5,'Q75GuidedDDQN')
        with self.assertRaisesRegex(ValueError,'evaluation-only'):
            guide.choose(self.state(.9),True,.1,np.random.default_rng(42))
        with self.assertRaisesRegex(ValueError,'state shape'):
            guide.choose(torch.zeros(21),True,0.,np.random.default_rng(42))

    def test_guided_run_has_immediate_feedback_and_excludes_independent_audit(self):
        f=Fixture(); learner=f.learner()
        with torch.no_grad():
            for p in learner.policy.parameters(): p.zero_()
            learner.policy.network[-1].bias.copy_(torch.tensor([0.,1.]))
        learner.policy.requires_grad_(False)
        audit=(torch.randn(8,1,8)+50,np.arange(8)%2,np.arange(9000,9008))
        originals=[(model,deepcopy(model.state_dict())) for model in (f.cnn,f.template,learner.policy)]
        for method in ('LearnedBetaDDQN','Q75GuidedDDQN','Q75Only'):
            result,trace=run_guided_case(f,learner,f.domain,audit,f.reference,42,method,5)
            self.assertEqual(result['target_holdout_rows'],8)
            self.assertEqual(result['samples'],32)
            self.assertLessEqual(result['labels'],int(.2*32))
            self.assertEqual(result['td_updates'],0)
            self.assertFalse(set(audit[2]) & {int(r['source_row']) for r in trace})
            for row in trace:
                if row['queried']:
                    self.assertEqual(row['feedback_step'],row['step'])
                    self.assertEqual(row['feedback_delay_rows'],0)
                if method!='LearnedBetaDDQN':
                    self.assertEqual(bool(row['guide_action']),row['queried'])
                    if row['queried']: self.assertTrue(row['guide_uncertainty_eligible'])
        for model,saved in originals:
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
        self.assertEqual(learner.steps,0); self.assertEqual(learner.updates,0)

    def test_guide_does_not_force_budget_when_ddqn_rejects(self):
        f=Fixture(); learner=f.learner()
        with torch.no_grad():
            for p in learner.policy.parameters(): p.zero_()
            learner.policy.network[-1].bias.copy_(torch.tensor([1.,0.]))
        f.threshold=0.
        audit=(torch.randn(8,1,8)+50,np.arange(8)%2,np.arange(9000,9008))
        result,trace=run_guided_case(f,learner,f.domain,audit,f.reference,42,'Q75GuidedDDQN',5)
        self.assertEqual(result['labels'],0)
        self.assertEqual(result['guide_high_entropy_skipped_by_ddqn'],32)
        self.assertFalse(any(r['queried'] for r in trace))


if __name__=='__main__': unittest.main()

