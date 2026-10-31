"""User-run CPU contracts: q75 behavior/exploration, real successor mask and resume."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.q75_masked_ddqn import Q75MaskedDDQN
from test_two_action_ddqn import Fixture
import amcal_q75_aligned_ddqn as runner


class AlignedQ75Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); runner.base.pilot.base.legacy.core.device=torch.device('cpu')

    def setUp(self):
        torch.manual_seed(42); self.f=Fixture()

    def agent(self,**kwargs):
        return Q75MaskedDDQN(self.f.spec,self.f.actor_saved['state'],'cpu',42,
            threshold=.5,classes=2,**kwargs)

    def state(self,entropy):
        value=torch.zeros(22); value[9]=entropy
        return value

    def constant(self,network,values):
        with torch.no_grad():
            for p in network.parameters(): p.zero_()
            network.network[-1].bias.copy_(torch.tensor(values,dtype=torch.float32))

    def test_exploration_and_greedy_cannot_query_below_q75_or_without_budget(self):
        a=self.agent(); self.constant(a.policy,[0.,100.]); rng=np.random.default_rng(42)
        for epsilon in (0.,.3,1.):
            for _ in range(50):
                self.assertEqual(a.choose(self.state(.1),True,epsilon,rng)[0],0)
                self.assertEqual(a.choose(self.state(.9),False,epsilon,rng)[0],0)
        self.assertEqual(a.choose(self.state(.5),True,0.,rng)[0],1)

    def test_bootstrap_ignores_illegal_query_even_with_huge_query_q(self):
        a=self.agent(batch_size=1,learn_every=100)
        self.constant(a.policy,[0.,100.]); self.constant(a.target,[2.,999.])
        a.remember(self.state(.9),0,0.,self.state(.1),False,True)
        self.assertFalse(a.memory[-1][-1])
        report=a.learn()
        self.assertAlmostEqual(report['target'],.95*2.,places=5)

    def test_legal_successor_query_and_terminal_have_correct_targets(self):
        a=self.agent(batch_size=1,learn_every=100)
        self.constant(a.policy,[0.,100.]); self.constant(a.target,[2.,999.])
        a.remember(self.state(.9),0,0.,self.state(.9),False,True)
        self.assertTrue(a.memory[-1][-1])
        self.assertAlmostEqual(a.learn()['target'],.95*999.,places=3)
        a=self.agent(batch_size=1,learn_every=100)
        self.constant(a.policy,[0.,100.]); self.constant(a.target,[2.,999.])
        a.remember(self.state(.9),1,3.,self.state(.1),True,False)
        self.assertAlmostEqual(a.learn()['target'],3.,places=5)
        with self.assertRaisesRegex(ValueError,'q75 action mask'):
            a.remember(self.state(.1),1,3.,self.state(.9),False,True)

    def test_warm_start_synchronizes_target_and_has_fresh_replay(self):
        source=self.f.learner(); self.constant(source.policy,[3.,7.])
        a=runner.new_agent(self.f,source.policy.state_dict(),True)
        self.assertEqual(len(a.memory),0); self.assertEqual(a.steps,0); self.assertEqual(a.updates,0)
        for key,value in a.policy.state_dict().items():
            torch.testing.assert_close(value,a.target.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(value,source.policy.state_dict()[key],rtol=0,atol=0)

    def test_snapshot_rejects_a_different_mask_and_restores_masked_replay(self):
        a=self.agent(learn_every=100)
        a.remember(self.state(.9),0,0.,self.state(.1),False,True)
        saved=a.snapshot()
        restored=self.agent(learn_every=100); restored.restore(saved)
        self.assertFalse(restored.memory[0][-1])
        different=Q75MaskedDDQN(self.f.spec,self.f.actor_saved['state'],'cpu',threshold=.4,classes=2)
        with self.assertRaisesRegex(ValueError,'mask differs'): different.restore(saved)

    def test_real_training_episode_preserves_original_cnn_context_and_legality(self):
        a=self.agent(batch_size=4,learn_every=1,target_every=2)
        hashes=[runner.base.weight_hash(m.state_dict()) for m in (self.f.cnn,self.f.template)]
        result,trace=runner.base.run_episode(self.f,a,self.f.domain,None,42,training=True,epsilon=1.)
        self.assertLessEqual(result['labels'],result['cap']); self.assertGreater(a.updates,0)
        for state,action,_,successor,terminal,legal in a.memory:
            if action==1: self.assertTrue(a.eligible(state))
            if legal: self.assertFalse(terminal); self.assertTrue(a.eligible(successor))
        self.assertEqual(hashes,[runner.base.weight_hash(m.state_dict()) for m in (self.f.cnn,self.f.template)])

    def test_epoch_zero_baseline_selection_uses_only_validation(self):
        a=self.agent()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.f.a.output=root; protocol={'fixture':True}
            def validation(experiment,learner,split,levels,directory,controls):
                self.assertEqual(split,'val_ids'); self.assertEqual(levels,[0,1,3])
                return [dict(AC=80.,forgetting=0.,labels_percent=12.)]
            def train(experiment,learner,protocol):
                candidate=experiment.a.output
                saved=torch.load(candidate/'Models'/'trainer_state.pth',map_location='cpu',weights_only=False)
                self.assertEqual(saved['epoch'],0); self.assertEqual(saved['best'],[80.,-12.])
                runner.base.pilot.write_json(candidate/'training_complete.json',dict(
                    protocol=protocol,selected_epoch=0))
            with patch.object(runner.base,'evaluate',side_effect=validation),patch.object(runner.base,'train',side_effect=train):
                runner.train_aligned(self.f,a,protocol)
            self.assertEqual(self.f.a.output,root)


if __name__=='__main__': unittest.main()

