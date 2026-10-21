"""Paired-only loss isolation and causal persistent-domain TD boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_final_enhancement as fixtures
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EnhancementAgent
from enhancements.sequence import next_sequence_features
from enhancements.state import build_state
from enhancements.stream import episode


class QueryCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def agent(self):
        torch.manual_seed(42)
        *_,a,domain,anchors,parent=fixtures.FinalEnhancementTests().fixture(query='policy',policy='learned')
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1,
            policy_state=parent.policy.state_dict(),query_value_only=True,paired_query_only=True)
        return agent

    def test_absolute_td_rewards_cannot_change_paired_only_gradient(self):
        left,right=self.agent(),self.agent()
        state=torch.randn(1,71); state[:,-3:]=torch.tensor([-1.,1.,.3])
        for agent,reward in ((left,-100.),(right,100.)):
            for _ in range(32):
                agent.push(state,2,reward,state,False)
                agent.push_pair(state,2,.1,state,state,True)
            agent.learn_td()
        for key,value in left.policy.state_dict().items():
            self.assertTrue(torch.equal(value,right.policy.state_dict()[key]),key)
        self.assertEqual(left.paired_target_totals['future'],0.)
        self.assertAlmostEqual(left.paired_target_totals['target']/32,.1,places=6)
        self.assertNotEqual(left.paired_target_totals['absolute_loss'],right.paired_target_totals['absolute_loss'])

    def test_discounted_difference_is_zero_only_for_terminal_pairs(self):
        agent=self.agent(); state=torch.randn(1,71); state[:,-3:]=torch.tensor([-1.,1.,.3])
        actual,skipped=state.clone(),state.clone(); actual[:,-1]=.8; skipped[:,-1]=.2
        for i in range(32):
            agent.push(state,2,1.,state,False)
            agent.push_pair(state,2,.1,actual,skipped,i%2==0)
        with patch.object(agent,'_next_values',side_effect=lambda s:10*s[:,-1]): agent.learn_td()
        stats=agent.paired_target_totals
        self.assertEqual(stats['terminal'],16)
        self.assertAlmostEqual(stats['future']/32,agent.gamma*6/2,places=5)
        self.assertAlmostEqual(stats['target']/32,.1+agent.gamma*6/2,places=5)

    def test_boundary_successor_matches_actual_next_level_state(self):
        torch.manual_seed(12)
        cnn,bank,initial,a,domain,anchors,parent=fixtures.FinalEnhancementTests().fixture(query='policy',policy='learned')
        bank=bank[:1]; a.budget_fraction=1.; a.eval_epsilon=0.; a.connected_sequence_td=True; a.levels=[0,1]
        following=(domain[0]+.2,1-domain[1],domain[2]+100)
        domains={0:domain,1:following}
        features=next_sequence_features(a,domains,0,a.seed)
        changed={0:domain,1:(following[0],np.full_like(following[1],99),following[2])}
        self.assertTrue(torch.equal(features[0],next_sequence_features(a,changed,0,a.seed)[0]))
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
            policy_state=parent.policy.state_dict(),slots=1)
        forward=agent.policy.forward_for_action
        def always_query(state):
            values,logits,raw,update=forward(state)
            values=values.clone(); values[:,2]=values[:,:2].max(1).values+100.
            return values,logits,raw,update
        continuation={}
        with patch.object(agent.policy,'forward_for_action',always_query):
            m,trace=episode(a,cnn,bank,initial,agent,domain,anchors,continuation=continuation,
                next_domain_features=features)
            pair=agent.pair_memory[-1]
            self.assertEqual(pair[5],0.)
            self.assertTrue(trace[-1]['td_boundary_successor'])
            self.assertFalse(trace[-1]['td_terminal'])
            self.assertEqual(float(pair[3][-2]),1.)
            self.assertEqual(float(pair[3][-1]),0.)
            captured=[]
            def capture(*args,**kwargs):
                state=build_state(*args,**kwargs)
                if args[5]==0: captured.append(state.detach().cpu().reshape(-1).clone())
                return state
            with patch('enhancements.stream.build_state',capture):
                _,last=episode(a,cnn,bank,initial,agent,following,anchors,continuation=continuation)
            difference=(pair[3]-captured[0]).abs()
            self.assertTrue(torch.allclose(pair[3],captured[0],atol=1e-6,rtol=1e-6),
                f'max difference {float(difference.max())}; indices {torch.where(difference>1e-6)[0].tolist()}')
            self.assertEqual(agent.pair_memory[-1][5],1.)
            self.assertTrue(last[-1]['td_terminal'])
            self.assertFalse(last[-1]['td_boundary_successor'])


if __name__=='__main__': unittest.main()
