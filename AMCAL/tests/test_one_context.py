"""Single real Context, three actions and paid-feedback invariants."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EfficientQ, EnhancementAgent
from enhancements.state import masked_q
from enhancements.stream import episode
import test_final_enhancement as fixtures


class OneContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def test_same_shared_weights_three_actions_and_query_masks(self):
        source=EfficientQ(34); model=EfficientQ(34,slots=1)
        model.load_state_dict(source.state_dict(),strict=True)
        states=torch.randn(5,71)
        states[:,-3]=torch.tensor([-1,0,1,2,-1.]); states[:,-2]=1; states[-1,-2]=0
        q=model(states)
        self.assertEqual(tuple(q.shape),(5,3))
        expected=[[0,1,2],[0,1],[2],[2],[0,1]]
        for row,legal in zip(masked_q(q,states,1),expected):
            self.assertEqual(torch.isfinite(row).nonzero().flatten().tolist(),legal)

    def test_one_context_td_learning_cannot_use_unqueried_labels(self):
        cnn,bank,initial,a,domain,anchors,source=fixtures.FinalEnhancementTests().fixture(policy='learned')
        bank=bank[:1]
        domain=(domain[0].repeat(2,1,1),np.tile(domain[1],2),np.arange(32))
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
            policy_state=source.policy.state_dict(),slots=1)
        start=deepcopy(agent.policy.state_dict())
        m,trace=episode(a,cnn,bank,initial,agent,domain,anchors)
        self.assertEqual(m['slot_queries'],[m['label_cap']])
        self.assertEqual(len(m['action_counts']),3)
        self.assertGreater(m['td_updates'],0)
        for row in trace:
            self.assertIn(row['action'],range(3))
            self.assertEqual(len(row['predictions']),2)
            self.assertEqual(len(row['scores']),5)
        after=deepcopy(agent.policy.state_dict()); bank_after=deepcopy(bank[0].state_dict())
        paid=set(m['queried_source_ids']); labels=domain[1].copy()
        for i,source_id in enumerate(domain[2]):
            if source_id not in paid: labels[i]=1-labels[i]
        other=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start,slots=1)
        _,again=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors)
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in again])
        for key,value in after.items(): self.assertTrue(torch.equal(value,other.policy.state_dict()[key]))
        for key,value in bank_after.items(): self.assertTrue(torch.equal(value,bank[0].state_dict()[key]))


if __name__=='__main__': unittest.main()
