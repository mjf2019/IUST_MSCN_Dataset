"""Actual two-Context execution, five actions and shared-checkpoint compatibility."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EfficientQ, EnhancementAgent, DeviceReliability
from enhancements.models import BankExecutor
from enhancements.state import masked_q, build_state
from enhancements.stream import episode, observer
import test_final_enhancement as fixtures


class TwoContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def test_shared_weights_five_actions_and_both_mask_sizes(self):
        source=EfficientQ(34); model=EfficientQ(34,slots=2)
        model.load_state_dict(source.state_dict(),strict=True)
        states=torch.randn(6,3*34+3)
        states[:,-3]=torch.tensor([-1,0,1,2,3,-1.]); states[:,-2]=1; states[-1,-2]=0
        q=model(states)
        self.assertEqual(tuple(q.shape),(6,5))
        masked=masked_q(q,states,2)
        expected=[[0,1,2,3,4],[0,1,2],[3,4],[3],[4],[0,1,2]]
        for row,legal in zip(masked,expected): self.assertEqual(torch.isfinite(row).nonzero().flatten().tolist(),legal)
        q3=torch.randn(6,7)
        self.assertTrue(torch.equal(masked_q(q3,states,3),legacy.masked_q(q3,states)))
        perm=torch.tensor([0,2,1])
        swapped=torch.cat([states[:,:102].reshape(-1,3,34)[:,perm].flatten(1),states[:,-3:]],1)
        qs=model(swapped)
        self.assertTrue(torch.allclose(qs,q[:,[0,2,1,4,3]],atol=1e-6))

    def test_two_real_contexts_learn_without_unqueried_labels_or_extra_routes(self):
        cnn,bank,initial,a,domain,anchors,old=fixtures.FinalEnhancementTests().fixture(policy='learned')
        bank=bank[:2]
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
                               policy_state=old.policy.state_dict(),slots=2)
        start=deepcopy(agent.policy.state_dict())
        m,trace=episode(a,cnn,bank,initial,agent,domain,anchors)
        self.assertEqual(m['labels'],m['label_cap'])
        self.assertEqual(len(m['slot_queries']),2); self.assertEqual(len(m['action_counts']),5)
        self.assertGreater(m['td_updates'],0)
        for row in trace:
            self.assertIn(row['action'],range(5))
            self.assertEqual(len(row['predictions']),3)
            self.assertEqual(len(row['scores']),8)
        after=deepcopy(agent.policy.state_dict()); bank_after=deepcopy([c.state_dict() for c in bank])
        labels=domain[1].copy(); paid=set(m['queried_source_ids'])
        for i,source in enumerate(domain[2]):
            if source not in paid: labels[i]=1-labels[i]
        other=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start,slots=2)
        _,again=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors)
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in again])
        for key,value in after.items(): self.assertTrue(torch.equal(value,other.policy.state_dict()[key]))
        for saved,context in zip(bank_after,bank):
            for key,value in saved.items(): self.assertTrue(torch.equal(value,context.state_dict()[key]))

    def test_state_and_observer_have_only_three_real_routes(self):
        cnn,bank,initial,a,domain,anchors,old=fixtures.FinalEnhancementTests().fixture(policy='learned')
        bank=bank[:2]; agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
                                           policy_state=old.policy.state_dict(),slots=2)
        frozen,embedding=legacy.base_cache(cnn,domain[0])
        executor=BankExecutor(cnn,bank)
        outputs=executor.outputs(domain[0],frozen)
        self.assertEqual(executor.context_forwards,2*len(domain[1]))
        self.assertEqual(tuple(outputs[0].shape),(16,3,2))
        history=DeviceReliability(2,torch.device('cpu'),routes=3)
        state=build_state(outputs,embedding,history,0,8,0,16,-1)
        self.assertEqual(tuple(state.shape),(16,105))
        prediction,_,matrix=observer(cnn,bank,agent,domain[0],domain[1],history,0,8,0,42,a.config)
        self.assertEqual(tuple(matrix.shape),(16,3))
        self.assertTrue(np.isin(prediction,[0,1]).all())


if __name__=='__main__': unittest.main()
