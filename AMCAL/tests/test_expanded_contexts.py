"""Four/five real Contexts with inherited shared weights and nine/eleven actions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EfficientQ, EnhancementAgent
from enhancements.state import masked_q
from enhancements.stream import episode
import test_final_enhancement as fixtures


class ExpandedContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def test_shared_weights_shapes_masks_and_context_permutation(self):
        source=EfficientQ(34)
        for count in (4,5):
            with self.subTest(contexts=count):
                model=EfficientQ(34,slots=count)
                model.load_state_dict(source.state_dict(),strict=True)
                states=torch.randn(5,(count+1)*34+3)
                states[:,-3]=torch.tensor([-1,0,1,count+1,-1.]); states[:,-2]=1; states[-1,-2]=0
                q=model(states)
                self.assertEqual(tuple(q.shape),(5,2*count+1))
                skips=list(range(count+1)); queries=list(range(count+1,2*count+1))
                expected=[skips+queries,skips,queries,[2*count],skips]
                for row,legal in zip(masked_q(q,states,count),expected):
                    self.assertEqual(torch.isfinite(row).nonzero().flatten().tolist(),legal)
                permutation=torch.tensor([0,*range(count,0,-1)])
                swapped=torch.cat([states[:,:-3].reshape(-1,count+1,34)[:,permutation].flatten(1),states[:,-3:]],1)
                action_permutation=[0,*range(count,0,-1),*range(2*count,count,-1)]
                self.assertTrue(torch.allclose(model(swapped),q[:,action_permutation],atol=1e-6))

    def test_expanded_online_learning_respects_cap_and_unqueried_labels(self):
        for count in (4,5):
            with self.subTest(contexts=count):
                torch.manual_seed(12)
                cnn,bank,initial,a,domain,anchors,source=fixtures.FinalEnhancementTests().fixture(policy='learned')
                bank.extend(deepcopy(bank[0]) for _ in range(count-len(bank)))
                agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
                    policy_state=source.policy.state_dict(),slots=count)
                start=deepcopy(agent.policy.state_dict())
                m,trace=episode(a,cnn,bank,initial,agent,domain,anchors)
                self.assertEqual(m['labels'],m['label_cap'])
                self.assertEqual(len(m['slot_queries']),count)
                self.assertEqual(len(m['action_counts']),2*count+1)
                self.assertGreater(m['td_updates'],0)
                for row in trace:
                    self.assertIn(row['action'],range(2*count+1))
                    self.assertEqual(len(row['predictions']),count+1)
                    self.assertEqual(len(row['scores']),3*count+2)
                after=deepcopy(agent.policy.state_dict()); bank_after=deepcopy([c.state_dict() for c in bank])
                paid=set(m['queried_source_ids']); labels=domain[1].copy()
                for i,source_id in enumerate(domain[2]):
                    if source_id not in paid: labels[i]=1-labels[i]
                other=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start,slots=count)
                _,again=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors)
                self.assertEqual([r['action'] for r in trace],[r['action'] for r in again])
                for key,value in after.items(): self.assertTrue(torch.equal(value,other.policy.state_dict()[key]))
                for saved,context in zip(bank_after,bank):
                    for key,value in saved.items(): self.assertTrue(torch.equal(value,context.state_dict()[key]))


if __name__=='__main__': unittest.main()
