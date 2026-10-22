"""Level-balanced routing draws preserve original storage and Context dynamics."""
from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import torch
import test_final_enhancement as fixtures
from enhancements.agent import EnhancementAgent
from enhancements.memory import ProtectedMemory
from enhancements.stream import episode


class LevelHistoryTests(unittest.TestCase):
    def setUp(self):
        fixtures.FinalEnhancementTests.setUpClass()
        self.helper=fixtures.FinalEnhancementTests();self.helper.setUp()

    def test_imbalanced_pool_balances_level_then_class_without_touching_reference_rng(self):
        c=replace(self.helper.config(),historical_capacity=64,recent_capacity=0,reference_size=32)
        anchors=(torch.randn(8,1,8),np.arange(8)%2)
        tracked=ProtectedMemory(anchors,2,c,42,track_levels=True)
        original=ProtectedMemory(anchors,2,c,42)
        for i in range(64):
            x=torch.full((1,1,8),float(i+100));y=i%2;level=5 if i<48 else 7
            tracked.add(x,torch.tensor([y]),level=level);original.add(x,torch.tensor([y]))
        current=torch.full((1,1,8),-999.)
        ref=tracked.sample(current);other=original.sample(current)
        self.assertTrue(torch.equal(ref.x,other.x));self.assertTrue(torch.equal(ref.y,other.y))
        before=deepcopy(tracked.rng.bit_generator.state)
        replay_before=deepcopy(tracked.replay_rng.bit_generator.state)
        batch=tracked.router_reference(current,ref)
        self.assertEqual(tracked.rng.bit_generator.state,before)
        self.assertEqual(tracked.replay_rng.bit_generator.state,replay_before)
        levels=batch.levels[batch.segments==1];labels=batch.y.cpu().numpy()[batch.segments==1]
        self.assertEqual(dict(zip(*np.unique(levels,return_counts=True))),{5:4,7:4})
        for level in (5,7):self.assertEqual(dict(zip(*np.unique(labels[levels==level],return_counts=True))),{0:2,1:2})
        self.assertEqual(tracked.counts()['historical_by_level'],{'5':48,'7':16})
        self.assertEqual(len(tracked.levels),64)
        # Current feature-identical rows must never appear in the routing draw.
        current=tracked.historical[0][0][0]
        batch=tracked.router_reference(current,tracked.sample(current))
        self.assertFalse(any(torch.equal(row[None],current) for row in batch.x))
        for actual,expected in zip(tracked.historical,original.historical):
            self.assertEqual([r[2] for r in actual],[r[2] for r in expected])

    def test_level_tags_remain_bounded_and_require_acquired_level(self):
        c=replace(self.helper.config(),historical_capacity=8,recent_capacity=2)
        memory=ProtectedMemory((torch.randn(8,1,8),np.arange(8)%2),2,c,42,track_levels=True)
        with self.assertRaises(ValueError):memory.add(torch.ones(1,1,8),torch.tensor([0]))
        for i in range(200):memory.add(torch.full((1,1,8),float(100+i)),torch.tensor([i%2]),level=i//20)
        self.assertLessEqual(len(memory.levels),10)
        counts=memory.counts()
        self.assertEqual(sum(counts['historical_by_level'].values()),counts['historical'])
        self.assertEqual(sum(counts['recent_by_level'].values()),counts['recent'])

    def test_fixed_queries_keep_contexts_and_ordinary_reference_identical_across_levels(self):
        c=replace(self.helper.config(),historical_capacity=8,recent_capacity=2)
        cnn,bank,initial,a,domain,anchors,_=self.helper.fixture(c,query='uniform',policy='fixed')
        bank=bank[:1];other_bank=deepcopy(bank)
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,c,slots=1)
        other=EnhancementAgent(34,12,2,torch.device('cpu'),42,c,slots=1,policy_state=deepcopy(agent.policy.state_dict()))
        a.router_protected_replay=True;a.router_protected_weight=.4
        b=deepcopy(a);a.router_level_balanced_history=True
        cont={};other_cont={}
        seen=set()
        for i,level in enumerate((5,7,10)):
            seen.add(str(level));stream=(domain[0]+i*2,domain[1],domain[2]+i*100)
            m,_=episode(a,cnn,bank,initial,agent,stream,anchors,continuation=cont,domain_level=level)
            om,_=episode(b,cnn,other_bank,initial,other,stream,anchors,continuation=other_cont)
            self.assertEqual(m['queried_source_ids'],om['queried_source_ids'])
            for key,value in bank[0].state_dict().items():self.assertTrue(torch.equal(value,other_bank[0].state_dict()[key]))
            self.assertEqual(cont['memory'].rng.bit_generator.state,other_cont['memory'].rng.bit_generator.state)
            self.assertEqual(cont['memory'].replay_rng.bit_generator.state,other_cont['memory'].replay_rng.bit_generator.state)
            self.assertTrue(set(m['memory']['historical_by_level']).issubset(seen))
            self.assertEqual(sum(m['router_level_history']['historical_exposures_by_level'].values()),m['protected_routing_replay']['historical_rows'])
        self.assertGreater(m['protected_routing_replay']['historical_rows'],0)
        with self.assertRaises(ValueError):episode(a,cnn,bank,initial,agent,domain,anchors)


if __name__=='__main__':unittest.main()
