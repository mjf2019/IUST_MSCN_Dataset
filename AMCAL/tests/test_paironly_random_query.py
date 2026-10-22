"""Matched random queries isolate frozen prediction routing from label selection."""
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack
import unittest
import torch
import test_final_enhancement as fixtures
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode


class RandomQueryTests(unittest.TestCase):
    def test_matches_random_queries_context_and_uses_only_frozen_router(self):
        fixtures.FinalEnhancementTests.setUpClass()
        helper=fixtures.FinalEnhancementTests(); helper.setUp()
        cnn, bank, initial, a, domain, anchors, _=helper.fixture(query='policy',policy='learned')
        bank=bank[:1]
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1)
        weights=deepcopy(agent.policy.state_dict())
        random_bank=deepcopy(bank)
        random_agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1)
        random_cont={}; route_cont={}
        with ExitStack() as stack:
            for obj,name in ((agent.policy,'forward_for_action'),(agent,'learn_td'),(agent,'observe_routes')):
                stack.enter_context(patch.object(obj,name,side_effect=AssertionError('No action value or RL learning')))
            # A known frozen route makes querying and prediction explicitly independent.
            stack.enter_context(patch.object(agent.policy,'route_logits',side_effect=lambda s: torch.tensor([[2.,-2.]]).expand(len(s),-1)))
            for _ in range(2):
                m,t=episode(a,cnn,bank,initial,agent,domain,anchors,mode='random_query',
                    continuation=route_cont,reference=anchors,same_level=True)
                rm,rt=episode(a,cnn,random_bank,initial,random_agent,domain,anchors,mode='random',continuation=random_cont)
                self.assertEqual(m['queried_source_ids'],rm['queried_source_ids'])
                self.assertGreater(m['labels'],0)
                self.assertLessEqual(m['labels'],m['label_cap'])
                self.assertEqual(m['td_updates'],0); self.assertEqual(m['route_updates'],0)
                self.assertTrue(all(r['route']==0 for r in t))
                self.assertTrue(all(r['prediction']==r['predictions'][0] for r in t))
                self.assertTrue(any(r['queried'] and r['route']==0 and r['action']==2 for r in t))
                for key,value in bank[0].state_dict().items():
                    self.assertTrue(torch.equal(value,random_bank[0].state_dict()[key]))
            for key,value in weights.items(): self.assertTrue(torch.equal(value,agent.policy.state_dict()[key]))
            a.budget_fraction=0.
            m,t=episode(a,cnn,bank,initial,agent,domain,anchors,mode='random_query')
            self.assertEqual(m['labels'],0)
            self.assertTrue(all(r['route']==0 and not r['queried'] for r in t))

    def test_rejects_training_and_unmatched_schedules(self):
        helper=fixtures.FinalEnhancementTests();helper.setUp()
        cnn,bank,initial,a,domain,anchors,_=helper.fixture(query='policy',policy='learned')
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1)
        with self.assertRaises(ValueError): episode(a,cnn,bank[:1],initial,agent,domain,anchors,mode='random_query',training=True)
        a.query_schedule='uniform'
        with self.assertRaises(ValueError): episode(a,cnn,bank[:1],initial,agent,domain,anchors,mode='random_query')


if __name__=='__main__': unittest.main()
