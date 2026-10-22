"""Paid-only route learning with matched random queries; TD heads stay frozen."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
import torch
import test_final_enhancement as fixtures
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode


class OnlineRouterTests(unittest.TestCase):
    def test_only_routing_learns_from_paid_feedback_with_matched_contexts(self):
        fixtures.FinalEnhancementTests.setUpClass()
        helper=fixtures.FinalEnhancementTests();helper.setUp()
        cnn,bank,initial,a,domain,anchors,_=helper.fixture(query='policy',policy='learned')
        bank=bank[:1];other_bank=deepcopy(bank)
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1)
        original=deepcopy(agent.policy.state_dict());target=deepcopy(agent.target.state_dict())
        other=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1,policy_state=original)
        cont={};other_cont={}
        with patch.object(agent,'learn_td',side_effect=AssertionError('No TD')), \
                patch.object(agent.policy,'forward_for_action',side_effect=AssertionError('No action-value inference')):
            m,t=episode(a,cnn,bank,initial,agent,domain,anchors,mode='random_query_online',continuation=cont)
            rm,_=episode(a,cnn,other_bank,initial,other,domain,anchors,mode='random_query',continuation=other_cont)
            self.assertGreater(m['labels'],0)
            self.assertEqual(m['route_updates'],m['labels'])
            self.assertEqual(m['td_updates'],0)
            self.assertEqual(m['queried_source_ids'],rm['queried_source_ids'])
            for k,v in bank[0].state_dict().items(): self.assertTrue(torch.equal(v,other_bank[0].state_dict()[k]))
            paid=set(m['queried_source_ids']);labels=domain[1].copy()
            for i,source in enumerate(domain[2]):
                if int(source) not in paid: labels[i]=1-labels[i]
            # A fresh learner must be insensitive to all unpaid TEST labels.
            replay_agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1,policy_state=original)
            replay_bank=deepcopy(other_bank)
            replay_m,replay_t=episode(a,cnn,replay_bank,initial,replay_agent,
                (domain[0],labels,domain[2]),anchors,mode='random_query_online')
            self.assertEqual([(r['action'],r['prediction']) for r in t],[(r['action'],r['prediction']) for r in replay_t])
            for k,v in agent.policy.state_dict().items(): self.assertTrue(torch.equal(v,replay_agent.policy.state_dict()[k]))
            for k,v in original.items():
                if not k.startswith('correctness.'):
                    self.assertTrue(torch.equal(v,agent.policy.state_dict()[k]),k)
            self.assertTrue(any(not torch.equal(v,agent.policy.state_dict()[k]) for k,v in original.items() if k.startswith('correctness.')))
            for k,v in target.items():self.assertTrue(torch.equal(v,agent.target.state_dict()[k]))
            m2,_=episode(a,cnn,bank,initial,agent,domain,anchors,mode='random_query_online',continuation=cont)
            self.assertEqual(m2['route_updates'],m2['labels'])
            a.budget_fraction=0.
            before=deepcopy(agent.policy.state_dict())
            zero,_=episode(a,cnn,bank,initial,agent,domain,anchors,mode='random_query_online')
            self.assertEqual(zero['route_updates'],0)
            for k,v in before.items():self.assertTrue(torch.equal(v,agent.policy.state_dict()[k]))


if __name__=='__main__':unittest.main()
