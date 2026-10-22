"""Protected routing replay: causal labels, one optimizer step, unchanged Context update."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
import torch
import test_final_enhancement as fixtures
from enhancements.agent import EnhancementAgent
from enhancements.stream import episode


class ProtectedRoutingTests(unittest.TestCase):
    def setup_fixture(self):
        fixtures.FinalEnhancementTests.setUpClass()
        helper=fixtures.FinalEnhancementTests();helper.setUp()
        config=replace(helper.config(),recent_capacity=2).validated()
        cnn,bank,initial,a,domain,anchors,_=helper.fixture(config,query='uniform',policy='fixed')
        bank=bank[:1]
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,config,slots=1)
        return cnn,bank,initial,a,domain,anchors,agent

    def test_joint_routing_step_changes_only_routing_parameters(self):
        *_,agent=self.setup_fixture()
        before=deepcopy(agent.policy.state_dict());target=deepcopy(agent.target.state_dict())
        state=torch.randn(1,71);states=torch.randn(4,71)
        agent.observe_routes(state,torch.tensor([1.,0.]),protected=(states,
            torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]]),torch.ones(4)))
        self.assertEqual(agent.route_updates,1)
        self.assertEqual(agent.protected_route_updates,1)
        self.assertEqual(agent.protected_route_rows,4)
        self.assertEqual(agent.td_updates,0)
        for k,v in before.items():
            if not k.startswith('correctness.'):self.assertTrue(torch.equal(v,agent.policy.state_dict()[k]),k)
        for k,v in target.items():self.assertTrue(torch.equal(v,agent.target.state_dict()[k]))
        self.assertTrue(any(not torch.equal(v,agent.policy.state_dict()[k]) for k,v in before.items() if k.startswith('correctness.')))

    def test_causal_balanced_replay_preserves_matched_context_and_ignores_unpaid_labels(self):
        cnn,bank,initial,a,domain,anchors,agent=self.setup_fixture()
        original=deepcopy(agent.policy.state_dict());baseline_bank=deepcopy(bank)
        baseline=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1,policy_state=original)
        base_m,_=episode(a,cnn,baseline_bank,initial,baseline,domain,anchors)
        a.router_protected_replay=True
        seen=[];original_observe=agent.observe_routes
        def capture(*args,**kwargs):
            protected=kwargs.get('protected')
            if protected is not None:
                states,correct,weights=protected
                self.assertFalse(states.requires_grad)
                self.assertTrue(bool((weights>0).all()))
                seen.append(len(states))
            return original_observe(*args,**kwargs)
        with patch.object(agent,'observe_routes',side_effect=capture):
            m,t=episode(a,cnn,bank,initial,agent,domain,anchors)
        report=m['protected_routing_replay']
        self.assertEqual(m['queried_source_ids'],base_m['queried_source_ids'])
        self.assertEqual(report['updates'],m['labels'])
        self.assertEqual(m['route_updates'],m['labels'])
        self.assertEqual(report['rows'],sum(seen))
        self.assertEqual(report['rows'],report['source_rows']+report['historical_rows'])
        self.assertGreater(report['source_rows'],0);self.assertGreater(report['historical_rows'],0)
        for k,v in bank[0].state_dict().items():self.assertTrue(torch.equal(v,baseline_bank[0].state_dict()[k]))
        labels=domain[1].copy();paid=set(m['queried_source_ids'])
        for i,source in enumerate(domain[2]):
            if int(source) not in paid:labels[i]=1-labels[i]
        replay=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,slots=1,policy_state=original)
        replay_bank=deepcopy(bank)
        replay_m,replay_t=episode(a,cnn,replay_bank,initial,replay,(domain[0],labels,domain[2]),anchors)
        self.assertEqual([(r['action'],r['prediction']) for r in t],[(r['action'],r['prediction']) for r in replay_t])
        for k,v in agent.policy.state_dict().items():self.assertTrue(torch.equal(v,replay.policy.state_dict()[k]))
        a.budget_fraction=0.
        before=deepcopy(agent.policy.state_dict())
        zero,_=episode(a,cnn,bank,initial,agent,domain,anchors)
        self.assertEqual(zero['protected_routing_replay']['updates'],0)
        for k,v in before.items():self.assertTrue(torch.equal(v,agent.policy.state_dict()[k]))
        with self.assertRaises(ValueError):episode(a,cnn,bank,initial,agent,domain,anchors,training=True)


if __name__=='__main__':unittest.main()
