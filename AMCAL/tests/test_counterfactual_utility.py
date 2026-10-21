"""Isolation, Adam fidelity and label availability of TRAIN-only alternatives."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import DeviceReliability, EnhancementAgent
from enhancements.counterfactual import alternatives, context_step
from enhancements.memory import ProtectedMemory, transfer
from enhancements.models import BankExecutor
from enhancements.stream import episode
import test_final_enhancement as fixtures


class CounterfactualUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def equal_nested(self, left, right):
        if isinstance(left, torch.Tensor): self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray): self.assertTrue(np.array_equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left: self.equal_nested(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a,b in zip(left,right): self.equal_nested(a,b)
        else: self.assertEqual(left,right)

    def test_virtual_adam_matches_real_update_without_mutating_live_state(self):
        cnn, bank, initial, a, domain, anchors, agent = fixtures.FinalEnhancementTests().fixture(policy='learned')
        x = domain[0]; label = torch.tensor([int(domain[1][0])])
        frozen, embedding = legacy.base_cache(cnn,x[:2])
        optimizers = [torch.optim.Adam(c.parameters(),lr=a.context_lr,weight_decay=1e-4) for c in bank]
        # Nonempty, unequal Adam moments ensure copies reproduce optimizer history.
        for slot in range(3):
            context_step(a,cnn,bank[slot],optimizers[slot],x[1:2],1-label,frozen[1:2],None)
        memory = ProtectedMemory(anchors,2,a.config,71)
        batch = memory.sample(x[:1]); replay = batch.protected(2,memory.replay_rng)
        history = DeviceReliability(2,x.device)
        executor = BankExecutor(cnn,bank)
        outputs = executor.outputs(x[:1],frozen[:1])
        next_outputs = executor.outputs(x[1:2],frozen[1:2])
        live = deepcopy(([c.state_dict() for c in bank], [o.state_dict() for o in optimizers],
                         history.__dict__, memory.__dict__, cnn.state_dict()))
        rng = torch.get_rng_state().clone()
        u, successor = alternatives(a,cnn,bank,optimizers,x[:1],label,outputs,batch,replay,
            history,0,8,0,16,next_outputs,x[1:2],embedding[1:2],0)
        self.equal_nested(live[:3],([c.state_dict() for c in bank],
                          [o.state_dict() for o in optimizers],history.__dict__))
        self.equal_nested(live[4],cnn.state_dict())
        self.equal_nested(live[3]['rng'].bit_generator.state,memory.rng.bit_generator.state)
        self.equal_nested(live[3]['replay_rng'].bit_generator.state,memory.replay_rng.bit_generator.state)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for slot in range(3):
            before = legacy.core.probe_losses(cnn,bank[slot],(batch.x,batch.y))
            context_step(a,cnn,bank[slot],optimizers[slot],x[:1],label,frozen[:1],replay)
            after = legacy.core.probe_losses(cnn,bank[slot],(batch.x,batch.y))
            real,_,_ = transfer(before,after,batch,a.gain_weight,a.retention_weight,a.selection_cost,a.config)
            self.assertAlmostEqual(u[slot],real,places=8)
            h = deepcopy(history); h.observe_paid(outputs[0].softmax(2),int(label[0]))
            changed = executor.refresh_output(x[:1],outputs,slot)
            change=(changed[0][:,slot+1].softmax(1)-outputs[0][:,slot+1].softmax(1)).abs().mean()
            h.updated(slot+1,1,change)
            nxt = executor.refresh_output(x[1:2],next_outputs,slot)
            expected = legacy.build_state(nxt,embedding[1:2],h,1,8,1,16,0)
            self.assertTrue(torch.allclose(successor[slot:slot+1],expected,atol=1e-7))

    def test_training_changes_only_update_head_and_ignores_unqueried_labels(self):
        cnn,bank,initial,a,domain,anchors,agent = fixtures.FinalEnhancementTests().fixture(policy='learned')
        a.counterfactual_utility = True
        start = deepcopy(agent.policy.state_dict()); target = deepcopy(agent.target.state_dict())
        m,trace = episode(a,cnn,bank,initial,agent,domain,anchors,training=True,paid_supervision=True)
        self.assertEqual(m['counterfactual_rows'],m['labels'])
        self.assertEqual(m['counterfactual_context_steps'],3*m['labels'])
        self.assertGreater(agent.counterfactual_updates,0)
        self.assertLess(m['counterfactual_selected_max_error'],1e-7)
        self.assertEqual(m['route_updates'],0); self.assertEqual(m['td_updates'],0)
        after = deepcopy(agent.policy.state_dict()); bank_after = deepcopy([c.state_dict() for c in bank])
        self.assertTrue(any(not torch.equal(start[k],v) for k,v in after.items() if k.startswith('update_value.')))
        for key,value in after.items():
            if not key.startswith('update_value.'): self.assertTrue(torch.equal(start[key],value))
        self.equal_nested(target,agent.target.state_dict())
        labels = domain[1].copy(); paid=set(m['queried_source_ids'])
        for i,source in enumerate(domain[2]):
            if source not in paid: labels[i]=1-labels[i]
        other = EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start)
        other_m,repeated=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors,
                               training=True,paid_supervision=True)
        self.assertEqual(m['labels'],other_m['labels'])
        for key in ('action','prediction','counterfactual_utilities','reward'):
            self.assertEqual([r[key] for r in trace],[r[key] for r in repeated])
        self.equal_nested(after,other.policy.state_dict())
        self.equal_nested(bank_after,[c.state_dict() for c in bank])

    def test_evaluation_never_simulates_alternatives_even_if_training_flag_is_set(self):
        cnn,bank,initial,a,domain,anchors,agent = fixtures.FinalEnhancementTests().fixture(policy='learned')
        a.counterfactual_utility = True
        with patch('enhancements.stream.alternatives',side_effect=AssertionError('TRAIN only')):
            m,_=episode(a,cnn,bank,initial,agent,domain,anchors)
        self.assertNotIn('counterfactual_rows',m)
        self.assertEqual(agent.counterfactual_updates,0)
        self.assertEqual(m['labels'],m['label_cap'])
        self.assertGreater(m['route_updates'],0)

    def test_terminal_alternatives_have_no_bootstrap_or_successor_label(self):
        cnn,bank,initial,a,domain,anchors,agent = fixtures.FinalEnhancementTests().fixture(policy='learned')
        a.counterfactual_utility=True; a.budget_fraction=1.
        domain=domain[0][:1],domain[1][:1],domain[2][:1]
        m,trace=episode(a,cnn,bank,initial,agent,domain,anchors,training=True,paid_supervision=True)
        self.assertEqual(m['counterfactual_rows'],1)
        row=agent.counterfactual_memory[0]
        self.assertEqual(row[4],1.)
        self.assertEqual(float(row[2].abs().sum()),0.)
        self.assertEqual(len(trace[0]['counterfactual_utilities']),3)


if __name__=='__main__': unittest.main()
