"""Adam-transaction fidelity, reference scope and causal guarded feedback."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
from enhancements.agent import EnhancementAgent
from enhancements.counterfactual import context_step
from enhancements.damage_guard import guarded_context_step
from enhancements.memory import ProtectedMemory
from enhancements.stream import episode
import test_final_enhancement as fixtures


class DamageGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def equal_nested(self,left,right):
        if isinstance(left,torch.Tensor): self.assertTrue(torch.equal(left,right))
        elif isinstance(left,dict):
            self.assertEqual(set(left),set(right))
            for k in left: self.equal_nested(left[k],right[k])
        elif isinstance(left,(list,tuple)):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right): self.equal_nested(a,b)
        else: self.assertEqual(left,right)

    def prepared(self):
        cnn,bank,_,a,domain,anchors,_=fixtures.FinalEnhancementTests().fixture(policy='learned')
        context=bank[0]; optimizer=torch.optim.Adam(context.parameters(),lr=a.context_lr,weight_decay=1e-4)
        x=domain[0]; frozen,_=legacy.base_cache(cnn,x[:2]); label=torch.tensor([int(domain[1][1])])
        context_step(a,cnn,context,optimizer,x[:1],1-label,frozen[:1],None)
        memory=ProtectedMemory(anchors,2,a.config,71); batch=memory.sample(x[1:2])
        replay=batch.protected(2,memory.replay_rng)
        before=legacy.core.probe_losses(cnn,context,(batch.x,batch.y))
        return a,cnn,context,optimizer,x[1:2],label,frozen[1:2],replay,batch,before

    def test_full_acceptance_matches_original_adam_step_with_existing_moments(self):
        args=list(self.prepared()); a,cnn,c,opt,x,y,z,replay,batch,before=args
        a.config=replace(a.config,damage_limit=1.)
        expected,eo=deepcopy((c,opt)); context_step(a,cnn,expected,eo,x,y,z,replay)
        info,after=guarded_context_step(*args)
        self.assertTrue(info['accepted']); self.assertEqual(info['scale'],1.)
        self.assertEqual(info['attempts'],1)
        self.equal_nested(c.state_dict(),expected.state_dict())
        self.equal_nested(opt.state_dict(),eo.state_dict())
        self.assertTrue(np.isfinite(after).all())

    def test_backtracking_restarts_original_adam_instead_of_accumulating_steps(self):
        args=list(self.prepared()); a,cnn,c,opt,x,y,z,replay,batch,_=args
        args[-1]=np.ones(len(batch.y))
        expected,eo=deepcopy((c,opt)); rate=eo.param_groups[0]['lr']; eo.param_groups[0]['lr']=rate*.5
        context_step(a,cnn,expected,eo,x,y,z,replay); eo.param_groups[0]['lr']=rate
        with patch('enhancements.damage_guard.legacy.core.probe_losses',
                   side_effect=[np.full(len(batch.y),2.),np.ones(len(batch.y))]):
            info,_=guarded_context_step(*args)
        self.assertEqual(info['scale'],.5); self.assertEqual(info['attempts'],2)
        self.equal_nested(c.state_dict(),expected.state_dict())
        self.equal_nested(opt.state_dict(),eo.state_dict())

    def test_rejection_preserves_live_weights_gradients_adam_and_rng(self):
        args=list(self.prepared()); _,cnn,c,opt,_,_,_,_,batch,_=args
        args[-1]=np.ones(len(batch.y))
        saved=deepcopy((c.state_dict(),opt.state_dict(),[p.grad for p in c.parameters()],cnn.state_dict()))
        rng=torch.get_rng_state().clone()
        with patch('enhancements.damage_guard.legacy.core.probe_losses',return_value=np.full(len(batch.y),2.)):
            info,after=guarded_context_step(*args)
        self.assertFalse(info['accepted']); self.assertEqual(info['attempts'],3)
        self.equal_nested(saved,(c.state_dict(),opt.state_dict(),[p.grad for p in c.parameters()],cnn.state_dict()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(np.array_equal(after,args[-1]))

    def test_guard_uses_old_reference_not_recent_rows(self):
        args=list(self.prepared()); batch=args[-2]
        batch.segments[-2:]=2; args[-1]=np.ones(len(batch.y))
        after=np.ones(len(batch.y)); after[-2:]=100.
        with patch('enhancements.damage_guard.legacy.core.probe_losses',return_value=after):
            info,_=guarded_context_step(*args)
        self.assertTrue(info['accepted']); self.assertEqual(info['scale'],1.)
        self.assertEqual(info['accepted_harm'],0.)
        self.assertEqual(info['protected_rows'],len(batch.y)-2)

    def test_missing_protected_reference_rejects_without_advancing_optimizer(self):
        args=list(self.prepared()); c,opt=args[2:4]; before=deepcopy((c.state_dict(),opt.state_dict()))
        args[-2]=None; args[-1]=None
        info,after=guarded_context_step(*args)
        self.assertFalse(info['accepted']); self.assertEqual(info['attempts'],0)
        self.assertEqual(info['reason'],'no_protected_reference'); self.assertIsNone(after)
        self.equal_nested(before,(c.state_dict(),opt.state_dict()))

    def test_guarded_stream_charges_labels_and_only_versions_committed_updates(self):
        cnn,bank,initial,a,domain,anchors,source=fixtures.FinalEnhancementTests().fixture(policy='learned')
        bank=bank[:1]; a.context_damage_guard=True
        agent=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,
            policy_state=source.policy.state_dict(),slots=1)
        start=deepcopy(agent.policy.state_dict()); continuation={}
        m,trace=episode(a,cnn,bank,initial,agent,domain,anchors,continuation=continuation)
        stats=m['damage_guard']
        self.assertEqual(m['labels'],m['label_cap'])
        self.assertEqual(stats['accepted']+stats['rejected'],m['labels'])
        self.assertEqual(int(agent.versions[0]),32+stats['accepted'])
        self.assertEqual(float(continuation['history'].updates[1]),stats['accepted'])
        self.assertEqual(continuation['history'].paid,m['labels'])
        steps=[float(row['step']) for row in continuation['optimizers'][0].state.values()]
        self.assertEqual(max(steps,default=0),stats['accepted'])
        for row in trace:
            decision=row['damage_guard']
            if decision is None: continue
            if decision['accepted']: self.assertLessEqual(decision['accepted_harm'],a.config.damage_limit)
            else:
                self.assertAlmostEqual(row['harm'],0.)
                self.assertAlmostEqual(row['reward'],(1. if row['truth']==row['prediction'] else -1.)-a.selection_cost)
        after=deepcopy(agent.policy.state_dict()); bank_after=deepcopy(bank[0].state_dict())
        labels=domain[1].copy(); paid=set(m['queried_source_ids'])
        for i,source_id in enumerate(domain[2]):
            if source_id not in paid: labels[i]=1-labels[i]
        other=EnhancementAgent(34,12,2,torch.device('cpu'),42,a.config,policy_state=start,slots=1)
        _,again=episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),anchors)
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in again])
        self.assertEqual([r['damage_guard'] for r in trace],[r['damage_guard'] for r in again])
        self.equal_nested(after,other.policy.state_dict())
        self.equal_nested(bank_after,bank[0].state_dict())


if __name__=='__main__': unittest.main()
