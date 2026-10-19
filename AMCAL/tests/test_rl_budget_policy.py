"""CPU checks for causal seven-action learned acquisition and paired credit."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as m
from baselines.rl_budget import BudgetState, STATE_WIDTH, GLOBAL_WIDTH, SLOT_WIDTH
from test_structured_bank_protocol import TinyCNN, TinyContext


class BudgetStateTests(unittest.TestCase):
    def test_no_reward_for_deficit_or_unused_budget_without_transfer(self):
        b=BudgetState([.8]*4)
        for pos in range(12):b.observe_paid([0]*4,0,.2,pos)
        weak=b.query_adjustment(0,0,0,3,20,15,100)
        self.assertGreater(weak['need'],0)
        self.assertLess(weak['utility'],0)
        useful=b.query_adjustment(0,.05,.01,3,20,15,100)
        bad=b.query_adjustment(0,0,.05,3,20,15,100)
        self.assertGreater(useful['utility'],0)
        self.assertLess(bad['utility'],weak['utility'])

    def test_soft_price_responds_without_forcing_queries(self):
        fast=BudgetState([.8]*4);slow=BudgetState([.8]*4)
        slow.price=.03
        for pos in range(10):
            fast.advance(True,pos,20,pos,100)
            slow.advance(False,0,20,pos,100)
        self.assertGreater(fast.price,0)
        self.assertLess(slow.price,.03)
        self.assertFalse(hasattr(fast,'decide'))

    def test_propensity_weights_are_capped_and_support_is_not_inflated(self):
        b=BudgetState([.8]*4)
        self.assertEqual(b.observe_paid([0,1,0,1],1,.001,0),20)
        b.observe_paid([1,1,1,1],0,1,1)
        _,_,support,_,_,_=b.estimates(2,100)
        self.assertLessEqual(support,2)
        self.assertGreater(support,1)
        self.assertEqual(len(b.features(1,20,2,100)),STATE_WIDTH)
        with self.assertRaises(ValueError):b.observe_paid([1]*4,0,0,1)
        for kwargs in ({'window':0},{'price_step':float('nan')},{'propensity_clip':.5}):
            with self.assertRaises(ValueError):BudgetState([.5]*4,**kwargs)


class BudgetPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);m.core.device=torch.device('cpu')

    def fixture(self):
        torch.manual_seed(12)
        cnn=TinyCNN().eval()
        for p in cnn.parameters():p.requires_grad_(False)
        bank=[TinyContext().eval() for _ in range(3)]
        initial=deepcopy(bank[0].state_dict())
        a=SimpleNamespace(seed=42,classes=['0','1'],context_lr=.001,budget_fraction=.25,
            query_schedule='policy',update_policy='learned',bank_utility=False,
            train_epsilon=.2,eval_epsilon=0.,gain_weight=1.,retention_weight=1.,selection_cost=.0001,
            budget_aware_policy=True,budget_window=64,budget_need_weight=1.,budget_price_step=.005,
            budget_price_max=.2,budget_propensity_clip=20.,budget_priors=[.8]*4)
        anchors=torch.randn(24,1,8)
        a.budget_anchors=(anchors,np.arange(24)%2)
        domain=torch.randn(48,1,8),np.arange(48)%2,np.arange(100,148)
        agent=m.Agent(34,torch.device('cpu'),42,gamma=.99,budget_aware=True)
        with torch.no_grad():
            # Frozen flag is route feature 25. Prefer frozen prediction while
            # asking to update a Context: this must be a legal combination.
            for p in agent.policy.correctness.parameters():p.zero_()
            agent.policy.correctness[0].weight[0,25]=4
            agent.policy.correctness[2].weight[0,0]=1
            agent.policy.correctness[4].weight[0,0]=1
            agent.policy.update_value[-1].bias.fill_(.3)
        agent.target.load_state_dict(agent.policy.state_dict())
        return cnn,bank,initial,a,domain,agent

    def test_query_value_has_no_weak_destination_prediction_penalty(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        history=m.Reliability(2);history.budget=BudgetState(a.budget_priors)
        z,h=m.base_cache(cnn,domain[0][:1])
        state=m.build_state(m.route_outputs(cnn,bank,domain[0][:1],z),h,history,0,12,0,48,-1)
        with torch.no_grad():
            q=agent.policy(state)[0]
            self.assertAlmostEqual(float(q[4]-q[0]),.3,places=5)
            self.assertGreater(float(q[0]),float(q[1]))
        self.assertEqual(m.action_route(4,q,True),0)
        self.assertEqual(m.action_route(4,q,False),1)
        self.assertEqual(torch.isfinite(m.masked_q(q[None],state)).sum().item(),7)
        state[:,-2]=0
        self.assertEqual(torch.isfinite(m.masked_q(q[None],state)).sum().item(),4)

    def test_permutation_and_budget_features_do_not_change_route_scorer(self):
        _,_,_,_,_,agent=self.fixture()
        state=torch.randn(6,4*34+STATE_WIDTH+3);state[:,-3]=-1;state[:,-2]=.5
        changed=state.clone();changed[:,4*34:-3]+=10
        self.assertTrue(torch.equal(agent.policy.route_logits(state),agent.policy.route_logits(changed)))
        perm=[0,3,1,2];extra=state[:,4*34:-3]
        moved=torch.cat([state[:,:4*34].reshape(-1,4,34)[:,perm].flatten(1),
            extra[:,:GLOBAL_WIDTH],extra[:,GLOBAL_WIDTH:].reshape(-1,4,SLOT_WIDTH)[:,perm].flatten(1),state[:,-3:]],1)
        self.assertTrue(torch.allclose(agent.policy(moved),agent.policy(state)[:,[0,3,1,2,6,4,5]],atol=1e-5))

    def test_paired_successors_preserve_skip_budget_and_prediction(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        frozen=deepcopy(cnn.state_dict())
        metrics,trace=m.episode(a,cnn,bank,initial,agent,domain)
        self.assertEqual(metrics['labels'],metrics['label_cap'])
        self.assertEqual(metrics['label_cap'],12)
        self.assertGreater(metrics['td_updates'],0)
        for row in trace:
            if row['queried']:
                self.assertEqual(row['route'],0)
                self.assertEqual(row['update_slot'],row['action']-4)
                self.assertGreater(row['query_propensity'],0)
            if row['labels_used']==metrics['label_cap'] and not row['queried']:self.assertLess(row['action'],4)
        for state,action,utility,actual,skip,terminal,_ in agent.pair_memory:
            if not terminal:
                self.assertAlmostEqual(float(skip[-2]-actual[-2]),1/12,places=6)
                self.assertEqual(float(actual[-1]),float(skip[-1]))
                self.assertEqual(float(actual[-3]),-1)
                self.assertEqual(float(skip[-3]),-1)
        for k,v in frozen.items():self.assertTrue(torch.equal(v,cnn.state_dict()[k]))

    def test_unpaid_labels_and_endpoint_observers_do_not_change_learning(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        snapshot=deepcopy(agent.policy.state_dict())
        first,trace=m.episode(a,cnn,bank,initial,agent,domain)
        weights=deepcopy(agent.policy.state_dict());bank_weights=[deepcopy(c.state_dict()) for c in bank]
        paid={r['source_row'] for r in trace if r['queried']}
        labels=domain[1].copy()
        labels[[i for i,v in enumerate(domain[2]) if v not in paid]]=99
        other=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot,gamma=.99,budget_aware=True)
        second,trace2=m.episode(a,cnn,bank,initial,other,(domain[0],labels,domain[2]),same_level=True)
        self.assertEqual(first['rl_budget'],second['rl_budget'])
        keys=['action','prediction','reward','queried','query_propensity','update_slot']
        self.assertEqual([[r[k] for k in keys] for r in trace],[[r[k] for k in keys] for r in trace2])
        for k,v in weights.items():self.assertTrue(torch.equal(v,other.policy.state_dict()[k]))
        for saved,c in zip(bank_weights,bank):
            for k,v in saved.items():self.assertTrue(torch.equal(v,c.state_dict()[k]))

    def test_references_exclude_current_duplicates_and_use_no_unpaid_rows(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        current=domain[0][:1]
        anchor=torch.cat([current,current+1,current+2])
        paid=m.core.PaidHistory(1)
        paid.add(current,torch.tensor([0]))
        paid.add(current+3,torch.tensor([1]))
        probe=m.MixedBudgetProbe((anchor,np.array([0,1,0])),paid,2)
        bx,by=probe.sample(current)
        self.assertFalse(any(torch.equal(row,current[0]) for row in bx))
        self.assertEqual(len(bx),3)
        before=m.budget_probe_before(cnn,bank,agent,(bx,by),
            self.history(a),0,12,0,48)
        self.assertFalse(before[0].requires_grad)
        self.assertFalse(before[1].requires_grad)
        gain,harm=m.budget_probe_after(cnn,bank,0,(bx,by),before)
        self.assertAlmostEqual(gain,0.,places=7);self.assertAlmostEqual(harm,0.,places=7)

    def history(self,a):
        history=m.Reliability(2);history.budget=BudgetState(a.budget_priors)
        return history

    def test_zero_budget_and_external_gate_rejection(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        a.budget_fraction=0
        metrics,trace=m.episode(a,cnn,bank,initial,agent,domain)
        self.assertEqual(metrics['labels'],0)
        self.assertTrue(all(r['action']<4 for r in trace))
        self.assertEqual(agent.route_updates,0)
        a.query_schedule='quality'
        with self.assertRaisesRegex(ValueError,'no external gate'):
            m.episode(a,cnn,bank,initial,agent,domain)

    def test_paired_learning_prefers_useful_updates_and_rejects_harmful_ones(self):
        cnn,bank,initial,a,domain,template=self.fixture()
        with torch.no_grad():template.policy.update_value[-1].bias.zero_()
        snapshot=deepcopy(template.policy.state_dict())
        z,h=m.base_cache(cnn,domain[0][:1])
        state=m.build_state(m.route_outputs(cnn,bank,domain[0][:1],z),h,self.history(a),0,12,0,48,-1)
        for utility in (-.15,.15):
            agent=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot,gamma=.99,budget_aware=True)
            route_weights=deepcopy(agent.policy.correctness.state_dict())
            rewards=2*agent.policy.route_logits(state).sigmoid()[0].detach()-1
            for repeat in range(40):
                for r in range(4):agent.push(state,r,float(rewards[r]),torch.zeros_like(state),True)
                agent.push(state,4,float(rewards.max())+utility,torch.zeros_like(state),True)
                agent.push_pair(state,4,utility,torch.zeros_like(state),torch.zeros_like(state),True)
            for _ in range(70):agent.learn_td()
            q=agent.policy(state)[0]
            self.assertGreater(float((q[4]-q[:4].max()).detach())*utility,0)
            for k,v in route_weights.items():self.assertTrue(torch.equal(v,agent.policy.correctness.state_dict()[k]))

    def test_short_training_saves_v2_and_never_scores_test(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        a.levels=[0,1,3];a.gamma=.99;a.router_warmup_episodes=1;a.epochs=1;a.patience=1
        requested=[]
        validation=(domain[0]+.1,domain[1],domain[2]+1000)
        def read(root,artifact,level,split):
            requested.append(split)
            return domain if split=='train_ids' else validation
        with tempfile.TemporaryDirectory() as folder:
            a.output=Path(folder);a.burst_dir=Path(folder);models=a.output/'Models';models.mkdir()
            for name in ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib'):
                (models/name).write_text('unchanged')
            for level in a.levels:(a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').write_text('domain')
            torch.save(dict(route_width=34,policy_state_dict=agent.policy.state_dict()),models/'structured_bank_ddqn.pth')
            with patch.object(m,'models_from_source',return_value=(cnn,bank,initial,34)), \
                 patch.object(m,'read_domain',side_effect=read),redirect_stdout(io.StringIO()):
                m.train(a,models,{})
            saved=torch.load(models/'structured_bank_ddqn.pth',weights_only=False)
            self.assertEqual(saved['version'],2)
            self.assertEqual(saved['actions'],7)
            self.assertTrue(saved['config']['budget_aware_policy'])
            self.assertTrue(set(saved['budget_anchor_source_rows'])<=set(domain[2]))
            self.assertNotIn('test_ids',requested)
            for name in ('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib'):
                self.assertEqual((models/name).read_text(),'unchanged')
            self.assertEqual(len(json.loads((a.output/'structured_training.json').read_text())['epochs']),1)

    def test_new_checkpoint_evaluates_without_val_labels_in_priors(self):
        cnn,bank,initial,a,domain,agent=self.fixture()
        a.levels=[5];a.modes=['learned','random'];a.gamma=.99
        clean_x=a.budget_anchors[0];clean_y=a.budget_anchors[1];clean_ids=np.arange(len(clean_y))
        def read(root,artifact,level,split):
            return (clean_x,clean_y,clean_ids) if split=='train_ids' else domain
        config={**vars(a),**{'budget_'+k:v for k,v in m.budget_settings(a).items()}}
        with tempfile.TemporaryDirectory() as folder:
            a.output=Path(folder);a.burst_dir=Path(folder);models=a.output/'Models';models.mkdir()
            torch.save(dict(version=2,budget_version=1,model_hashes={},training_levels=[0,1,3],route_width=34,
                policy_state_dict=agent.policy.state_dict(),budget_priors=a.budget_priors,
                budget_anchor_source_rows=clean_ids.tolist(),config=config),models/'structured_bank_ddqn.pth')
            with patch.object(m,'models_from_source',return_value=(cnn,bank,initial,34)), \
                 patch.object(m,'read_domain',side_effect=read),redirect_stdout(io.StringIO()) as output:
                m.evaluate(a,models,{})
            self.assertIn('no external query gate',output.getvalue())
            self.assertIn('Online runtime',output.getvalue())
            files=list((a.output/'Results').glob('*_metrics.json'));self.assertEqual(len(files),2)
            for path in files:
                result=json.loads(path.read_text())
                self.assertTrue(result['execution_config']['budget_aware_policy'])
                self.assertEqual(result['rl_budget']['reference_accuracy'],.8)
                self.assertLessEqual(result['labels'],result['label_cap'])
            a.budget_need_weight=2
            with self.assertRaisesRegex(ValueError,'differ from training'):m.evaluate(a,models,{})


if __name__=='__main__':unittest.main()
