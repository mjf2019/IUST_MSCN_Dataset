"""CPU checks for structured value semantics, causal feedback and bank controls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import tempfile
import json
import csv
import io
from contextlib import redirect_stdout

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as m


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1=self.conv_block2=self.conv_block3=nn.Identity()
        self.flatten=nn.Flatten()
        self.dense_layers=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,2))

    def forward(self,x):return self.dense_layers(self.flatten(x))


class TinyContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.Linear(8,8)
        nn.init.eye_(self.layer.weight);nn.init.zeros_(self.layer.bias)

    def forward(self,x):return self.layer(x.squeeze(1)).unsqueeze(1)


class StructuredBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        m.core.device=torch.device('cpu')

    def setUp(self):torch.manual_seed(12)

    def state(self,n=8,width=34):
        x=torch.randn(n,4*width+3)
        x[:,-3]=0;x[:,-2]=.5;x[:,-1]=.25
        return x

    def fixture(self):
        cnn=TinyCNN().eval()
        for p in cnn.parameters():p.requires_grad_(False)
        bank=[TinyContext().eval() for _ in range(3)]
        initial=deepcopy(bank[0].state_dict())
        for context in bank:context.load_state_dict(initial)
        a=SimpleNamespace(seed=42,classes=['0','1'],context_lr=.001,budget_fraction=.5,
                          query_schedule='uniform',update_policy='fixed',bank_utility=False,
                          train_epsilon=.2,eval_epsilon=0)
        x=torch.randn(16,1,8);y=np.arange(16)%2
        return cnn,bank,initial,a,(x,y,np.arange(16))

    def test_q_ranking_is_exactly_correctness_ranking(self):
        net=m.StructuredQ(34);x=self.state()
        q=net(x);p=net.route_logits(x).sigmoid()
        self.assertTrue(torch.allclose(q[:,:4]-q[:,:1],2*(p-p[:,:1]),atol=1e-6))

    def test_context_permutation_equivariance(self):
        net=m.StructuredQ(34);x=self.state()
        perm=[0,3,1,2]
        moved=torch.cat([x[:,:-3].reshape(-1,4,34)[:,perm].flatten(1),x[:,-3:]],1)
        q0,q1=net(x),net(moved)
        self.assertTrue(torch.allclose(q1,q0[:,[0,3,1,2,6,4,5]],atol=1e-6))

    def test_no_gradient_cross_talk(self):
        agent=m.Agent(34,torch.device('cpu'),42)
        x=self.state()
        route_before=deepcopy(agent.policy.correctness.state_dict())
        for i in range(40):agent.push(x[i%8],i%4,.5,x[(i+1)%8],False)
        agent.learn_td()
        self.assertEqual(agent.td_updates,1)
        for name,value in route_before.items():self.assertTrue(torch.equal(value,agent.policy.correctness.state_dict()[name]))
        future_before=deepcopy(agent.policy.future.state_dict())
        update_before=deepcopy(agent.policy.update_value.state_dict())
        agent.observe_routes(x[:1],torch.tensor([0,1,1,0]))
        self.assertEqual(agent.route_updates,1)
        self.assertTrue(any(not torch.equal(v,agent.policy.correctness.state_dict()[k]) for k,v in route_before.items()))
        for k,v in future_before.items():self.assertTrue(torch.equal(v,agent.policy.future.state_dict()[k]))
        for k,v in update_before.items():self.assertTrue(torch.equal(v,agent.policy.update_value.state_dict()[k]))

    def test_legal_masks(self):
        for remaining in (0,1):
            for gate in (-1,0,1,2,3,4):
                s=self.state(n=1);s[:,-3]=gate;s[:,-2]=remaining
                legal=torch.isfinite(m.masked_q(torch.zeros(1,7),s))[0].nonzero().flatten().tolist()
                wanted=list(range(4)) if not remaining or gate==0 else list(range(7)) if gate==-1 else [4,5,6] if gate==1 else [gate+2]
                self.assertEqual(legal,wanted)

    def test_fixed_update_control_keeps_identical_banks(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42)
        snapshot=deepcopy(agent.policy.state_dict())
        before=deepcopy(cnn.state_dict())
        learned,trace0=m.episode(a,cnn,bank,initial,agent,domain)
        weights0=[deepcopy(c.state_dict()) for c in bank]
        random=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot)
        other,trace1=m.episode(a,cnn,bank,initial,random,domain,mode='random',learning=False)
        self.assertEqual(learned['labels'],8);self.assertEqual(other['labels'],8)
        self.assertEqual(learned['slot_queries'],other['slot_queries'])
        self.assertEqual(learned['oracle_all'],other['oracle_all'])
        self.assertEqual(learned['oracle_legal'],other['oracle_legal'])
        for saved,context in zip(weights0,bank):
            for k,v in saved.items():self.assertTrue(torch.equal(v,context.state_dict()[k]))
        for k,v in before.items():self.assertTrue(torch.equal(v,cnn.state_dict()[k]))
        for row in trace0:
            self.assertGreaterEqual(row['oracle_all'],row['oracle_legal'])
        self.assertGreaterEqual(learned['oracle_legal'],learned['accuracy'])
        self.assertGreater(learned['route_updates'],0)
        self.assertGreater(learned['td_updates'],0)
        self.assertEqual(other['route_updates'],0)

    def test_unqueried_labels_cannot_change_online_learning(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42)
        snapshot=deepcopy(agent.policy.state_dict())
        _,trace=m.episode(a,cnn,bank,initial,agent,domain)
        first_weights=deepcopy(agent.policy.state_dict())
        queried={r['source_row'] for r in trace if r['queried']}
        y=domain[1].copy()
        for i in range(len(y)):
            if i not in queried:y[i]=1-y[i]
        other=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot)
        _,second=m.episode(a,cnn,bank,initial,other,(domain[0],y,domain[2]))
        self.assertEqual([r['action'] for r in trace],[r['action'] for r in second])
        self.assertEqual([r['prediction'] for r in trace],[r['prediction'] for r in second])
        for k,v in first_weights.items():self.assertTrue(torch.equal(v,other.policy.state_dict()[k]))

    def test_offline_dense_supervision_is_explicit(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42,offline=True)
        metrics,_=m.episode(a,cnn,bank,initial,agent,domain,mode='random',training=True,warmup=True)
        self.assertEqual(len(agent.route_memory),16)
        self.assertEqual(metrics['labels'],8)
        self.assertEqual(len(agent.td_memory),0)

    def test_bank_utility_ignores_redundant_slot_gain(self):
        cnn,bank,initial,a,domain=self.fixture()
        a.bank_utility=True
        inputs=domain[0][:1];label=torch.tensor([0])
        # Frozen already has virtually zero CE; improving a weaker slot adds no envelope gain.
        before=torch.tensor([[1e-8],[10.0],[5.0],[8.0]])
        utility,gain,harm=m.utility_after(a,cnn,bank,0,(inputs,label),before)
        self.assertAlmostEqual(gain,0)
        self.assertAlmostEqual(harm,0)
        self.assertAlmostEqual(utility,-m.core.SELECTION_COST)

    def test_retention_uses_identical_unqueried_rows_and_counts_damage(self):
        y=np.array([0,1,0,1])
        before=np.array([0,1,1,0]);after=np.array([1,1,0,1])
        first=(before,None,np.tile(before[:,None],(1,4)))
        final=(after,None,np.tile(after[:,None],(1,4)))
        result=m.retention_summary(first,final,y,np.array([True,False,True,False]))
        self.assertEqual(result['samples'],2)
        self.assertEqual(result['initial'],.5);self.assertEqual(result['final'],.5)
        self.assertEqual(result['drop_pp'],0)
        self.assertEqual(result['broken'],1);self.assertEqual(result['fixed'],1)
        self.assertEqual(result['per_route_broken'],[1]*4)
        self.assertEqual(result['per_route_fixed'],[1]*4)
        empty=m.retention_summary(first,final,y,np.zeros(4,dtype=bool))
        self.assertEqual(empty['samples'],0);self.assertIsNone(empty['drop_pp'])

    def test_same_level_observer_cannot_change_trajectory_or_weights(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42)
        snapshot=deepcopy(agent.policy.state_dict())
        original,trace=m.episode(a,cnn,bank,initial,agent,domain)
        weights=deepcopy(agent.policy.state_dict())
        bank_weights=[deepcopy(c.state_dict()) for c in bank]
        other=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot)
        metrics,second=m.episode(a,cnn,bank,initial,other,domain,same_level=True)
        self.assertEqual(trace,second)
        self.assertEqual(original['accuracy'],metrics['accuracy'])
        self.assertEqual(metrics['same_level']['all_rows']['samples'],16)
        self.assertEqual(metrics['same_level']['unqueried']['samples'],8)
        self.assertEqual(metrics['samples'],16)
        self.assertEqual(metrics['actual_label_fraction'],.5)
        self.assertEqual(metrics['budget_utilization_pct'],100)
        self.assertEqual(metrics['labels_remaining'],0)
        self.assertEqual(metrics['same_level']['unqueried']['per_route_initial'][0],
                         metrics['same_level']['unqueried']['per_route_final'][0])
        self.assertEqual(metrics['same_level']['unqueried']['per_route_broken'][0],0)
        for k,v in weights.items():self.assertTrue(torch.equal(v,other.policy.state_dict()[k]))
        for saved,context in zip(bank_weights,bank):
            for k,v in saved.items():self.assertTrue(torch.equal(v,context.state_dict()[k]))

    def test_evaluate_saves_readable_same_level_table_and_csv(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42)
        a.levels=[5];a.modes=['learned','random'];a.gamma=.95
        a.selection_cost=.0001;a.gain_weight=1;a.retention_weight=1
        with tempfile.TemporaryDirectory() as directory:
            a.output=Path(directory);a.burst_dir=Path(directory)
            models=a.output/'Models';models.mkdir()
            torch.save(dict(version=1,model_hashes={},training_levels=[0,1,3],
                            route_width=34,policy_state_dict=agent.policy.state_dict()),
                       models/'structured_bank_ddqn.pth')
            with patch.object(m,'models_from_source',return_value=(cnn,bank,initial,34)), \
                 patch.object(m,'read_domain',return_value=domain),redirect_stdout(io.StringIO()) as output:
                m.evaluate(a,models,{})
            self.assertIn('Same-level retention on unqueried rows',output.getvalue())
            self.assertIn('Label budget consumption',output.getvalue())
            self.assertTrue(list((a.output/'Results').glob('*_budget.csv')))
            self.assertIn('Online runtime',output.getvalue())
            self.assertTrue(list((a.output/'Results').glob('*_timing.csv')))
            path=next((a.output/'Results').glob('*_same_level.csv'))
            with path.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),4)
            self.assertEqual({row['Subset'] for row in rows},{'All','Unqueried'})
            metrics=json.loads(next((a.output/'Results').glob('*learned*_metrics.json')).read_text())
            self.assertEqual(metrics['same_level']['unqueried']['samples'],8)
            self.assertGreater(metrics['cache_seconds'],0)
            self.assertGreater(metrics['update_seconds'],0)
            self.assertAlmostEqual(metrics['online_seconds'],sum(metrics[k] for k in
                ('prediction_seconds','selection_seconds','update_seconds','other_seconds')))

    def test_timing_cannot_change_policy_actions_or_bank(self):
        cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42)
        snapshot=deepcopy(agent.policy.state_dict())
        measured,trace=m.episode(a,cnn,bank,initial,agent,domain)
        weights=[deepcopy(c.state_dict()) for c in bank]
        other=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot)
        timer=m.StreamTimer
        with patch.object(m,'StreamTimer',side_effect=lambda device,enabled:timer(device,enabled=False)):
            unmeasured,second=m.episode(a,cnn,bank,initial,other,domain)
        self.assertEqual(trace,second)
        self.assertEqual(measured['accuracy'],unmeasured['accuracy'])
        for before,context in zip(weights,bank):
            for k,v in before.items():self.assertTrue(torch.equal(v,context.state_dict()[k]))


if __name__=='__main__':unittest.main()

