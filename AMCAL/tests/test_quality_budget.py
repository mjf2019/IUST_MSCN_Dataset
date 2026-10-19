"""Quality feedback scheduling, strict cap, causality and router integration."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import csv
import io
import json
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.quality_budget import QualityBudget


class QualityBudgetTests(unittest.TestCase):
    def rollout(self,correct=False,gain=.02,harm=.001,rows=886,cap=177):
        controller=QualityBudget(rows,cap,.7,seed=42)
        used=0;positions=[]
        for position in range(rows):
            d=controller.decide(position,used)
            if d['query']:
                if d['reason']=='adaptive':
                    self.assertLess(used,d['released'])
                    self.assertLess(used,cap-d['reserved_future'])
                controller.observe(position,correct,gain,harm)
                used+=1;positions.append(position)
            self.assertLessEqual(used,cap)
        self.assertTrue(controller.monitor.issubset(set(positions)))
        return controller,positions

    def test_weak_and_useful_uses_more_than_good_or_harmful(self):
        useful,positions=self.rollout()
        good,_=self.rollout(correct=True)
        harmful,_=self.rollout(gain=0,harm=.02)
        self.assertEqual(useful.labels,177)
        self.assertEqual(good.labels,len(good.monitor))
        self.assertLess(harmful.labels,useful.labels)
        self.assertLessEqual(harmful.labels,len(harmful.monitor)+harmful.min_return_updates)
        self.assertGreaterEqual(positions[-1],int(.9*886))
        self.assertEqual(useful.counts['bootstrap'],12)
        self.assertEqual(useful.counts['audit'],18)

    def test_small_and_zero_caps_never_overshoot(self):
        for n in (1,7,35):
            for cap in (0,1,int(n*.2),n):
                controller,positions=self.rollout(rows=n,cap=cap)
                self.assertLessEqual(len(positions),cap)
                self.assertEqual(controller.labels,len(positions))

    def test_paid_monitors_only_measure_quality_and_history_only_yield(self):
        c=QualityBudget(100,20,.8,bootstrap_labels=2,audit_fraction=0,min_return_updates=1)
        c.monitor_correct.append(0.)
        for p in range(100):
            d=c.decide(p,c.labels)
            if d['query']:
                prior=list(c.monitor_correct)
                c.observe(p,True,None,None)
                if d['reason']=='adaptive':self.assertEqual(prior,list(c.monitor_correct))
        self.assertEqual(len(c.returns),0)
        self.assertIsNone(c.statistics()['mean_net_transfer'])

    def test_plateau_reduces_requests_but_keeps_future_monitoring(self):
        plateau,positions=self.rollout(gain=.0001,harm=0)
        self.assertEqual(plateau.statistics()['efficiency'],0.)
        self.assertTrue(plateau.audits.issubset(positions))
        self.assertLess(plateau.labels,177)

    def test_unmeasurable_yield_has_bounded_trial_and_keeps_monitors(self):
        controller,positions=self.rollout(gain=None,harm=None)
        self.assertIsNone(controller.statistics()['mean_net_transfer'])
        self.assertEqual(controller.statistics()['efficiency'],0.)
        self.assertLessEqual(controller.labels,len(controller.monitor)+controller.trial_labels)
        self.assertLess(controller.labels,177)
        self.assertTrue(controller.monitor.issubset(positions))

    def test_decisions_are_causal_cached_and_feedback_must_be_paid(self):
        a=QualityBudget(100,20,.7);b=QualityBudget(100,20,.7)
        for p in range(100):
            da,db=a.decide(p,a.labels),b.decide(p,b.labels)
            self.assertEqual(da,db)
            self.assertIs(a.decide(p,a.labels),da)
            if da['query']:
                with self.assertRaises(RuntimeError):a.decide(p+1,a.labels)
                a.observe(p,False,.01,.002);b.observe(p,False,.01,.002)
            else:
                with self.assertRaises(RuntimeError):a.observe(p,True,.01,.002)
        self.assertEqual(a.statistics(),b.statistics())

    def test_parameter_validation(self):
        for options in ({'gap_scale':float('nan')},{'gain_scale':0},{'return_window':2,'min_return_updates':3},
                        {'audit_fraction':1.1},{'cost':float('inf')}):
            with self.assertRaises(ValueError):QualityBudget(10,2,.7,**options)


class QualityEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import amcal_burst_structured_bank_protocol as m
        torch.set_num_threads(1);m.core.device=torch.device('cpu')

    def fixture(self):
        import torch
        import amcal_burst_structured_bank_protocol as m
        from test_structured_bank_protocol import TinyCNN,TinyContext
        torch.manual_seed(12)
        cnn=TinyCNN().eval()
        for p in cnn.parameters():p.requires_grad_(False)
        bank=[TinyContext().eval() for _ in range(3)];initial=deepcopy(bank[0].state_dict())
        for c in bank:c.load_state_dict(initial)
        a=SimpleNamespace(seed=42,classes=['0','1'],context_lr=.001,budget_fraction=.35,
            query_schedule='quality',update_policy='learned',bank_utility=False,
            train_epsilon=.2,eval_epsilon=.05,gain_weight=1.,retention_weight=1.,selection_cost=.0001,
            quality_reference_accuracy=.8,quality_bootstrap_labels=3,quality_audit_fraction=.2,
            quality_window=8,quality_return_window=8,quality_min_return_updates=4,
            quality_tolerance=.02,quality_gap_scale=.15,quality_gain_scale=.001,quality_burst=2)
        m.core.GAIN_WEIGHT=m.core.RETENTION_WEIGHT=1.;m.core.SELECTION_COST=.0001
        domain=(torch.randn(48,1,8),np.arange(48)%2,np.arange(48))
        return m,torch,cnn,bank,initial,a,domain

    def test_unqueried_labels_and_observers_cannot_change_quality_or_weights(self):
        m,torch,cnn,bank,initial,a,domain=self.fixture()
        agent=m.Agent(34,torch.device('cpu'),42);snapshot=deepcopy(agent.policy.state_dict())
        frozen=deepcopy(cnn.state_dict())
        first,trace=m.episode(a,cnn,bank,initial,agent,domain)
        agent_weights=deepcopy(agent.policy.state_dict());bank_weights=[deepcopy(c.state_dict()) for c in bank]
        ids={r['source_row'] for r in trace if r['queried']}
        y=domain[1].copy()
        for i in range(len(y)):
            if i not in ids:y[i]=99
        other=m.Agent(34,torch.device('cpu'),42,policy_state=snapshot)
        second,trace2=m.episode(a,cnn,bank,initial,other,(domain[0],y,domain[2]),same_level=True)
        self.assertEqual(first['quality_budget'],second['quality_budget'])
        keys=['action','prediction','queried','reward','quality_query','quality_need','quality_mean_net_transfer','quality_reason']
        self.assertEqual([[r[k] for k in keys] for r in trace],[[r[k] for k in keys] for r in trace2])
        for k,v in frozen.items():self.assertTrue(torch.equal(v,cnn.state_dict()[k]))
        for k,v in agent_weights.items():self.assertTrue(torch.equal(v,other.policy.state_dict()[k]))
        for saved,c in zip(bank_weights,bank):
            for k,v in saved.items():self.assertTrue(torch.equal(v,c.state_dict()[k]))
        self.assertLessEqual(first['labels'],first['label_cap'])
        self.assertGreater(first['td_updates'],0)
        for r in trace:self.assertEqual(r['queried'],r['quality_query'])
        monitors=[r for r in trace if r['quality_reason'] in ('bootstrap','audit')][-a.quality_window:]
        successes=sum(r['quality_monitor_prediction']==r['truth'] for r in monitors)
        self.assertAlmostEqual(first['quality_budget']['estimated_accuracy'],
                               (successes+2*a.quality_reference_accuracy)/(len(monitors)+2))
        # Actual paid transitions use the exact gate that next behavior sees.
        for transition in other.td_memory:
            _,action,_,successor,terminal,clock=transition
            if action>=4 and not terminal:
                self.assertEqual(int(successor[-3]),int(trace2[clock]['quality_query']))

    def test_evaluate_uses_clean_frozen_validation_once_and_saves_diagnostics(self):
        m,torch,cnn,bank,initial,a,domain=self.fixture()
        a.levels=[5];a.modes=['learned','random'];a.gamma=.95
        clean=(domain[0][:10],np.zeros(10,dtype=int),np.arange(10))
        expected=float(np.mean(cnn(clean[0]).argmax(1).numpy()==clean[1]))
        agent=m.Agent(34,torch.device('cpu'),42)
        def read(root,artifact,level,split):return clean if split=='val_ids' else domain
        with tempfile.TemporaryDirectory() as folder:
            a.output=Path(folder);a.burst_dir=Path(folder);models=a.output/'Models';models.mkdir()
            torch.save(dict(version=1,model_hashes={},training_levels=[0,1,3],route_width=34,
                            policy_state_dict=agent.policy.state_dict()),models/'structured_bank_ddqn.pth')
            with patch.object(m,'models_from_source',return_value=(cnn,bank,initial,34)), \
                 patch.object(m,'read_domain',side_effect=read),redirect_stdout(io.StringIO()) as output:
                m.evaluate(a,models,{})
            self.assertIn('Quality budget reference',output.getvalue())
            self.assertIn('Queries bootstrap/audit/adaptive',output.getvalue())
            self.assertIn('not a purely RL query policy',output.getvalue())
            files=list((a.output/'Results').glob('*_metrics.json'));self.assertEqual(len(files),2)
            for file in files:
                result=json.loads(file.read_text())
                self.assertAlmostEqual(result['quality_budget']['reference_accuracy'],expected)
                self.assertEqual(result['execution_config']['quality_budget_version'],1)
                self.assertEqual(result['quality_budget']['labels'],result['labels'])
            trace=next((a.output/'Results').glob('*_trace.csv'))
            with trace.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertIn('quality_reason',rows[0]);self.assertIn('quality_adaptive_probability',rows[0])


if __name__=='__main__':unittest.main()
