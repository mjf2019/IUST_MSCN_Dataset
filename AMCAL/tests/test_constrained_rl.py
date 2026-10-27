from copy import deepcopy
import ast
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires existing PyTorch; nothing is installed')
class ConstrainedRLTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(42)

    def test_dual_ascent_penalizes_only_constraint_violation_and_projects(self):
        from enhancements.constrained_agent import dual_ascent,constraint_values
        np.testing.assert_allclose(dual_ascent([5,5,5],[.1,-.1,.2],.5),[5.05,4.95,5.1])
        np.testing.assert_allclose(dual_ascent([0,99,0],[-1,5,-1],1),[0,100,0])
        r=dict(AC=.8,F1=.7,forgetting=.02);reference=dict(AC=.82,F1=.72,forgetting=.01)
        np.testing.assert_allclose(constraint_values(r,reference,[.01,.01,.005]),[.01,.01,.005])

    def test_macro_metrics_use_all_classes_and_zero_denominators(self):
        from enhancements.constrained_agent import quality
        metrics=quality([[3,1],[2,4]])
        self.assertAlmostEqual(metrics['AC'],.7)
        self.assertAlmostEqual(metrics['PR'],.7)
        self.assertAlmostEqual(metrics['RC'],(3/4+4/6)/2)
        self.assertAlmostEqual(quality(np.zeros((2,2)))['F1'],0.)

    def test_train_curriculum_is_unique_class_covered_seeded_and_harder_than_uniform(self):
        from enhancements.constrained_agent import training_indices
        labels=np.repeat(np.arange(6),50);losses=np.tile(np.arange(50),6)
        hard=training_indices(labels,losses,96,42,.5)
        uniform=training_indices(labels,losses,96,42,0.)
        np.testing.assert_array_equal(hard,training_indices(labels,losses,96,42,.5))
        self.assertEqual(len(hard),96);self.assertEqual(len(set(hard)),96)
        self.assertTrue(all((labels[hard]==c).sum()>=3 for c in range(6)))
        self.assertGreater(losses[hard].mean(),losses[uniform].mean())

    def test_ddqn_uses_online_legal_argmax_and_target_value(self):
        from enhancements.constrained_agent import ConstrainedDDQN
        agent=ConstrainedDDQN(4,'cpu',42)
        for network,bias in [(agent.policy,[1.,2.,100.]),(agent.target,[10.,3.,200.])]:
            for p in network.parameters():p.data.zero_()
            network.network[-1].bias.data.copy_(torch.tensor(bias))
        for _ in range(32):agent.memory.append((np.zeros(4,dtype=np.float32),0,np.zeros(4,dtype=np.float32),np.zeros(4,dtype=np.float32),False,False))
        self.assertAlmostEqual(agent.learn([1.,1.,1.],1),1.5,places=5)

    def test_terminal_transitions_have_no_bootstrap(self):
        from enhancements.constrained_agent import ConstrainedDDQN
        agent=ConstrainedDDQN(4,'cpu',42)
        for network in (agent.policy,agent.target):
            for p in network.parameters():p.data.zero_()
        agent.target.network[-1].bias.data.fill_(100.)
        for _ in range(32):agent.memory.append((np.zeros(4,dtype=np.float32),0,np.zeros(4,dtype=np.float32),np.zeros(4,dtype=np.float32),True,False))
        self.assertAlmostEqual(agent.learn([1.,1.,1.],1),0.)

    def test_training_resume_restores_optimizer_target_replay_and_sampler(self):
        from enhancements.constrained_agent import ConstrainedDDQN
        first=ConstrainedDDQN(4,'cpu',42);rng=np.random.default_rng(10)
        states=[rng.normal(size=4).astype(np.float32) for _ in range(64)]
        first.add_episode(states,[1]*64,[np.array([0,.01,0,0],dtype=np.float32)]*64,[True]*64)
        first.learn([5,5,5],2);snapshot=first.snapshot()
        second=ConstrainedDDQN(4,'cpu',99);second.restore(snapshot)
        self.assertAlmostEqual(first.learn([4,6,3],3),second.learn([4,6,3],3),places=7)
        for key,value in first.policy.state_dict().items():torch.testing.assert_close(value,second.policy.state_dict()[key])

    def test_training_orchestration_records_constraints_and_completed_run_reuses_model(self):
        from enhancements.constrained_agent import ConstrainedDDQN,constraint_values,dual_ascent
        source=Path(__file__).resolve().parents[1]/'amcal_constrained_rl.py'
        functions=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)
            and n.name in ('save_torch','policy_hash','row','show','print_table','validate','train')]
        def write_json(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        def write_csv(path,rows):
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        domain=(torch.zeros(32,2),np.array([0,1]*16),np.arange(32))
        metric=dict(AC=.7,PR=.7,RC=.7,F1=.7,samples=32,labels=0,cap=6,label_fraction=0.,
            source_initial_AC=.8,source_final_AC=.8,forgetting=0.,milliseconds_per_row=.1)
        class FakeExperiment:
            def __init__(self,a):
                self.a=a;self.train={i:domain for i in (0,1,3)};self.reference=domain
                self.stats=dict(mean=np.zeros(2),scale=np.ones(2));self.calls=0;self.threshold=.75
            def split(self,*args):return self.train,self.reference
            def episode_domain(self,*args):return domain
            def cached(self,*args):return dict(metric)
            def run(self,*args):
                self.calls+=1
                states=[np.zeros(4,dtype=np.float32)]*32
                components=[np.array([0,.7/32,.7/32,0],dtype=np.float32)]*32
                return dict(metric),[dict(step=i) for i in range(32)],(states,[1]*32,components,[True]*32),[(s,1,1,True) for s in states]
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(output=Path(tmp),seed=42,pilot_rows=32,episode_rows=32,episodes=3,validate_every=1,
                updates_per_episode=1,dual_lr=.5,ac_tolerance=.01,f1_tolerance=.01,forgetting_tolerance=.005)
            exp=FakeExperiment(a);agent=ConstrainedDDQN(4,'cpu',42)
            ns=dict(torch=torch,np=np,deepcopy=deepcopy,hashlib=hashlib,json=json,LEVELS=[0,1,3],
                MODES=['FrontLoadedContext','Uncertainty','Random'],sample=lambda domain,*args:domain,
                constraint_values=constraint_values,dual_ascent=dual_ascent,
                pilot=SimpleNamespace(write_json=write_json,write_csv=write_csv,
                    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()))
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
            with contextlib.redirect_stdout(io.StringIO()):ns['train'](a,exp,agent,dict(test=True))
            self.assertTrue((a.output/'training_complete.json').exists())
            selected=json.loads((a.output/'selected_validation.json').read_text());self.assertTrue(selected['feasible'])
            self.assertEqual(len(json.loads((a.output/'training.json').read_text())['episodes']),3)
            calls=exp.calls
            with contextlib.redirect_stdout(io.StringIO()):ns['train'](a,exp,agent,dict(test=True))
            self.assertEqual(calls,exp.calls)

    def test_experiment_factory_audit_separation_and_cache_with_tiny_real_networks(self):
        from enhancements.constrained_agent import ConstrainedDDQN,training_indices
        from enhancements.constrained_episode import run_episode,probabilities,entropy,context_accuracy
        from enhancements.minimax_query import context_min_step
        source=Path(__file__).resolve().parents[1]/'amcal_constrained_rl.py'
        names=('policy_hash','domain_hash','clean_domain','sample','Experiment')
        nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        old_source=Path(__file__).resolve().parents[1]/'amcal_minimax_paironly.py'
        old_names=('subset','pilot_indices','reference_partition','feature_keys')
        old_nodes=[n for n in ast.parse(old_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in old_names]
        support=dict(torch=torch,np=np)
        exec(compile(ast.Module(body=old_nodes,type_ignores=[]),str(old_source),'exec'),support)
        cnn=torch.nn.Linear(2,2).eval().requires_grad_(False);template=torch.nn.Linear(2,2).eval()
        def cache(model,x):
            with torch.no_grad():return model(x),x
        class Memory:
            def __init__(self,*args):self.rows=[]
            def draw(self,*args):return self.rows[-4:],None
            def add(self,i,x,y):self.rows.append((x.detach().clone(),torch.tensor([y])))
        train_x=torch.randn(120,2);train_y=np.array([0,1]*60);train_ids=np.arange(120)
        val_x=torch.randn(40,2)+10;val_y=np.array([0,1]*20);val_ids=np.arange(200,240)
        reference=(train_x[:20],train_y[:20],train_ids[:20]);anchors=(train_x[20:],train_y[20:],train_ids[20:])
        domains={(level,split):(x+level*.1,y,ids) for level in (0,1,3)
            for split,(x,y,ids) in [('train_ids',(train_x,train_y,train_ids)),('val_ids',(val_x,val_y,val_ids))]}
        base=SimpleNamespace(legacy=SimpleNamespace(read_domain=lambda root,artifact,level,split:domains[level,split],
            base_cache=cache),load_cnn=lambda *args:cnn,build_bank=lambda *args,**kwargs:[deepcopy(template)])
        def write_json(path,value):path.write_text(json.dumps(value))
        def write_csv(path,rows):
            with path.open('w',newline='') as output:
                writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(models=Path(tmp),dataset='IAT',classes=['0','1'],pilot_rows=32,episode_rows=32,
                hard_fraction=.5,seed=42,config=object(),burst_dir=Path(tmp),budget_fraction=.2,context_lr=.01,replay_weight=.1)
            torch.save(dict(architecture='legacy'),a.models/'initial_cnn_model.pth')
            pilot=SimpleNamespace(**{key:support[key] for key in old_names},base=base,Replay=Memory,
                context_min_step=context_min_step,write_json=write_json,write_csv=write_csv,
                digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
            healthy=dict(provenance=dict(reference_source_ids=list(range(20))),context_state_dict=deepcopy(template.state_dict()))
            budget=SimpleNamespace(setup=lambda args:(healthy,{},anchors,reference,'teacherhash'))
            ns=dict(torch=torch,np=np,deepcopy=deepcopy,hashlib=hashlib,json=json,LEVELS=[0,1,3],pilot=pilot,budget=budget,
                run_episode=run_episode,probabilities=probabilities,entropy=entropy,context_accuracy=context_accuracy,training_indices=training_indices)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
            experiment=ns['Experiment'](a);agent=ConstrainedDDQN(experiment.width,'cpu',42)
            self.assertEqual(experiment.width,28)
            for domain in experiment.train.values():self.assertFalse(set(domain[2])&set(reference[2]))
            audit=set(experiment.references['val_ids'][2])
            for domain in experiment.domains['val_ids'].values():self.assertFalse(set(domain[2])&audit)
            directory=Path(tmp)/'cache'
            result=experiment.cached(directory,experiment.episode_domain(0,100),experiment.reference,agent,'RL',42)
            self.assertLessEqual(result['labels'],result['cap'])
            self.assertEqual(result,experiment.cached(directory,experiment.episode_domain(0,100),experiment.reference,agent,'RL',42))
            with torch.no_grad():agent.policy.network[-1].bias.add_(1.)
            with self.assertRaises(ValueError):experiment.cached(directory,experiment.episode_domain(0,100),experiment.reference,agent,'RL',42)

    def fixture(self):
        from enhancements.constrained_agent import ConstrainedDDQN
        from enhancements.minimax_query import context_min_step
        cnn=torch.nn.Linear(2,2).eval().requires_grad_(False)
        template=torch.nn.Linear(2,2).eval();copies=[];updates=[]
        class Memory:
            def __init__(self):self.rows=[]
            def draw(self,*args):return self.rows[-4:],None
            def add(self,source_id,x,truth):self.rows.append((x.detach().clone(),torch.tensor([truth])))
        def make_context():
            value=deepcopy(template);copies.append(value);return value
        def update(*args):updates.append(1);return context_min_step(*args,replay_weight=.1)
        def cache(model,x):
            with torch.no_grad():return model(x),x
        x=torch.randn(10,2);y=np.array([0,1]*5);ids=np.arange(10)
        ref=(torch.randn(6,2),np.array([0,1]*3),np.arange(100,106))
        agent=ConstrainedDDQN(28,'cpu',42)
        for p in agent.policy.parameters():p.data.zero_()
        agent.policy.network[-1].bias.data.copy_(torch.tensor([0.,1.,3.]))
        kwargs=dict(cnn=cnn,make_context=make_context,make_memory=lambda *args:Memory(),cache=cache,update=update,
            domain=(x,y,ids),anchors=(x,y,ids),reference=ref,stats=dict(mean=np.zeros(2),scale=np.ones(2)),classes=2,
            agent=agent,mode='RL',seed=42,budget=.2,threshold=.75,context_lr=.01)
        return kwargs,template,copies,updates

    def test_feedback_cap_and_reward_components_match_the_episode_objective(self):
        from enhancements.constrained_episode import run_episode
        from enhancements.constrained_agent import scalar_rewards
        kwargs,template,copies,updates=self.fixture();before=deepcopy(template.state_dict())
        policy_before=deepcopy(kwargs['agent'].policy.state_dict());cnn_before=deepcopy(kwargs['cnn'].state_dict())
        result,trace,experience,_=run_episode(**kwargs)
        self.assertEqual(result['labels'],2);self.assertEqual(len(updates),2)
        self.assertTrue(all(not r['queried'] for r in trace[2:]))
        components=np.stack(experience[2]);totals=components.sum(0)
        np.testing.assert_allclose(totals,[-result['label_fraction'],result['AC'],result['F1'],-result['forgetting']],atol=1e-6)
        multipliers=torch.tensor([2.,3.,4.]);rewards=scalar_rewards(torch.tensor(components),multipliers)
        self.assertAlmostEqual(float(rewards.sum()),-result['label_fraction']+2*result['AC']+3*result['F1']-4*result['forgetting'],places=5)
        for key,value in before.items():torch.testing.assert_close(value,template.state_dict()[key])
        for key,value in policy_before.items():torch.testing.assert_close(value,kwargs['agent'].policy.state_dict()[key])
        for key,value in cnn_before.items():torch.testing.assert_close(value,kwargs['cnn'].state_dict()[key])
        self.assertTrue(any(not torch.equal(value,copies[0].state_dict()[key]) for key,value in before.items()))

    def test_current_query_prediction_precedes_its_truth(self):
        from enhancements.constrained_episode import run_episode
        kwargs,_,_,_=self.fixture();domain=kwargs['domain']
        _,first,_,_=run_episode(**kwargs)
        y=domain[1].copy();y[first[0]['source_row']]=1-y[first[0]['source_row']]
        kwargs['domain']=(domain[0],y,domain[2])
        _,second,_,_=run_episode(**kwargs)
        for key in ('action','prediction','route','q0','q1','q2'):
            self.assertEqual(first[0][key],second[0][key])

    def test_query_action_has_fixed_context_semantics_even_if_frozen_q_is_larger(self):
        from enhancements.constrained_episode import run_episode
        kwargs,_,_,_=self.fixture()
        with torch.no_grad():kwargs['agent'].policy.network[-1].bias.copy_(torch.tensor([2.,1.,3.]))
        _,trace,_,_=run_episode(**kwargs)
        self.assertTrue(all(row['route']==1 for row in trace if row['queried']))
        self.assertTrue(all(row['route']==0 for row in trace if not row['queried']))

    def test_unqueried_observer_labels_cannot_change_any_policy_decision(self):
        from enhancements.constrained_episode import run_episode
        kwargs,_,_,_=self.fixture();domain=kwargs['domain']
        _,first,exp1,_=run_episode(**kwargs)
        y=domain[1].copy()
        for r in first:
            if not r['queried']:y[r['source_row']]=1-y[r['source_row']]
        kwargs['domain']=(domain[0],y,domain[2])
        _,second,exp2,_=run_episode(**kwargs)
        np.testing.assert_array_equal(exp1[0],exp2[0])
        self.assertEqual([(r['action'],r['prediction']) for r in first],[(r['action'],r['prediction']) for r in second])

    def test_future_features_do_not_change_earlier_decisions(self):
        from enhancements.constrained_episode import run_episode
        kwargs,_,_,_=self.fixture();domain=kwargs['domain']
        _,first,_,_=run_episode(**kwargs)
        inputs=domain[0].clone()
        for r in first[2:]:inputs[r['source_row']]+=100.
        kwargs['domain']=(inputs,domain[1],domain[2])
        _,second,_,_=run_episode(**kwargs)
        self.assertEqual([(r['action'],r['prediction']) for r in first[:2]],[(r['action'],r['prediction']) for r in second[:2]])

    def test_all_controls_use_the_same_order_and_reset_context(self):
        from enhancements.constrained_episode import run_episode
        kwargs,template,copies,updates=self.fixture();orders=[]
        for mode in ('RL','Uncertainty','Random','FrontLoadedContext','NoQueryContext'):
            kwargs['mode']=mode;result,trace,_,_=run_episode(**kwargs)
            orders.append([r['source_row'] for r in trace]);self.assertLessEqual(result['labels'],2)
        self.assertTrue(all(order==orders[0] for order in orders))
        self.assertEqual(len(copies),5)

    def test_zero_budget_never_updates_context(self):
        from enhancements.constrained_episode import run_episode
        kwargs,_,_,updates=self.fixture();kwargs['budget']=0.
        result,trace,_,_=run_episode(**kwargs)
        self.assertEqual(result['labels'],0);self.assertFalse(updates)
        self.assertTrue(all(r['action']!=2 for r in trace))


if __name__=='__main__':unittest.main()
