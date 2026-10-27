"""No real datasets, package installation or long training runs."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from enhancements.budget_actor_math import request_by_rank, query_window_count, fit_signal_scales, scaled_competitive_score
try:
    import torch
except ImportError:
    torch=None


class BudgetDeploymentTests(unittest.TestCase):
    def test_early_window_consumes_exact_cap_and_never_queries_after_deadline(self):
        for n in (17,300,886,1073):
            for fraction in (.05,.1):
                for score in (-1e6,0.,1e6):
                    rng=np.random.default_rng(42);used=0;cap=int(n*fraction)
                    deadline=query_window_count(n,cap,.2)
                    for pos in range(n):
                        paid,_=request_by_rank(score,[-1.,0.,1.],used,cap,pos,n,float(rng.random()),.2)
                        if paid:self.assertLess(pos,deadline)
                        used+=int(paid)
                        if pos==deadline-1:self.assertEqual(used,cap)
                    self.assertEqual(used,cap)

    def test_scaling_makes_component_units_comparable_without_dropping_gap(self):
        rows=[dict(gap=v,weighted_gain=v/100,weighted_harm=v/1000,request_cost=.05) for v in (1.,2.,3.)]
        scales=fit_signal_scales(rows)
        self.assertAlmostEqual(scales['gap']/scales['weighted_gain'],100.)
        baseline=scaled_competitive_score(rows[1],scales)
        changed=dict(rows[1],gap=rows[1]['gap']+scales['gap'])
        self.assertAlmostEqual(scaled_competitive_score(changed,scales)-baseline,1.)
        changed=dict(rows[1],weighted_gain=rows[1]['weighted_gain']+scales['weighted_gain'])
        self.assertAlmostEqual(scaled_competitive_score(changed,scales)-baseline,1.)
        changed=dict(rows[1],weighted_harm=rows[1]['weighted_harm']+scales['weighted_harm'])
        self.assertAlmostEqual(scaled_competitive_score(changed,scales)-baseline,-1.)

    def test_constant_components_and_invalid_scales_are_handled(self):
        row=dict(gap=1.,weighted_gain=0.,weighted_harm=0.,request_cost=.05)
        self.assertEqual(fit_signal_scales([row,row]),dict(gap=1.,weighted_gain=1.,weighted_harm=1.))
        with self.assertRaises(ValueError):query_window_count(100,30,.2)
        with self.assertRaises(ValueError):fit_signal_scales([dict(row,gap=float('nan')),row])
        with self.assertRaises(ValueError):scaled_competitive_score(row,dict(gap=0.,weighted_gain=1.,weighted_harm=1.))

    def test_causal_policy_never_exceeds_cap_and_finishes_exactly(self):
        for n in (1,17,300,886,1073):
            for fraction in (.05,.1):
                for score in (-1e6,0.,1e6):
                    rng=np.random.default_rng(42);used=0;cap=int(n*fraction)
                    for pos in range(n):
                        paid,_=request_by_rank(score,[-1.,0.,1.],used,cap,pos,n,float(rng.random()))
                        used+=int(paid)
                        self.assertLessEqual(used,cap)
                    self.assertEqual(used,cap)

    def test_high_rank_is_preferred_and_negative_scores_can_rank(self):
        high,_=request_by_rank(-.01,[-.5,-.2,-.1],0,5,0,100,.5)
        low,_=request_by_rank(-1.,[-.5,-.2,-.1],0,5,0,100,.5)
        self.assertTrue(high);self.assertFalse(low)

    def test_constant_scores_have_randomized_ties(self):
        first,_=request_by_rank(0.,[0.,0.],0,1,0,10,.95)
        second,_=request_by_rank(0.,[0.,0.],0,1,0,10,.05)
        self.assertTrue(first);self.assertFalse(second)

    def test_forced_tail_is_explicit_and_exhaustion_skips(self):
        paid,detail=request_by_rank(-100.,[0.,1.],0,2,8,10,.2)
        self.assertTrue(paid);self.assertTrue(detail['forced_tail'])
        paid,_=request_by_rank(100.,[0.,1.],2,2,8,10,.2)
        self.assertFalse(paid)

    def test_invalid_states_are_rejected(self):
        for args in [(0.,[],0,1,0,10,.5),(float('nan'),[0.],0,1,0,10,.5),
                     (0.,[0.],2,1,0,10,.5),(0.,[0.],0,1,0,10,1.)]:
            with self.assertRaises(ValueError):request_by_rank(*args)


@unittest.skipIf(torch is None,'Requires existing PyTorch; nothing is installed')
class BudgetQuotaTests(unittest.TestCase):
    def test_probability_quota_and_gradient_are_correct(self):
        from enhancements.budget_actor import quota_probabilities
        for quota in (.05,.1):
            z=torch.tensor([-1.,-.3,.4,1.2],dtype=torch.double,requires_grad=True)
            p=quota_probabilities(z,quota)
            self.assertAlmostEqual(float(p.detach().mean()),quota,places=10)
            self.assertTrue(torch.autograd.gradcheck(lambda value:quota_probabilities(value,quota),(z,)))

    def test_shared_logit_offset_and_score_cost_do_not_change_ranking_gradient(self):
        from enhancements.budget_actor import quota_probabilities
        z=torch.tensor([-1.,-.3,.4,1.2],dtype=torch.double,requires_grad=True)
        targets=torch.tensor([-.4,-.2,.05,.3],dtype=torch.double)
        p=quota_probabilities(z,.1)
        g1=torch.autograd.grad(-(p*targets).mean(),z,retain_graph=True)[0]
        g2=torch.autograd.grad(-(p*(targets-10)).mean(),z)[0]
        torch.testing.assert_close(g1,g2)
        torch.testing.assert_close(quota_probabilities(z,.1),quota_probabilities(z+20,.1))

    def test_all_negative_targets_learn_relative_preference(self):
        from enhancements.budget_actor import quota_probabilities
        z=torch.zeros(4,dtype=torch.double,requires_grad=True)
        targets=torch.tensor([-.4,-.3,-.2,-.1],dtype=torch.double)
        (-torch.mean(quota_probabilities(z,.1)*targets)).backward()
        updated=z.detach()-.5*z.grad
        self.assertGreater(float(updated[-1]),float(updated[0]))

    def test_quota_fit_does_not_update_diagnostic_value_head(self):
        from enhancements.budget_actor import quota_max_step
        from enhancements.standardized_actor import StandardizedQueryMaximizer
        query=StandardizedQueryMaximizer(2,dict(mean=[0.,0.],scale=[1.,1.]))
        old=deepcopy(query.value.state_dict());optimizer=torch.optim.Adam(query.actor.parameters(),lr=.01)
        for _ in range(3):quota_max_step(query,optimizer,torch.randn(10,2),torch.linspace(-1,1,10),.1)
        for k,v in old.items():torch.testing.assert_close(v,query.value.state_dict()[k])

    def test_actor_deployment_bypasses_router_and_keeps_feedback_causal(self):
        from test_minimax_query import StreamContractTests
        fixture=StreamContractTests();run,a,runtime,domain,reference,calls=fixture.fixture()
        a.budget_actor_mode=True;a.budget_score_reference=[-100.,0.,100.]
        def forbidden(*args,**kwargs):raise AssertionError('Budget Actor called routing RL.')
        runtime[2].policy.route_logits=forbidden;runtime[2].observe_routes=forbidden
        query_before=deepcopy(runtime[3].state_dict());twin=deepcopy(runtime)
        flipped=domain[0],1-domain[1],domain[2]
        with contextlib.redirect_stdout(io.StringIO()):
            updated,result,trace=run(a,runtime,domain,domain,None,domain,42)
            _,_,changed=run(a,twin,flipped,domain,None,domain,42)
        self.assertEqual(result['labels'],2);self.assertEqual(calls,[])
        self.assertTrue(all(r['route']==1 and r['prediction']==r['context_prediction'] for r in trace))
        first=next(i for i,r in enumerate(trace) if r['queried'])
        self.assertEqual(trace[first]['prediction'],changed[first]['prediction'])
        self.assertEqual(trace[first]['queried'],changed[first]['queried'])
        for k,v in query_before.items():torch.testing.assert_close(v,updated[3].state_dict()[k])

    def test_train_collection_preserves_query_weights(self):
        from test_minimax_query import StreamContractTests
        fixture=StreamContractTests();run,a,runtime,domain,reference,calls=fixture.fixture(training=True)
        a.budget_actor_mode=True;a.collect_query_targets_only=True;a.train_query_schedule='uniform'
        initial=deepcopy(runtime[3].state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            updated,result,trace=run(a,runtime,domain,domain,reference,domain,42,training=True)
        self.assertEqual(result['labels'],2);self.assertGreater(len(calls),0)
        for k,v in initial.items():torch.testing.assert_close(v,updated[3].state_dict()[k])

    def test_uniform_control_is_evenly_spaced_context_only_and_does_not_fit_actor(self):
        from test_minimax_query import StreamContractTests
        fixture=StreamContractTests();run,a,runtime,domain,reference,calls=fixture.fixture()
        uniform_query_at=fixture.namespace['uniform_query_at']
        a.budget_actor_mode=True;a.eval_query_schedule='uniform'
        def forbidden(*args,**kwargs):raise AssertionError('Uniform control called routing RL.')
        runtime[2].policy.route_logits=forbidden;runtime[2].observe_routes=forbidden
        before=deepcopy(runtime[3].state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            updated,result,trace=run(a,runtime,domain,domain,None,domain,42)
        self.assertEqual(result['labels'],result['cap'])
        expected=[uniform_query_at(i,len(trace),result['cap']) for i in range(len(trace))]
        self.assertEqual([r['queried'] for r in trace],expected)
        self.assertTrue(all(r['route']==1 and r['prediction']==r['context_prediction'] for r in trace))
        for k,v in before.items():torch.testing.assert_close(v,updated[3].state_dict()[k])

    def test_windowed_uniform_and_actor_keep_predictions_causal_and_weights_fixed(self):
        from test_minimax_query import StreamContractTests
        for schedule in ('uniform','policy'):
            fixture=StreamContractTests();run,a,runtime,domain,reference,calls=fixture.fixture()
            a.budget_actor_mode=True;a.eval_query_schedule=schedule
            a.query_window_fraction=.5;a.budget_score_reference=[1e9]
            before=deepcopy(runtime[3].state_dict())
            with contextlib.redirect_stdout(io.StringIO()):
                _,result,trace=run(a,runtime,domain,domain,None,domain,42)
            self.assertEqual(result['labels'],result['cap'])
            self.assertTrue(all(row['step']<=int(len(trace)*.5) for row in trace if row['queried']))
            self.assertTrue(all(row['route']==1 for row in trace))
            for k,v in before.items():torch.testing.assert_close(v,runtime[3].state_dict()[k])


if __name__=='__main__':unittest.main()
