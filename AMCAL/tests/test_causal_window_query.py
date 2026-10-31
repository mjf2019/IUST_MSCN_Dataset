"""User-run CPU contracts for delayed causal-window uncertainty selection."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.causal_window_query import window_quota,rank_arrived,run_window_episode,base
from test_two_action_ddqn import Fixture


class WindowRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); base.pilot.base.legacy.core.device=torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def test_quotas_exactly_recover_cap_for_full_and_partial_windows(self):
        for n in [1,2,4,5,6,17,32,801,968]:
            for width in [1,5,25]:
                quotas=[window_quota(i,min(n,i+width),.2) for i in range(0,n,width)]
                self.assertEqual(sum(quotas),int(.2*n))
                self.assertTrue(all(0<=q<=width for q in quotas))
        self.assertEqual(window_quota(965,968,.2),0)
        self.assertEqual(window_quota(800,801,.2),0)

    def test_top_rank_and_ties_use_no_labels(self):
        self.assertEqual(rank_arrived([.2,.9,.7,.9],2),[1,3])
        self.assertEqual(rank_arrived([.2,.9,.7,.9],0),[])
        with self.assertRaises(ValueError): rank_arrived([float('nan')],1)
        with self.assertRaises(ValueError): rank_arrived([.1],2)

    def test_all_predictions_in_window_precede_its_updates(self):
        fixture=Fixture(); agent=fixture.learner()
        predicted=[0]; acquired=[]; events=[]
        actual_outputs=base.BankExecutor.outputs
        actual_update=base.pilot.context_min_step
        def outputs(executor,*args,**kwargs):
            predicted[0]+=1; events.append(('predict',predicted[0]))
            return actual_outputs(executor,*args,**kwargs)
        def update(*args,**kwargs):
            acquired.append(predicted[0]); events.append(('feedback',predicted[0]))
            self.assertEqual(predicted[0]%5,0)
            return actual_update(*args,**kwargs)
        with patch.object(base.BankExecutor,'outputs',autospec=True,side_effect=outputs), \
                patch.object(base.pilot,'context_min_step',side_effect=update):
            result,trace=run_window_episode(fixture,agent,fixture.domain,fixture.reference,42,
                'UncertaintyWindow',5)
        self.assertEqual(acquired,[5,10,15,20,25,30])
        self.assertEqual(events[:5],[('predict',i) for i in range(1,6)])
        self.assertEqual(result['labels'],6)
        for row in trace:
            if row['queried']:
                self.assertEqual(row['feedback_step'],row['window_end'])
                self.assertGreaterEqual(row['feedback_step'],row['step'])
                self.assertLessEqual(row['feedback_delay_rows'],4)
        self.assertEqual(agent.steps,0); self.assertEqual(agent.updates,0)

    def test_first_window_selection_does_not_use_its_labels_or_later_inputs(self):
        fixture=Fixture(); agent=fixture.learner()
        _,original=run_window_episode(fixture,agent,fixture.domain,fixture.reference,42,'UncertaintyWindow',5)
        order=np.random.default_rng(42).permutation(len(fixture.domain[1]))
        x,y,ids=fixture.domain
        new_x=x.clone(); new_x[torch.tensor(order[5:])]+=100.
        changed=(new_x,1-y,ids)
        _,modified=run_window_episode(fixture,agent,changed,fixture.reference,42,'UncertaintyWindow',5)
        for a,b in zip(original[:5],modified[:5]):
            self.assertEqual(a['source_row'],b['source_row'])
            self.assertEqual(a['prediction'],b['prediction'])
            self.assertEqual(a['queried'],b['queried'])
            self.assertAlmostEqual(a['context_entropy'],b['context_entropy'],places=7)

    def test_modes_share_quota_and_arrival_order_and_preserve_source_models(self):
        fixture=Fixture(); agent=fixture.learner()
        cnn=deepcopy(fixture.cnn.state_dict()); context=deepcopy(fixture.template.state_dict())
        query=deepcopy(agent.policy.state_dict())
        reference_order=None
        for mode in ['UncertaintyWindow','RandomWindow','DDQNWindowRank']:
            result,trace=run_window_episode(fixture,agent,fixture.domain,fixture.reference,42,mode,5)
            self.assertEqual(result['labels'],result['cap'])
            self.assertEqual(result['samples'],32); self.assertEqual(result['td_updates'],0)
            order=[r['source_row'] for r in trace]
            if reference_order is None: reference_order=order
            self.assertEqual(order,reference_order)
            for start in range(0,32,5):
                group=trace[start:start+5]
                self.assertEqual(sum(r['queried'] for r in group),
                    window_quota(start,start+len(group),.2))
        for model,saved in [(fixture.cnn,cnn),(fixture.template,context),(agent.policy,query)]:
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,saved[key],rtol=0,atol=0)


if __name__=='__main__': unittest.main()

