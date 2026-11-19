"""Standard-library budget-grid and frozen-policy contracts; no models."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amcal_final_budget_sweep import validate_budgets,make_row,summarize_rows,run_case,FIELDS
import amcal_final_multiseed as run


def result(dataset='IAT',budget=20,labels=None):
    n=968 if dataset=='IAT' else 801
    cap=int((budget/100)*n)
    scores=dict(AC=70.,PR=71.,RC=68.,F1=69.)
    return dict(samples=n,cap=cap,labels=cap if labels is None else labels,td_updates=0,
        online=scores,source_before=scores,source_after=dict(AC=68.,PR=69.,RC=67.,F1=66.),
        target_before=scores,target_after=scores,milliseconds_per_row=10.)


class FinalBudgetSweepTests(unittest.TestCase):
    def test_grid_is_sorted_and_invalid_values_rejected(self):
        self.assertEqual(validate_budgets([50,10,20]),[10.,20.,50.])
        for values in ([],[20,20],[0],[101],[float('nan')]):
            with self.assertRaises(ValueError): validate_budgets(values)

    def test_actual_caps_use_all_test_rows(self):
        for dataset,n in [('IAT',968),('PS',801)]:
            for budget in [10,15,20,25,30,35,40,45,50]:
                row=make_row(dataset,42,5,budget,50.,result(dataset,budget))
                self.assertEqual(row['samples'],n)
                self.assertEqual(row['cap'],int((budget/100)*n))
                self.assertEqual(row['original_forgetting_AC'],2.)

    def test_budget_excess_and_test_learning_are_rejected(self):
        value=result()
        value['labels']=value['cap']+1
        with self.assertRaises(ValueError): make_row('IAT',42,5,20,1.,value)
        value=result(); value['td_updates']=1
        with self.assertRaises(ValueError): make_row('IAT',42,5,20,1.,value)

    def test_sd_is_across_seed_means_and_missing_cells_rejected(self):
        rows=[]
        for seed in [42,48]:
            for budget in [10.,20.]:
                for level in run.LEVELS:
                    row=make_row('IAT',seed,level,budget,1.,result(budget=budget))
                    row['AC']=60. if seed==42 else 80.
                    rows.append(row)
        per_seed,by_level,overall=summarize_rows(rows,[42,48],[10.,20.],'IAT')
        self.assertEqual(len(per_seed),4)
        self.assertEqual(len(by_level),14)
        self.assertEqual(overall[0]['AC_mean'],70.)
        self.assertAlmostEqual(overall[0]['AC_sd'],200.**.5)
        with self.assertRaises(ValueError):
            summarize_rows(rows[:-1],[42,48],[10.,20.],'IAT')

    def test_test_adapter_preserves_legality_and_restores_choose(self):
        calls=[]
        def choose(state,legal,epsilon,rng):
            calls.append((legal,epsilon))
            return int(legal),[0.,1.]
        learner=SimpleNamespace(choose=choose)
        original=learner.choose
        def episode(exp,agent,domain,reference,seed,training,mode):
            self.assertFalse(training)
            self.assertEqual(mode,'DDQN')
            self.assertEqual(seed,42)
            action,_=agent.choose(None,False,0.,None)
            self.assertEqual(action,0)
            return {},[dict(epsilon=0.)]
        base=SimpleNamespace(run_episode=episode)
        _,trace=run_case(base,None,learner,None,None)
        self.assertEqual(calls,[(False,.05)])
        self.assertIs(learner.choose,original)
        self.assertEqual(trace[0]['epsilon'],.05)


if __name__=='__main__':
    unittest.main()
