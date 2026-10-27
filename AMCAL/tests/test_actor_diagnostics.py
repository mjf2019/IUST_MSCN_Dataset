import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from enhancements.actor_diagnostics import average_ranks,correlation,ranking_report,print_ranking_report


def candidates(actor,target,gain,harm=None):
    harm=[0.]*len(actor) if harm is None else harm
    return [dict(actor_logit_before=a,ranking_target=s,weighted_gain=g,weighted_harm=f,request_cost=.05)
        for a,s,g,f in zip(actor,target,gain,harm)]


class RankingDiagnosticTests(unittest.TestCase):
    def test_tied_spearman_ranks_and_constant_vectors(self):
        np.testing.assert_allclose(average_ranks([3,1,1,2]),[3,.5,.5,2])
        self.assertAlmostEqual(correlation([1,2,3],[9,4,1],True),-1.)
        self.assertIsNone(correlation([1,1,1],[1,2,3],True))

    def test_perfect_actor_has_zero_target_regret(self):
        report=ranking_report(candidates([1,2,3,4],[1,2,3,4],[.1,.2,.3,.4]),2)
        self.assertAlmostEqual(report['correlations']['actor_target']['spearman'],1.)
        self.assertAlmostEqual(report['actor_target_regret'],0.)
        self.assertAlmostEqual(report['actor_utility_regret'],0.)

    def test_target_can_be_learned_perfectly_but_rank_utility_backwards(self):
        report=ranking_report(candidates([1,2,3,4],[1,2,3,4],[.4,.3,.2,.1]),2)
        self.assertAlmostEqual(report['correlations']['actor_target']['spearman'],1.)
        self.assertAlmostEqual(report['correlations']['target_utility']['spearman'],-1.)
        self.assertGreater(report['actor_utility_regret'],0.)
        self.assertLess(report['selectors']['OracleTarget']['net_utility'],
            report['selectors']['RandomExpectation']['net_utility'])

    def test_bad_actor_is_distinguished_from_a_good_target(self):
        report=ranking_report(candidates([4,3,2,1],[1,2,3,4],[.1,.2,.3,.4]),2)
        self.assertAlmostEqual(report['correlations']['actor_target']['spearman'],-1.)
        self.assertAlmostEqual(report['correlations']['target_utility']['spearman'],1.)
        self.assertGreater(report['actor_target_regret'],0.)

    def test_signed_utility_includes_forgetting_and_original_cost(self):
        report=ranking_report(candidates([1,2],[1,2],[.1,.2],[.05,.5]),1)
        self.assertAlmostEqual(report['selectors']['Actor']['net_utility'],-.35)
        self.assertAlmostEqual(report['selectors']['OracleUtility']['net_utility'],0.)

    def test_invalid_values_and_counts_are_rejected(self):
        rows=candidates([1,2],[1,2],[.1,.2])
        for k in (0,3):
            with self.assertRaises(ValueError):ranking_report(rows,k)
        with self.assertRaises(ValueError):ranking_report([dict(rows[0],ranking_target=float('nan')),rows[1]],1)

    def test_ties_are_deterministic_without_favoring_input_order(self):
        rows=candidates([0]*10,[0]*10,list(range(10)))
        self.assertEqual(ranking_report(rows,3,42),ranking_report(rows,3,42))
        self.assertIsNone(ranking_report(rows,3)['correlations']['actor_target']['spearman'])

    def test_console_reports_target_learning_and_utility_separately(self):
        report=ranking_report(candidates([1,2,3],[1,2,3],[.3,.2,.1]),1)
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):print_ranking_report('VAL',0,report)
        for label in ('Actor~target','target~utility','OracleUtility','regret'):
            self.assertIn(label,capture.getvalue())


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires existing PyTorch; nothing is installed')
class OfflineObserverContracts(unittest.TestCase):
    def test_observer_records_targets_without_fitting_actor_or_changing_saved_initial_context(self):
        from test_minimax_query import StreamContractTests
        fixture=StreamContractTests();run,a,runtime,domain,reference,calls=fixture.fixture(training=True)
        a.budget_actor_mode=True;a.collect_query_targets_only=True
        a.train_query_schedule='uniform';a.query_window_fraction=.5
        query_before=deepcopy(runtime[3].state_dict());router_before=deepcopy(runtime[2].policy.state_dict())
        context_before=deepcopy(runtime[1].state_dict());copy_runtime=deepcopy(runtime)
        class Recorder:
            def __init__(self):self.rows=[]
            def observe(self,state,row):
                if row.get('score') is not None:self.rows.append(dict(row))
        sink=Recorder()
        with contextlib.redirect_stdout(io.StringIO()):
            after,result,trace=run(a,copy_runtime,domain,domain,reference,domain,42,training=True,diagnostic_sink=sink)
        self.assertTrue(sink.rows);self.assertTrue(calls)
        self.assertEqual(result['labels'],result['cap'])
        for key,value in query_before.items():torch.testing.assert_close(value,after[3].state_dict()[key])
        for key,value in router_before.items():torch.testing.assert_close(value,after[2].policy.state_dict()[key])
        for key,value in context_before.items():torch.testing.assert_close(value,runtime[1].state_dict()[key])
        self.assertTrue(any(not torch.equal(value,after[1].state_dict()[key]) for key,value in context_before.items()))


if __name__=='__main__':unittest.main()
