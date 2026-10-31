"""User-run budget caps and frozen-DDQN evaluation contracts; no real data."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_selected_beta_budgets as runner
from test_two_action_ddqn import Fixture


class BudgetTests(unittest.TestCase):
    def test_budget_directories_do_not_overlap(self):
        paths = [runner.budget_directory(300,b) for b in (.2,.3,.4,.5)]
        self.assertEqual(len(set(paths)),4)
        self.assertNotIn('testp300',paths)
        self.assertNotEqual(runner.budget_directory(0,.3),paths[1])

    def test_each_budget_uses_requested_cap_without_ddqn_training(self):
        fixture = Fixture(); fixture.a.dataset = 'IAT'; fixture.a.beta = 26.
        learner = fixture.learner()
        common = dict(AC=80.,PR=80.,RC=80.,F1=80.)
        for budget in (.3,.4,.5):
            cap = int(budget*len(fixture.domain[1]))
            result = dict(online=common,samples=len(fixture.domain[1]),labels=cap,cap=cap,
                labels_percent=100*cap/len(fixture.domain[1]),source_before=common,
                source_after=common,original_domain_forgetting={k:0. for k in common},
                target_after=common,td_updates=0)
            def episode(*args,**kwargs):
                self.assertEqual(fixture.a.budget_fraction,budget)
                self.assertFalse(kwargs['training']); self.assertEqual(kwargs['mode'],'DDQN')
                return deepcopy(result),[dict(step=1)]
            with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()), \
                    patch.object(runner.base,'run_episode',side_effect=episode) as invoked:
                directory = Path(temp)/'case'; execution = dict(budget_fraction=budget)
                row = runner.compare_case(fixture,learner,'Current',5,fixture.domain,fixture.reference,
                    common,directory,execution)
                self.assertEqual(row['labels'],cap)
                runner.compare_case(fixture,learner,'Current',5,fixture.domain,fixture.reference,
                    common,directory,execution)
                self.assertEqual(invoked.call_count,1)
                with self.assertRaisesRegex(ValueError,'case changed'):
                    runner.compare_case(fixture,learner,'Current',5,fixture.domain,fixture.reference,
                        common,directory,dict(budget_fraction=.6))

    def test_illegal_budget_is_rejected_before_episode(self):
        fixture = Fixture(); fixture.a.dataset = 'IAT'; fixture.a.beta = 26.
        for budget in (0.,-1.,1.1,float('nan')):
            with tempfile.TemporaryDirectory() as temp, patch.object(runner.base,'run_episode') as episode:
                with self.assertRaisesRegex(ValueError,'Invalid TEST budget'):
                    runner.compare_case(fixture,fixture.learner(),'Current',5,fixture.domain,fixture.reference,
                        {},Path(temp),dict(budget_fraction=budget))
                episode.assert_not_called()


if __name__ == '__main__': unittest.main()

