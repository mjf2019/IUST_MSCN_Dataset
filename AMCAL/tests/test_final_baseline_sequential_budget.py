"""Budget-boundary contracts; no model imports or execution."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_sequential_baseline_budget import reset_domain_budget


class SequentialBaselineBudgetTests(unittest.TestCase):
    def test_ral_keeps_fitted_model_feedback_and_thresholds(self):
        pool=object(); committee=object()
        model=SimpleNamespace(_n_seen=900,_n_acquired_samples=150,
            _training_data_samples=pool,_committee_classifier=committee,
            _threshold_uncertainty=.77,_alpha=[.2,.3,.5],_budget=.2)
        learner=SimpleNamespace(model=model)
        reset_domain_budget(learner,'ral')
        self.assertEqual(model._n_seen,0)
        self.assertEqual(model._n_acquired_samples,0)
        self.assertIs(model._training_data_samples,pool)
        self.assertIs(model._committee_classifier,committee)
        self.assertEqual(model._threshold_uncertainty,.77)
        self.assertEqual(model._alpha,[.2,.3,.5])

    def test_active_methods_keep_native_windows_rng_and_adaptive_budget(self):
        for name in ['calmid','micfoal']:
            windows,rng,trees=object(),object(),object()
            model=SimpleNamespace(pending=None,seen=900,queried=150,windows=windows,
                rng=rng,trees=trees,budget=.23,last_period_f1=.64,clock=5000)
            reset_domain_budget(SimpleNamespace(model=model),name)
            self.assertEqual((model.seen,model.queried),(0,0))
            self.assertIs(model.windows,windows)
            self.assertIs(model.rng,rng)
            self.assertIs(model.trees,trees)
            self.assertEqual(model.budget,.23)
            self.assertEqual(model.clock,5000)
            self.assertEqual(model.last_period_f1,.64)

    def test_dfe_and_arf_are_not_reset(self):
        for name in ['dfe','arf']:
            learner=SimpleNamespace(model=object(),steps=100)
            before=learner.model
            reset_domain_budget(learner,name)
            self.assertIs(learner.model,before)
            self.assertEqual(learner.steps,100)

    def test_recent_window_ral_and_pending_active_feedback_are_rejected(self):
        with self.assertRaises(ValueError):
            reset_domain_budget(SimpleNamespace(model=SimpleNamespace(_online_update_window=5)),'ral')
        with self.assertRaises(ValueError):
            reset_domain_budget(SimpleNamespace(model=SimpleNamespace(pending=object())),'calmid')


if __name__=='__main__':
    unittest.main()
