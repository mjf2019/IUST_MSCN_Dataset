"""User-run CPU contracts; no real datasets or installation required."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_supplied_ral_af20 as runner


class SuppliedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.baseline.core.device=torch.device('cpu')

    def test_af_uses_exact_cap_even_for_imbalanced_classes(self):
        y=np.repeat(np.arange(3),[5,15,80])
        selected=runner.balanced_budget_indices(y,np.arange(3),20,42)
        self.assertEqual(len(selected),20)
        self.assertEqual(len(set(selected)),20)
        mask=np.ones(len(y),dtype=bool);mask[selected]=False
        for cls in range(3):
            self.assertTrue(np.any(y[selected]==cls))
            self.assertTrue(np.any(y[mask]==cls))

    def test_af_support_is_reproducible_and_spends_nondivisible_cap(self):
        y=np.arange(105)%2
        first=runner.balanced_budget_indices(y,np.arange(2),21,42)
        second=runner.balanced_budget_indices(y,np.arange(2),21,42)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(len(first),21)
        self.assertEqual(sorted(np.bincount(y[first]).tolist()),[10,11])

    def test_af_rejects_an_impossible_budget_without_duplicates_or_query_fallback(self):
        with self.assertRaises(ValueError):
            runner.balanced_budget_indices(np.array([0,1]),np.arange(2),1,42)
        with self.assertRaises(ValueError):
            runner.balanced_budget_indices(np.array([0,0,1,1]),np.arange(2),3,42)

    def test_af_query_purge_excludes_all_support_and_source_feature_duplicates(self):
        target=np.array([[1,2],[1,2],[2,2],[10,10],[3,3]],dtype=np.float32)
        mask,count=runner.af_query_mask(target,np.array([[10,10]],dtype=np.float32),np.array([0]))
        np.testing.assert_array_equal(np.flatnonzero(mask),[2,4])
        self.assertEqual(count,2)

    def test_ral_calls_full_refit_after_every_feedback_and_predicts_first(self):
        rng=np.random.default_rng(42)
        source=rng.normal(size=(20,8)).astype(np.float32);sy=np.arange(20)%2
        model=runner.RAL([KNeighborsClassifier(5),
            RandomForestClassifier(n_estimators=10,random_state=42,n_jobs=1),
            DecisionTreeClassifier(random_state=42)],.9,.025,.01,
            reward=1,penalty=-1,budget=.2,model_update_frequency=1)
        model.fit(source,sy)
        saved=dict(model=model,ral_protocol_version=2,source_labels=20)
        a=SimpleNamespace(seed=42,classes=['0','1'],budget_fraction=.2,ral_update_window=0,
            ral_budget_mode='prefix',progress_every=0)
        x=rng.normal(size=(10,8)).astype(np.float32);y=np.arange(10)%2
        domain=torch.as_tensor(x).unsqueeze(1),y,np.arange(100,110)
        reference=torch.as_tensor(source).unsqueeze(1),sy
        first_index=np.random.default_rng(42).permutation(10)[0]
        expected_first=int(model._committee_classifier.predict(x[first_index:first_index+1])[0])
        original=runner.RAL.acquire_label; lengths=[]
        def observed(live,*args):
            before=len(live._training_data_samples.y)
            original(live,*args)
            after=len(live._training_data_samples.y)
            lengths.append((before,after))
            self.assertEqual(after,20+live._n_acquired_samples)
            np.testing.assert_array_equal(np.asarray(live._training_data_samples.X[:20]),source)
        with patch.object(runner.RAL,'ask_committee',return_value=(True,[True]*3)), \
                patch.object(runner.RAL,'acquire_label',new=observed):
            metrics,trace=runner.baseline.evaluate_level(a,'ral',saved,domain,reference)
        self.assertEqual(metrics['labels'],2)
        self.assertEqual(lengths,[(20,21),(21,22)])
        self.assertEqual(trace[0]['prediction'],expected_first)
        self.assertEqual(len(saved['model']._training_data_samples.y),20)
        self.assertEqual(saved['model']._n_acquired_samples,0)
        self.assertEqual(sum(r['queried'] for r in trace),2)

    def test_af_fit_receives_only_budget_support_and_cached_case_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            rng=np.random.default_rng(42)
            sx=rng.normal(size=(100,8)).astype(np.float32);sy=np.arange(100)%2
            tx=(rng.normal(size=(50,8))+3).astype(np.float32);ty=np.arange(50)%2
            source=torch.as_tensor(sx).unsqueeze(1),sy,np.arange(1000,1100)
            target=torch.as_tensor(tx).unsqueeze(1),ty,np.arange(2000,2050)
            a=SimpleNamespace(classes=['0','1'],seed=42,burst_dir=Path(temporary)/'clean',
                output=Path(temporary)/'out',action='All',device='cpu',iterations=1,tuning_epochs=1,dataset='IUST')
            view=SimpleNamespace(domain=lambda *args:source)
            descriptions={1:dict(attack='CW',strength=.1,level=1)}
            seen=[]
            def fit(source_x,source_y,target_x,target_y,classes,seed,parameters,device):
                seen.append(target_x.copy())
                self.assertEqual(len(source_y),50)
                self.assertEqual(len(target_y),10)
                return SimpleNamespace(extractor=object(),state_dict=lambda:{}),{}
            with redirect_stdout(io.StringIO()), \
                    patch.object(runner.shared,'attack_cases',return_value=({1:target},descriptions)), \
                    patch.object(runner,'fit',side_effect=fit), \
                    patch.object(runner,'embed',side_effect=lambda model,x,device:x), \
                    patch.object(runner,'finish'):
                runner.run_af(a,view,dict(test=True))
                runner.run_af(a,view,dict(test=True))
            self.assertEqual(len(seen),1)
            m=runner.shared.read(a.output/'Results/CW/level_1/metrics.json')
            self.assertEqual(m['labels'],10)
            self.assertEqual(m['cap'],10)
            self.assertEqual(m['samples'],40)
            self.assertIsNone(m['original_forgetting_AC'])
            self.assertFalse(set(m['calibration_ids']) & set(m['query_ids']))
            self.assertEqual(m['k'],5)


if __name__=='__main__':
    unittest.main()
