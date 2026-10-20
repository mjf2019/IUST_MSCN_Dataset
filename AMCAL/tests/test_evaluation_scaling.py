import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.evaluation_scaling import notebook_scaler
from baselines.adversarial_domains import domain_name

class ScalingTests(unittest.TestCase):
    def test_exact_old_notebook_rule_and_unchanged_training_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=pd.DataFrame(dict(a=[1.,3.,5.],b=[4.,4.,4.],label=['x','y','x']))
            later=pd.DataFrame(dict(a=[100.,200.,300.],b=[10.,20.,30.],label=['y','x','y']))
            first.to_csv(root/domain_name(1),index=False)
            later.to_csv(root/domain_name(2),index=False)
            original=StandardScaler().fit(pd.DataFrame(dict(a=[-10.,10.],b=[0.,2.])))
            before=original.mean_.copy()
            artifact=dict(feature_names=['a','b'],label_name='label',scaler=original)
            changed,info=notebook_scaler(root,artifact,[0,1,2])
            expected=StandardScaler().fit(first.iloc[:,:-1])
            np.testing.assert_allclose(changed['scaler'].transform(first.iloc[:,:-1]),expected.transform(first.iloc[:,:-1]))
            np.testing.assert_allclose(changed['scaler'].transform(later.iloc[:,:-1]),expected.transform(later.iloc[:,:-1]))
            self.assertFalse(np.allclose(changed['scaler'].transform(later.iloc[:,:-1]),StandardScaler().fit_transform(later.iloc[:,:-1])))
            self.assertIs(artifact['scaler'],original)
            np.testing.assert_equal(original.mean_,before)
            self.assertEqual(info['fit_level'],1)
            self.assertEqual(info['fit_rows'],3)
            self.assertEqual(changed['scaler'].scale_[1],1.)

    def test_fit_uses_first_requested_positive_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pd.DataFrame(dict(a=[10.,20.],label=[0,1])).to_csv(root/domain_name(3),index=False)
            changed,info=notebook_scaler(root,dict(feature_names=['a'],label_name='label'),[0,3,5])
            self.assertEqual(info['fit_level'],3)
            np.testing.assert_equal(changed['scaler'].mean_,[15.])

    def test_clean_only_request_has_no_attack_fit(self):
        with self.assertRaises(ValueError):notebook_scaler('.',{},[0])

if __name__=='__main__':unittest.main()
