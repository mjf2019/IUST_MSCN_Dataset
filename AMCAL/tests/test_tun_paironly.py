from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
import amcal_tun_paironly as tun


class TUNPairOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.backends.nnpack.set_flags(False)

    def fixture(self,root):
        rng=np.random.default_rng(53);n=480
        role=np.tile([1,2,3,3],n//4)
        x=rng.normal(size=(n,8));x[:,0]+=(role!=3)*.6
        frame=pd.DataFrame(x,columns=[f'F{i}' for i in range(8)])
        frame.loc[0,'F1']=np.nan
        frame['id']=np.arange(n);frame['SrcIP']=['10.0.0.1']*n
        frame['DstPort']=80;frame['label_3']=role;frame['label_2']=role!=3
        frame['label_poly']=[f'{v}_Apache' for v in role]
        path=root/'ASNM-TUN.csv';frame.to_csv(path,index=False)
        return path

    def test_real_training_no_test_access_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.fixture(root);out=root/'run'
            args=['--csv',str(path),'--output',str(out),'--device','cpu','--max-features','8',
                '--cnn-epochs','1','--context-epochs','1','--warm-epochs','1','--pair-epochs','1','--patience','1']
            original=path.read_bytes();load=tun.load_domain
            def no_test(prepared,name,device):
                self.assertNotIn('test',name)
                return load(prepared,name,device)
            with redirect_stdout(io.StringIO()),patch.object(tun,'load_domain',side_effect=no_test):
                tun.main([*args,'--action','Train'])
            self.assertFalse((out/'Results').exists())
            artifact=joblib.load(out/'Prepared/preprocessing.joblib')
            self.assertTrue(all(not tun.excluded(c) for c in artifact['feature_names']))
            self.assertEqual(len(artifact['feature_names']),8)
            pools=artifact['pools'];groups=artifact['row_groups']
            for name in ('val','test'):
                self.assertFalse(set(groups[pools['target_'+name]]) & set(groups[pools['source_'+name]]))
                self.assertFalse(set(groups[pools['target_'+name]]) & set(groups[pools['audit_'+name]]))
            warm=torch.load(out/'Models/warmup.pth',weights_only=False)
            final=torch.load(out/'Models/paironly.pth',weights_only=False)
            for k,v in warm['policy_state_dict'].items():
                if not k.startswith('update_value.'):
                    self.assertTrue(torch.equal(v,final['policy_state_dict'][k]),k)
            with redirect_stdout(io.StringIO()):tun.main([*args,'--action','Evaluate'])
            result=json.loads((out/'Results/summary.json').read_text())
            self.assertTrue(result['classifier_unchanged']);self.assertLessEqual(result['labels'],result['cap'])
            self.assertAlmostEqual(result['source_drop_pp'],100*(result['source_initial_accuracy']-result['source_final_accuracy']))
            metrics=json.loads((out/'Results/stream_metrics.json').read_text())
            self.assertTrue(set(metrics['queried_source_ids'])<=set(pools['target_test']))
            self.assertFalse(set(metrics['source_ids'])&set(pools['audit_test']))
            self.assertEqual(path.read_bytes(),original)
            with redirect_stdout(io.StringIO()),patch.object(tun,'train',side_effect=AssertionError('No retraining')):
                tun.main(args)

    def test_known_tun_role_counts_have_valid_seed42_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(1)
            role=np.array([1]*130+[2]*87+[3]*177)
            frame=pd.DataFrame(rng.normal(size=(394,8)),columns=[f'F{i}' for i in range(8)])
            frame['label_3']=role;frame['label_2']=role!=3
            path=root/'ASNM-TUN.csv';frame.to_csv(path,index=False)
            artifact=tun.prepare(path,root/'prepared',42,8)
            self.assertEqual(set(artifact['pools']),{'source_train','target_train','source_val','target_val','audit_val','source_test','target_test','audit_test'})
            self.assertTrue(all(len(v)>0 for v in artifact['pools'].values()))

    def test_train_only_imputation_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=self.fixture(root)
            with redirect_stdout(io.StringIO()):a=tun.prepare(path,root/'prepared',42,8)
            frame=pd.read_csv(path);ids=np.concatenate([a['pools'][k] for k in ('source_test','target_test','audit_test')])
            frame.loc[ids,'F1']=1e15
            other=root/'altered.csv';frame.to_csv(other,index=False)
            with redirect_stdout(io.StringIO()):b=tun.prepare(other,root/'altered_prepared',42,8)
            np.testing.assert_array_equal(a['pools']['source_train'],b['pools']['source_train'])
            np.testing.assert_allclose(a['median'],b['median'])
            np.testing.assert_allclose(a['scaler'].mean_,b['scaler'].mean_)


if __name__=='__main__':unittest.main()
