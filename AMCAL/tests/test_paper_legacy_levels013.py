"""Source partition isolation and real reuse of the existing binary runner."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_paper_legacy_levels013 as runner


class LegacyDataAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self, root):
        burst=root/'burst'; burst.mkdir(); source=root/'source'; (source/'Models').mkdir(parents=True)
        rng=np.random.default_rng(42); raw=rng.normal(size=(60,8)); labels=np.arange(60)%6
        for level in (*runner.LEVELS,*runner.TEST_LEVELS):
            path=burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
            frame=pd.DataFrame(raw+level*.01,columns=[f'f{i}' for i in range(8)])
            frame['label']=labels; frame.to_csv(path,index=False)
            pd.DataFrame({'source_row':np.arange(60)}).to_csv(path.with_name(path.stem+'_rows.csv'),index=False)
        artifact=dict(source_sha256=runner.sha(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'),
            feature_names=[f'f{i}' for i in range(8)],label_name='label',source_rows=60,source_labels=labels,
            train_ids=np.arange(36),val_ids=np.arange(36,48),test_ids=np.arange(48,60))
        joblib.dump(artifact,source/'Models'/'protocol_preprocessing.joblib')
        return burst,source

    def test_prepared_domains_preserve_disjoint_id_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); burst,source=self.fixture(root); output=root/'run'
            data=runner.prepare(burst,source,output)
            base=pd.read_csv(data/'oversampled_train_dataset.csv')
            joint=pd.read_csv(data/'context_train_levels013.csv')
            self.assertEqual(len(base),36); self.assertEqual(len(joint),108)
            self.assertEqual(len(pd.read_csv(data/'context_validation_levels013.csv')),36)
            self.assertTrue(pd.read_csv(data/'oversampled_test_dataset.csv').equals(
                pd.read_csv(data/'oversampled_validation_dataset.csv')))
            protocol=json.loads((output/'legacy_data_protocol.json').read_text())
            self.assertFalse(set(protocol['train_ids'])&set(protocol['test_ids']))
            self.assertFalse(set(protocol['val_ids'])&set(protocol['test_ids']))
            bad=burst/'AdvBurstTrainedBurst_size_1_Step_2000_rows.csv'
            mapping=pd.read_csv(bad); mapping.loc[0,'source_row']=1; mapping.to_csv(bad,index=False)
            with self.assertRaises(ValueError): runner.prepare(burst,source,root/'bad')

    def test_trace_summary_keeps_pre_feedback_and_legacy_scoring_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.csv'
            pd.DataFrame(dict(true_label=[0,1],prediction_before_update=[0,0],
                prediction_after_update=[0,1],base_prediction=[1,0],labels_used=[0,1])).to_csv(path,index=False)
            result=runner.trace_summary(path,5)
            self.assertEqual(result['accuracy_before_feedback_pct'],50.)
            self.assertEqual(result['accuracy_after_feedback_pct'],100.)
            self.assertEqual(result['labels'],1)

    def test_actual_legacy_training_and_one_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); burst,source=self.fixture(root); output=root/'run'
            common=['--source-run',str(source),'--burst-dir',str(burst),'--output',str(output),
                '--device','cpu','--seed','42']
            legacy_path=Path(__file__).resolve().parents[1]/'ANT_Burst_TEST'/'amcal_burst_legacy.py'
            before=runner.sha(legacy_path)
            with redirect_stdout(io.StringIO()):
                runner.main(['train',*common,'--base-epochs','1','--context-epochs','1','--patience','1'])
                with patch.object(runner,'TEST_LEVELS',(5,)): runner.main(['evaluate',*common])
            report=json.loads((output/'legacy_execution.json').read_text())
            self.assertEqual(report['train']['implementation_sha256'],before)
            self.assertEqual(report['evaluate']['implementation_sha256'],before)
            self.assertEqual(runner.sha(legacy_path),before)
            self.assertEqual(report['evaluate']['reward'],'paper')
            self.assertEqual(report['evaluate']['transition'],'local')
            self.assertEqual(report['evaluate']['query_gate'],'none')
            training=json.loads((output/'legacy_training_summary.json').read_text())
            self.assertEqual(len(training),3)
            import amcal_burst_legacy as legacy
            self.assertEqual(legacy.train_adversarial_system.__module__,'amcal_burst_legacy')
            summary=pd.read_csv(output/'Results'/'legacy_levels_summary.csv')
            self.assertEqual(summary['samples'].tolist(),[12])
            self.assertLessEqual(int(summary.iloc[0]['labels']),2)


if __name__=='__main__': unittest.main()
