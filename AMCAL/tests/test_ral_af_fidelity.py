"""Primary-source parity and leakage checks for the RAL/AF audit."""
import ast
from contextlib import redirect_stdout
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import amcal_burst_baselines as matched
import amcal_af_paper_burst as runner
from baselines import af_author_protocol as af


class FidelityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        matched.core.device = torch.device('cpu')

    def test_every_ral_core_method_matches_pinned_author_ast(self):
        reference = json.loads((Path(matched.__file__).parent/'baselines/ral_author_fingerprint.json').read_text())
        cls = ast.parse(inspect.getsource(matched.RAL)).body[0]
        actual = {}
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name not in reference['methods']:
                continue
            if (isinstance(method.body[0], ast.Expr) and isinstance(method.body[0].value, ast.Constant)
                    and isinstance(method.body[0].value.value, str)):
                method.body.pop(0)
            actual[method.name] = hashlib.sha256(ast.dump(method,include_attributes=False).encode()).hexdigest()
        self.assertEqual(actual, reference['methods'])

    def test_legacy_af_cannot_be_silently_called_original(self):
        with self.assertRaisesRegex(ValueError, 'NOT the original AF'):
            matched.require_explicit_af_extension(SimpleNamespace(methods=['af']))
        matched.require_explicit_af_extension(SimpleNamespace(methods=['af'],allow_af_extension=True))
        self.assertIn('NOT-paper', matched.METHOD_NAMES['af'])

    def test_af_vote_is_target_only_distance_cosine_k_equals_n(self):
        support = np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]])
        model = af.target_knn(support, np.array([0,0,1,1]), 2)
        self.assertEqual(model.n_neighbors, 2)
        self.assertEqual(model.metric,'cosine')
        self.assertEqual(model.weights,'distance')
        self.assertEqual(model.n_samples_fit_,4)
        np.testing.assert_array_equal(model.predict([[1.,.01],[.01,1.]]),[0,1])
        # Refuse undersized classes, rather than reducing N or oversampling.
        with self.assertRaises(ValueError):
            af.per_class_indices([0,0,1],[0,1],2,np.random.default_rng(42))

    def test_author_alternating_step_restores_extractor_bn_in_second_pass(self):
        p=af.AFParameters(iterations=1,embedding_dim=8)
        net=af.AFNetwork(2,p)
        first=torch.optim.Adam(list(net.extractor.parameters())+list(net.classifier.parameters()),lr=p.learning_rate)
        second=torch.optim.Adam(net.discriminator.parameters(),lr=p.learning_rate)
        snapshots={}
        original=first.step
        def snapshot():
            result=original()
            snapshots.update({k:v.detach().clone() for k,v in net.extractor.state_dict().items()})
            return result
        x=np.random.default_rng(42).normal(size=(4,8)).astype(np.float32)
        with patch.object(first,'step',side_effect=snapshot):
            af.author_adversarial_step(net,first,second,x[:2],np.array([0,1]),x[2:],p)
        for key,value in net.extractor.state_dict().items():
            torch.testing.assert_close(value,snapshots[key],rtol=0,atol=0)

    def source(self,root):
        source=root/'source';(source/'Models').mkdir(parents=True)
        burst=root/'burst';burst.mkdir()
        raw=np.random.default_rng(42).normal(size=(180,8)).astype(np.float32)
        labels=np.arange(180)%2
        features=[f'f{i}' for i in range(8)]
        for level in (0,5):
            path=burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
            frame=pd.DataFrame(raw+level*.02,columns=features);frame['label']=labels
            frame.to_csv(path,index=False)
            pd.DataFrame({'source_row':np.arange(180)}).to_csv(path.with_name(path.stem+'_rows.csv'),index=False)
        artifact=dict(feature_names=features,label_name='label',source_rows=180,source_labels=labels,
                      train_ids=np.arange(120),val_ids=np.arange(120,140),test_ids=np.arange(140,180),
                      scaler=StandardScaler().fit(raw[:120]),label_encoder=LabelEncoder().fit(labels),
                      source_sha256=matched.core.file_sha256(burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv'))
        joblib.dump(artifact,source/'Models/protocol_preprocessing.joblib')
        return SimpleNamespace(source_run=source,burst_dir=burst,output=root/'new_af',seed=42,
                               iterations=1,tuning_epochs=1,levels=[5],shots=[1,5],test_per_class=0,device='cpu')

    def test_real_csv_partition_training_checkpoint_and_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            a=self.source(Path(directory))
            with redirect_stdout(io.StringIO()):
                runner.run(a)
            for shots in (1,5):
                root=a.output/f'level5_shot{shots}'
                result=json.loads((root/'metrics.json').read_text())
                self.assertEqual(result['source_labels'],50)
                self.assertEqual(result['target_calibration_labels'],2*shots)
                self.assertEqual(result['online_labels'],0)
                self.assertFalse(set(result['test_ids']) & set(result['source_ids']))
                self.assertFalse(set(result['test_ids']) & set(result['target_calibration_ids']))
                self.assertFalse(set(result['source_ids']) & set(result['target_calibration_ids']))
                state=torch.load(root/'af.pth',map_location='cpu',weights_only=False)
                model=af.AFNetwork(state['classes'],af.AFParameters(**state['parameters']))
                model.load_state_dict(state['state'])
                self.assertTrue(torch.isfinite(model.extractor.features.weight).all())
                self.assertEqual(len(pd.read_csv(root/'predictions.csv')),40)
            with self.assertRaisesRegex(ValueError,'fresh'):
                runner.run(a)
            # Failed preflight leaves no partial experiment directory.
            a.output=a.output.parent/'invalid';a.test_per_class=70
            with self.assertRaisesRegex(ValueError,'need 70 rows'):
                runner.run(a)
            self.assertFalse(a.output.exists())


if __name__ == '__main__':
    unittest.main()
