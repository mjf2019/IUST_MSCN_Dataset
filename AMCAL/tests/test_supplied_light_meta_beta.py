"""User-run contracts for the final supplied-data Light architecture control."""
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
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import amcal_supplied_light_meta_beta as runner


class SuppliedLightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.api.base.legacy.core.device = torch.device('cpu')

    def test_shared_query_learning_and_evaluation_are_used_without_overrides(self):
        import amcal_clean_meta_beta as reference
        self.assertIs(runner.shared.train,reference.train)
        self.assertIs(runner.shared.evaluate,reference.evaluate)
        self.assertIs(runner.LightExperiment.bind,reference.CleanExperiment.bind)
        self.assertIs(runner.LightExperiment.context,reference.CleanExperiment.context)
        self.assertIs(runner.LightExperiment.memory,reference.CleanExperiment.memory)

    def test_source_fits_only_clean_data_with_light_cnn_and_copies_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); original = root/'original'
            models = original/'source/Models'; models.mkdir(parents=True)
            preprocessor = models/'protocol_preprocessing.joblib'
            preprocessor.write_bytes(b'unchanged TRAIN-only preprocessing fixture')
            a = SimpleNamespace(output=root/'light',source_root=original,burst_dir=original/'TrainingDomains',
                dataset='IUST',seed=42,action='Train',cnn_epochs=2,context_epochs=2,patience=1)
            artifact = dict(feature_names=[f'f{i}' for i in range(8)],
                label_encoder=LabelEncoder().fit([0,1]))
            manifest = dict(recipe=dict(training_levels=[0],validation_levels=[0]))
            reads = []; fits = []
            def domain(directory,saved,level,split):
                reads.append((level,split)); self.assertEqual(level,0)
                return torch.randn(40,1,8),np.arange(40)%2,np.arange(40)
            def fit(args,model,cnn,train,val,context=False):
                fits.append(context)
                if context:
                    self.assertIsInstance(cnn,runner.LightCNN1D)
                    self.assertFalse(cnn.training)
                    self.assertFalse(any(p.requires_grad for p in cnn.parameters()))
                else:
                    self.assertIsInstance(model,runner.LightCNN1D)
                    self.assertIsNone(cnn)
                return [dict(epoch=1,validation_loss=1.)]
            with redirect_stdout(io.StringIO()), \
                    patch.object(runner.api.base.legacy,'read_domain',side_effect=domain), \
                    patch.object(runner.source_fit,'fit',side_effect=fit):
                runner.source_stage(a,manifest,artifact)
                runner.source_stage(a,manifest,artifact)
            self.assertEqual(reads,[(0,'train_ids'),(0,'val_ids')])
            self.assertEqual(fits,[False,True])
            self.assertEqual(preprocessor.read_bytes(),
                (a.output/'source/Models/protocol_preprocessing.joblib').read_bytes())
            saved = torch.load(a.output/'source/Models/initial_cnn_model.pth',weights_only=False)
            self.assertEqual(saved['architecture'],runner.ARCHITECTURE)
            self.assertEqual(saved['training_levels'],[0])

    def test_clean_reservations_do_not_depend_on_the_cnn_architecture(self):
        torch.manual_seed(42)
        full = torch.randn(1200,1,8),np.arange(1200)%2,np.arange(1200)
        a = runner.shared.reserve_clean(full,42,2,300)
        b = runner.shared.reserve_clean(full,42,2,300)
        for role in a:
            self.assertEqual(runner.base.domain_hash(a[role]),runner.base.domain_hash(b[role]))
        self.assertGreater(len(a['inner'][1]),300)


if __name__ == '__main__':
    unittest.main()
