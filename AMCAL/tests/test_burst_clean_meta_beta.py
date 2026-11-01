"""User-run CPU contracts; no official data, installation or training required."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import amcal_burst_clean_meta_beta as runner
from test_clean_meta_beta import CleanFixture


class BurstCleanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.api.base.legacy.core.device = torch.device('cpu')

    def test_all_levels_are_selected_including_zero_one_three_and_extra_levels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for level in [*runner.EXPECTED_LEVELS, 25]:
                (root/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').touch()
                (root/f'AdvBurstTrainedBurst_size_{level}_Step_2000_rows.csv').touch()
            self.assertEqual(runner.levels_on_disk(root), [*runner.EXPECTED_LEVELS, 25])

    def test_missing_expected_intensity_fails_without_silently_skipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                runner.levels_on_disk(Path(temporary))

    def test_dataset_source_fit_is_fresh_light_and_clean_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template = root/'template'; (template/'Models').mkdir(parents=True)
            (template/'Models/protocol_preprocessing.joblib').write_bytes(b'metadata')
            burst = root/'domains'; burst.mkdir()
            (burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv').write_text('clean', encoding='utf-8')
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            import pandas as pd
            frame = pd.DataFrame(np.arange(320).reshape(40,8), columns=[f'f{i}' for i in range(8)])
            frame['label'] = np.arange(40)%2
            artifact = dict(feature_names=list(frame.columns[:-1]), train_ids=np.arange(24),
                val_ids=np.arange(24,32), test_ids=np.arange(32,40),
                scaler=StandardScaler().fit(frame.iloc[:24,:-1]),
                label_encoder=LabelEncoder().fit([0,1]))
            a = SimpleNamespace(output=root/'out', source_template=template, burst_dir=burst,
                dataset='IAT', seed=42, action='Train', context_lr=.00025,
                cnn_epochs=1, context_epochs=1, patience=1)
            touched = []
            def read_domain(directory, saved, level, split):
                touched.append((level, split))
                self.assertEqual(level, 0)
                ids = saved[split]
                return torch.randn(len(ids),1,8), np.arange(len(ids))%2, ids
            def fit_cnn(args, cnn, ignored, train, validation):
                self.assertIsInstance(cnn, runner.LightCNN1D)
                self.assertIsNone(ignored)
                return [dict(epoch=1)]
            with redirect_stdout(io.StringIO()), \
                    patch.object(runner.api.base, 'load_artifact', return_value=artifact), \
                    patch.object(runner.api.base.legacy.base.official, 'mapped_domain',
                        return_value=(frame,np.arange(40),None)), \
                    patch.object(runner.api.base.legacy, 'read_domain', side_effect=read_domain), \
                    patch.object(runner.source_fit, 'fit', side_effect=fit_cnn), \
                    patch.object(runner, 'fit_clean_context', return_value=[dict(epoch=1)]):
                recipe = runner.source_stage(a)
            self.assertEqual(recipe['training_levels'], [0])
            self.assertEqual(recipe['validation_levels'], [0])
            self.assertEqual(recipe['architecture'], runner.ARCHITECTURE)
            self.assertEqual(touched, [(0,'train_ids'),(0,'val_ids')])
            saved = torch.load(a.output/'source/Models/initial_cnn_model.pth', weights_only=False)
            self.assertEqual(saved['training_levels'], [0])
            self.assertEqual(saved['architecture'], runner.ARCHITECTURE)

    def test_test_covers_all_levels_resets_context_and_keeps_ddqn_beta_frozen(self):
        from enhancements.meta_beta_ddqn import MetaBetaDDQN
        with tempfile.TemporaryDirectory() as temporary:
            exp = CleanFixture(Path(temporary)/'run')
            exp.a.dataset = 'IAT'; exp.a.burst_dir = Path(temporary)/'domains'
            exp.a.burst_dir.mkdir(); exp.a.protocol = 'Online'
            for level in runner.EXPECTED_LEVELS:
                (exp.a.burst_dir/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv').touch()
            target = (torch.randn(40,1,8), np.arange(40)%2, np.arange(9000,9040))
            exp.artifact = dict(test_ids=target[2])
            exp.domain = lambda directory, level, split: target
            agent = MetaBetaDDQN(dict(mean=[0.]*22,scale=[1.]*22), None, 'cpu', 42)
            exp.a.beta = agent.beta_value()
            with torch.no_grad():
                agent.policy.network[-1].bias.copy_(torch.tensor([0.,1.]))
            original_context = exp.context; calls = []
            def new_context():
                value = original_context(); calls.append(value); return value
            exp.context = new_context
            with redirect_stdout(io.StringIO()), patch.object(runner.shared, 'load_selected',
                    return_value=(agent,dict(sha256='fixture'))):
                runner.evaluate(exp, dict(training_levels=[0]))
            self.assertEqual(len(calls), len(runner.EXPECTED_LEVELS))
            self.assertEqual(len({id(context) for context in calls}), len(calls))
            self.assertEqual((agent.steps,agent.updates), (0,0))
            rows = []
            for level in runner.EXPECTED_LEVELS:
                metric = runner.shared.read(exp.a.output/f'Results/Online/level_{level}/metrics.json')
                self.assertEqual(metric['samples'],40)
                self.assertEqual(metric['cap'],8)
                self.assertEqual(metric['labels'],8)
                rows.append(metric['online'])
            self.assertTrue(all(row == rows[0] for row in rows))


if __name__ == '__main__':
    unittest.main()
