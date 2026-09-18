import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cdr_mlc import (BASE_FEATURES, SENSITIVE, STATS, CDRMLC, Config,
                     trend_features, evaluate, universal_clustering_pipeline)
from run_cdr_mlc import run_experiment


def fixture():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(rng.uniform(.1, 1, (180, len(BASE_FEATURES))), columns=BASE_FEATURES)
    df['label'] = np.tile(['HTTP', 'SMTP'], 90)
    df['SrcBytes'] = np.tile([1., 100.], 90)
    level = np.repeat([0, 1, 2], 60)
    for col in SENSITIVE:
        df[col] = level * 10 + rng.uniform(.01, .2, len(df))
    df['capture_id'] = level.astype(str)
    df['label_encoded'] = np.tile([0, 1], 90)
    df['level_label'] = level
    df['index_in_label_under_level'] = np.arange(len(df))
    return df


class TestCDRMLC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = fixture()
        cls.model = CDRMLC(Config(n_jobs=1), stream_column='capture_id').fit(cls.df)

    def test_window_has_15_features_and_expected_statistics(self):
        df = pd.DataFrame({c: [1., 4., 2., 8.] for c in SENSITIVE})
        out = trend_features(df)
        self.assertEqual(out.shape, (4, 15))
        expected = [8, 2, 4, 14/3, np.std([4, 2, 8], ddof=1)]
        np.testing.assert_allclose(out.iloc[3, :5], expected)
        self.assertEqual(out.iloc[0]['AckDat_std'], 0)
        np.testing.assert_allclose(out.iloc[2]['AckDat_median'], 2)

    def test_causal_and_duplicate_index_safe(self):
        df = self.df.copy()
        df.index = np.zeros(len(df), dtype=int)
        original = trend_features(df)
        df.iloc[50:, df.columns.get_loc('AckDat')] = 1e9
        np.testing.assert_allclose(original.iloc[:50], trend_features(df).iloc[:50])
        self.assertEqual(len(original), len(df))

    def test_resets_only_at_capture_boundaries(self):
        df = pd.DataFrame({c: [1., 2., 100., 200., 7.] for c in SENSITIVE})
        df['capture_id'] = ['a', 'a', 'b', 'b', 'a']
        out = trend_features(df, stream_column='capture_id')
        self.assertEqual(out.iloc[2]['AckDat_mean'], 100)
        self.assertEqual(out.iloc[4]['AckDat_mean'], 7)

    def test_32_inputs_and_three_twenty_tree_experts(self):
        self.assertEqual(len(self.model.classification_features), 32)
        self.assertEqual(len(self.model.experts_), 3)
        for expert in self.model.experts_.values():
            self.assertEqual(len(expert.estimators_), 20)
        self.assertFalse(set(SENSITIVE).intersection(self.model.classification_features))
        self.assertNotIn('label_encoded', self.model.classification_features)
        self.assertNotIn('level_label', self.model.classification_features)

    def test_training_router_matches_inference_without_reassignment(self):
        _, routed = self.model.predict_with_clusters(self.df)
        np.testing.assert_array_equal(self.model.training_clusters_, routed)

    def test_predictions_ignore_labels_and_metadata(self):
        expected = self.model.predict(self.df)
        altered = self.df.copy()
        altered['label'] = 'UNSEEN'
        altered['label_encoded'] = 99
        altered['level_label'] = -100
        altered['index_in_label_under_level'] = 1e9
        np.testing.assert_array_equal(expected, self.model.predict(altered))
        np.testing.assert_array_equal(expected, self.model.predict(altered.drop(columns=['label', 'label_encoded'])))

    def test_evaluation_uses_only_routed_expert(self):
        metrics, preds, ids = evaluate(self.model, self.df)
        for cid, expert in self.model.experts_.items():
            mask = ids == cid
            expected = expert.predict(self.df.loc[mask, list(self.model.classification_features)])
            np.testing.assert_array_equal(preds[mask], expected)
        changed = self.df.copy()
        changed['label'] = 'NEVER_SEEN'
        other, preds2, _ = evaluate(self.model, changed)
        np.testing.assert_array_equal(preds, preds2)
        self.assertEqual(other['accuracy'], 0)
        self.assertGreater(metrics['accuracy'], .95)

    def test_stream_batch_and_saved_model_agree(self):
        data = self.df.iloc[55:70].drop(columns=['label', 'label_encoded'])
        preds, ids = self.model.predict_with_clusters(data)
        stream = self.model.stream()
        actual = [stream.predict_one(r) for r in data.to_dict('records')]
        np.testing.assert_array_equal(preds, [r['prediction'] for r in actual])
        np.testing.assert_array_equal(ids, [r['cluster_id'] for r in actual])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.joblib'
            joblib.dump(self.model, path)
            np.testing.assert_array_equal(preds, joblib.load(path).predict(data))

    def test_validation_and_unsafe_options(self):
        with self.assertRaises(ValueError):
            CDRMLC(feature_columns=[*BASE_FEATURES, 'label_encoded'])
        with self.assertRaises(ValueError):
            CDRMLC(stream_column='level_label')
        with self.assertRaises(ValueError):
            universal_clustering_pipeline(self.df, use_at_least_one_rule=True)
        with self.assertRaises(ValueError):
            universal_clustering_pipeline(self.df, ensure_label_coverage=True)
        with self.assertRaises(ValueError):
            self.model.predict(self.df.drop(columns=['SynAck']))
        bad = self.df.copy()
        bad.loc[0, 'AckDat'] = np.inf
        with self.assertRaises(ValueError):
            self.model.predict(bad)
        with self.assertRaises(ValueError):
            CDRMLC().fit(pd.DataFrame({**{c: np.zeros(10) for c in BASE_FEATURES}, 'label': ['a']*10}))

    def test_test_data_does_not_change_fitted_state(self):
        center = self.model.router_.cluster_centers_.copy()
        mean = self.model.scaler_.mean_.copy()
        test = self.df.copy()
        test[list(SENSITIVE)] *= 1000
        evaluate(self.model, test)
        np.testing.assert_array_equal(center, self.model.router_.cluster_centers_)
        np.testing.assert_array_equal(mean, self.model.scaler_.mean_)

    def test_missing_classes_are_reported_not_moved(self):
        df = self.df.copy()
        df['label'] = np.repeat(['A', 'B', 'C'], 60)
        with warnings.catch_warnings(record=True) as caught:
            model = CDRMLC(Config(n_jobs=1), stream_column='capture_id').fit(df)
        self.assertTrue(caught)
        self.assertTrue(all(model.missing_classes_.values()))
        np.testing.assert_array_equal(model.training_clusters_, model.predict_with_clusters(df)[1])
        np.testing.assert_array_equal(model.router_.cluster_centers_, self.model.router_.cluster_centers_)

    def test_custom_target_and_report_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train, test = self.df.copy(), self.df.copy()
            train = train.rename(columns={'label': 'service'})
            test = test.rename(columns={'label': 'service'})
            train.to_csv(root / 'train.csv', index=False)
            test.to_csv(root / 'test.csv', index=False)
            report = run_experiment([root / 'train.csv'], [root / 'test.csv'], root / 'out',
                                    Config(n_jobs=1), target='service', stream_column='capture_id',
                                    level_column='level_label', silhouette_samples=100)
            self.assertGreater(report['metrics']['accuracy'], .95)
            self.assertIn('level_adjusted_rand', report['test_diagnostics'])
            for name in ('report.json', 'predictions.csv', 'train_cluster_class_counts.csv', 'model.joblib'):
                self.assertTrue((root / 'out' / name).is_file())
            json.loads((root / 'out/report.json').read_text())
            with self.assertRaises(ValueError):
                run_experiment([root / 'train.csv'], [root / 'train.csv'], root / 'out')


if __name__ == '__main__':
    unittest.main()
