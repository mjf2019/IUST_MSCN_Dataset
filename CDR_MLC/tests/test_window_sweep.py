import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_sweep import TIMING, prepare_partitions, window_matrix, training_mapping, fixed_sample


def fixture(n=100):
    d = pd.DataFrame({c: np.arange(n, dtype=float) for c in TIMING})
    d['flow_key'] = [f'flow{i}' for i in range(n)]
    d['timing_valid'] = True
    d['capture'] = 'synthetic'
    d['timestamp'] = pd.date_range('2026-01-01', periods=n, freq='s')
    return d


class WindowSweepTests(unittest.TestCase):
    def test_causal_statistics(self):
        d = fixture()
        parts, _ = prepare_partitions(d, 5, 'none')
        g = parts['train']
        x = window_matrix(g, 3)
        # First shared endpoint is raw index 4: history [2,3,4].
        np.testing.assert_allclose(x[0, :5], [3, 4, 3, 2, np.std([2,3,4])])
        changed = g.copy()
        changed.loc[20:, TIMING] = 999
        np.testing.assert_allclose(x[:16], window_matrix(changed, 3)[:16])

    def test_all_windows_have_identical_endpoints(self):
        parts, _ = prepare_partitions(fixture(), 10, 'none')
        for g in parts.values():
            sizes = [len(window_matrix(g, w)) for w in [1,3,5,10]]
            self.assertEqual(len(set(sizes)), 1)
            self.assertEqual(sizes[0], len(g)-9)
            self.assertTrue(np.all(window_matrix(g, 1)[:, 4::5] == 0))

    def test_no_history_crosses_partition(self):
        parts, _ = prepare_partitions(fixture(), 3, 'none')
        v = window_matrix(parts['validation'], 3)
        np.testing.assert_allclose(v[0, :5], [61, 62, 61, 60, np.std([60,61,62])])

    def test_shared_tuples_are_purged_and_gaps_not_compressed(self):
        d = fixture()
        d.loc[5, 'flow_key'] = d.loc[65, 'flow_key']
        d.loc[70, 'flow_key'] = d.loc[90, 'flow_key']
        parts, audit = prepare_partitions(d, 3, 'five-tuple')
        tr, va = parts['train'], parts['validation']
        self.assertFalse(tr.eligible.iloc[5])
        self.assertFalse(tr.common_endpoint.iloc[5:8].any())
        self.assertFalse(va.eligible.iloc[10])
        self.assertFalse(va.common_endpoint.iloc[10:13].any())
        self.assertEqual(audit['train_validation_shared_tuples'], 0)
        _, dependent = prepare_partitions(d, 3, 'none')
        self.assertEqual(dependent['train_validation_shared_tuples'], 1)

    def test_invalid_timing_breaks_history(self):
        d = fixture()
        d.loc[10, 'timing_valid'] = False
        d.loc[10, TIMING] = np.nan
        parts, _ = prepare_partitions(d, 5, 'none')
        self.assertFalse(parts['train'].common_endpoint.iloc[10:15].any())
        self.assertTrue(np.isfinite(window_matrix(parts['train'], 3)).all())

    def test_training_mapping_is_permutation_invariant(self):
        levels = np.repeat([0,1,2], [20,10,30])
        clusters = np.array([2,0,1])[levels]
        mapping, table = training_mapping(levels, clusters)
        np.testing.assert_array_equal(mapping[clusters], levels)
        # Later validation contingency cannot change the frozen mapping.
        frozen = mapping.copy()
        validation_clusters = np.array([0,2,1])
        np.testing.assert_array_equal(mapping[validation_clusters], [1,0,2])
        np.testing.assert_array_equal(frozen, mapping)

    def test_labels_not_in_feature_matrix(self):
        d = fixture()
        parts, _ = prepare_partitions(d, 3, 'none')
        g = parts['train']; before = window_matrix(g, 3)
        g['level'] = 'fake'; g['application'] = 'fake'
        np.testing.assert_array_equal(before, window_matrix(g, 3))
        self.assertEqual(before.shape[1], 15)

    def test_common_silhouette_sample_reproducible(self):
        np.testing.assert_array_equal(fixed_sample(100, 20), fixed_sample(100, 20))
        self.assertEqual(len(np.unique(fixed_sample(100,20))),20)


if __name__ == '__main__':
    unittest.main()
