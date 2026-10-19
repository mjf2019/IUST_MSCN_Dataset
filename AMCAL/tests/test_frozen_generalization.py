"""Diagnostic/selection/provenance checks that do not require a CUDA runtime."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / 'amcal_burst_frozen_generalization.py'
spec = importlib.util.spec_from_file_location('frozen_generalization', SOURCE)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


class FrozenGeneralizationTests(unittest.TestCase):
    def test_stable_cross_entropy_and_macro_metrics(self):
        metric = frozen.classification_metrics([[1000, 999], [1001, 1000]], [0, 1])
        self.assertAlmostEqual(metric['accuracy'], .5)
        self.assertAlmostEqual(metric['balanced_accuracy'], .5)
        self.assertAlmostEqual(metric['macro_f1'], 1 / 3)
        self.assertAlmostEqual(metric['ce'], np.log1p(np.exp(-1)) + .5)

    def test_balanced_accuracy_does_not_follow_majority_accuracy(self):
        metric = frozen.classification_metrics([[4, 0]] * 10, [0] * 9 + [1])
        self.assertAlmostEqual(metric['accuracy'], .9)
        self.assertAlmostEqual(metric['balanced_accuracy'], .5)

    def test_reject_nonfinite_and_invalid_truth(self):
        for logits, labels in [([[np.inf, 0]], [0]), ([[1, 0]], [2]),
                               ([[1, 0]], [.5]), ([], [])]:
            with self.subTest(logits=logits, labels=labels), self.assertRaises(ValueError):
                frozen.classification_metrics(logits, labels)

    def test_easy_domain_cannot_mask_worsening_worst_domain(self):
        before = [{'ce': .9}, {'ce': .3}, {'ce': .2}]
        after = [{'ce': 1.}, {'ce': .1}, {'ce': .1}]
        self.assertLess(frozen.validation_score(after, 'mean-loss'),
                        frozen.validation_score(before, 'mean-loss'))
        self.assertGreater(frozen.validation_score(after, 'worst-loss'),
                           frozen.validation_score(before, 'worst-loss'))

    def test_selection_uses_mean_as_tiebreak_and_rejects_missing_metric(self):
        self.assertLess(frozen.validation_score([{'ce': 1.}, {'ce': .1}], 'worst-loss'),
                        frozen.validation_score([{'ce': 1.}, {'ce': .3}], 'worst-loss'))
        for rows in ([], [{'ce': np.nan}]):
            with self.assertRaises(ValueError):
                frozen.validation_score(rows, 'worst-loss')

    def test_partition_overlap_and_fractional_ids_are_rejected(self):
        valid = dict(train_ids=[0, 1], val_ids=[2], test_ids=[3], source_rows=4)
        self.assertEqual(frozen.validate_partitions(valid)['test_ids'].tolist(), [3])
        for changed in (dict(val_ids=[1]), dict(test_ids=[4]), dict(train_ids=[.5, 1])):
            with self.assertRaises(ValueError):
                frozen.validate_partitions({**valid, **changed})

    def test_seen_domains_cannot_be_relabelled_as_heldout(self):
        self.assertEqual(frozen.validate_levels([7, 5], [0, 1, 3], True), [5, 7])
        for requested, evaluating in [([0, 1], False), ([3, 7], True), ([5, 5], True)]:
            with self.assertRaises(ValueError):
                frozen.validate_levels(requested, [0, 1, 3], evaluating)

    def test_batchnorm_tail_does_not_duplicate_or_drop_training_rows(self):
        for count in (2, 63, 64, 65, 66, 128, 129, 9690):
            slices = frozen.batch_slices(count)
            ids = [i for start, stop in slices for i in range(start, stop)]
            self.assertEqual(ids, list(range(count)))
            self.assertTrue(all(stop - start >= 2 for start, stop in slices))

    def test_domain_loader_uses_only_requested_source_partitions(self):
        # Domain row order deliberately differs from source order. TRAIN and
        # validation label transforms must never receive heldout source IDs 4/5.
        ids = np.array([5, 2, 0, 4, 1, 3])
        frame = pd.DataFrame({'a': ids, 'b': ids + 10, 'label': ids})
        touched = []

        class Encoder:
            def transform(self, labels):
                touched.extend(labels.tolist())
                return labels.to_numpy()

        class Tensor:
            def __init__(self, array):
                self.array = array

            def unsqueeze(self, axis):
                return np.expand_dims(self.array, axis)

        torch = SimpleNamespace(from_numpy=Tensor, long=np.int64,
                                as_tensor=lambda values, dtype: np.asarray(values, dtype=dtype))
        official = SimpleNamespace(mapped_domain=lambda *args: (frame, ids, 'explicit'))
        artifact = dict(train_ids=[0, 1], val_ids=[2, 3], test_ids=[4, 5],
                        scaler=SimpleNamespace(transform=lambda features: np.asarray(features)),
                        label_encoder=Encoder())
        data = frozen.load_domains(torch, official, Path('.'), artifact, [0], ('train_ids', 'val_ids'))
        self.assertEqual(touched, [0, 1, 2, 3])
        np.testing.assert_equal(data[0, 'train_ids'][0][:, 0, 0], [0, 1])
        np.testing.assert_equal(data[0, 'val_ids'][0][:, 0, 0], [2, 3])
        self.assertNotIn((0, 'test_ids'), data)


if __name__ == '__main__':
    unittest.main()
