"""User-run contract tests; no AF training or checkpoint loading."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_af_clean_reference as audit


class CleanReferenceContracts(unittest.TestCase):
    def test_source_calibration_counts_and_separation(self):
        x = np.arange(800, dtype=np.float32).reshape(400, 2)
        y = np.repeat([0, 1], 200)
        ids = np.arange(400)
        si, ci = audit.clean_split(x, y, ids, np.arange(2), 42)
        self.assertEqual([int(np.sum(y[si] == c)) for c in range(2)], [25, 25])
        self.assertEqual(len(ci), 80)
        self.assertFalse(set(ids[si]) & set(ids[ci]))
        self.assertFalse({r.tobytes() for r in x[si]} & {r.tobytes() for r in x[ci]})
        other = audit.clean_split(x, y, ids, np.arange(2), 42)
        np.testing.assert_array_equal(si, other[0])
        np.testing.assert_array_equal(ci, other[1])

    def test_reference_excludes_cross_id_feature_duplicates(self):
        x = np.asarray([[1.], [2.], [3.], [4.]], dtype=np.float32)
        mask = audit.shared_reference_mask(x, np.arange(4), [x[[0]]],
                                          [np.asarray([99])], np.asarray([0, 0, 1, 1]), [0, 1])
        np.testing.assert_array_equal(mask, [False, True, True, True])

    def test_reference_fails_if_overlap_purge_loses_class(self):
        x = np.asarray([[1.], [2.]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'loses a class'):
            audit.shared_reference_mask(x, np.arange(2), [x[[0]]], [], np.asarray([0, 1]), [0, 1])

    def test_clean_calibration_feature_duplicates_cannot_enter_target_query(self):
        x = np.asarray([[1.], [2.], [3.], [4.]], dtype=np.float32)
        mask, purged = audit.existing.af_query_mask(x, np.asarray([[2.]], dtype=np.float32), [0])
        np.testing.assert_array_equal(mask, [False, False, True, True])

    def test_checkpoint_loaded_before_author_training(self):
        import torch
        from baselines.af_author_protocol import AFNetwork, AFParameters, fit
        p = AFParameters(iterations=0, tuning_epochs=0)
        original = AFNetwork(2, p)
        with torch.no_grad():
            original.extractor.blocks[0].weight.fill_(0.123)
        state = audit.cpu_state(original)
        captured = {}
        class StopObserver(Exception):
            pass
        def observe(model):
            captured.update(audit.cpu_state(model))
            raise StopObserver()
        x = np.zeros((4, 64), dtype=np.float32)
        y = np.asarray([0, 0, 1, 1])
        with self.assertRaises(StopObserver):
            fit(x, y, x, y, np.arange(2), 42, p, 'cpu',
                before_tuning=observe, initial_state={k:v.clone() for k,v in state.items()})
        for key in state:
            torch.testing.assert_close(captured[key], state[key])
            torch.testing.assert_close(original.state_dict()[key], state[key])

    def test_signed_drop_preserves_improvements(self):
        initial = dict(AC=80., PR=70., RC=60., F1=65.)
        final = dict(AC=82., PR=60., RC=65., F1=66.)
        self.assertEqual(audit.metric_drop(initial, final), dict(AC=-2., PR=10., RC=-5., F1=-1.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
