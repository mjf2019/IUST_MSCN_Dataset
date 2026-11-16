"""Standard-library endpoint arithmetic and coverage checks; no model execution."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from burst_table_completion_utils import table_row, aggregate, METRICS, LEVELS


def scores(value):
    return dict.fromkeys(METRICS,float(value))


class BurstTableCompletionTests(unittest.TestCase):
    def row(self, seed=42, level=5):
        return table_row('IAT','test',seed,level,scores(70),scores(50),scores(60),
                         scores(80),scores(77),100,20)

    def test_gain_and_forgetting_have_opposite_directions(self):
        row = self.row()
        for metric in METRICS:
            self.assertEqual(row['target_gain_'+metric],10.)
            self.assertEqual(row['target_forgetting_'+metric],-10.)
            self.assertEqual(row['original_forgetting_'+metric],3.)

    def test_feedback_cap_is_strict(self):
        with self.assertRaises(ValueError):
            table_row('IAT','test',42,5,scores(70),scores(50),scores(60),
                      scores(80),scores(77),100,21)

    def test_missing_level_and_duplicate_case_are_rejected(self):
        rows = [self.row(level=l) for l in LEVELS]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                aggregate(Path(folder),rows[:-1],[42],'test')
            with self.assertRaises(ValueError):
                aggregate(Path(folder),rows+[rows[0]],[42],'test')

    def test_sd_is_across_seed_means(self):
        rows = [self.row(seed=seed,level=level) for seed in [42,48] for level in LEVELS]
        for row in rows:
            row['AC'] = 70. if row['seed']==42 else 80.
        with tempfile.TemporaryDirectory() as folder:
            overall = aggregate(Path(folder),rows,[42,48],'test')
        self.assertEqual(overall['AC_mean'],75.)
        self.assertAlmostEqual(overall['AC_sd'],(50.)**.5)

    def test_nonfinite_observer_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            table_row('IAT','test',42,5,scores(70),scores(float('nan')),scores(60),
                      scores(80),scores(77),100,20)


if __name__=='__main__':
    unittest.main()
