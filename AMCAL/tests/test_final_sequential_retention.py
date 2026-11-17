"""Standard-library retention arithmetic tests; no models or experiments."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_sequential_metrics import LEVELS, METRICS, summarize, support_scale


def fixture():
    matrix = [dict(stage=t,domain=d,**dict.fromkeys(METRICS,80.))
              for t in range(8) for d in [0,*LEVELS]]
    online = [dict(level=l,samples=100,labels=20,labels_percent=20.,
                   **dict.fromkeys(METRICS,75.)) for l in LEVELS]
    return matrix,online


class FinalSequentialRetentionTests(unittest.TestCase):
    def test_fixed_predictor_has_zero_forgetting_and_bwt(self):
        matrix,online = fixture()
        result = summarize(matrix,online)
        self.assertEqual(result['AC']['domain_forgetting'],0.)
        self.assertEqual(result['AC']['BWT'],0.)
        self.assertEqual(result['AC']['original_forgetting'],0.)

    def test_known_degradation_has_expected_signed_statistics(self):
        matrix,online = fixture()
        for row in matrix:
            if row['stage']==7 and row['domain'] in LEVELS[:-1]:
                for m in METRICS: row[m] = 70.
            if row['stage']==7 and row['domain']==0:
                for m in METRICS: row[m] = 75.
        result = summarize(matrix,online)
        self.assertEqual(result['AC']['domain_forgetting'],10.)
        self.assertEqual(result['AC']['BWT'],-10.)
        self.assertEqual(result['AC']['original_forgetting'],5.)

    def test_final_improvement_has_positive_bwt_and_nonnegative_forgetting(self):
        matrix,online = fixture()
        for row in matrix:
            if row['stage']==7 and row['domain'] in LEVELS[:-1]:
                row['AC'] = 85.
        result = summarize(matrix,online)
        self.assertEqual(result['AC']['BWT'],5.)
        self.assertEqual(result['AC']['domain_forgetting'],0.)

    def test_last_domain_does_not_enter_forgetting_or_bwt(self):
        matrix,online = fixture()
        for row in matrix:
            if row['stage']==7 and row['domain']==20:
                row['AC'] = 0.
        result = summarize(matrix,online)
        self.assertEqual(result['AC']['BWT'],0.)
        self.assertEqual(result['AC']['domain_forgetting'],0.)

    def test_missing_and_duplicate_matrix_cells_are_rejected(self):
        matrix,online = fixture()
        with self.assertRaises(ValueError): summarize(matrix[:-1],online)
        with self.assertRaises(ValueError): summarize(matrix+[matrix[0]],online)

    def test_support_scaling_separates_domain_budget_from_cumulative_feedback(self):
        self.assertEqual(support_scale(0,0),1.)
        self.assertEqual(support_scale(0,20),1/20)
        self.assertEqual(support_scale(5,25),5/25)
        with self.assertRaises(ValueError): support_scale(6,5)


if __name__=='__main__':
    unittest.main()
