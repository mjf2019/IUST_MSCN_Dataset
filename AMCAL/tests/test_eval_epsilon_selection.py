"""Standard-library selection contracts; no model imports or execution."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amcal_select_eval_epsilon import select_candidate, EPSILONS


def cases():
    return {dataset: [
        dict(training_seed=seed, level=level, epsilon=epsilon, AC=70., F1=65.,
             forgetting=0., labels_percent=20.)
        for seed in [42, 48, 54, 62, 72] for epsilon in EPSILONS for level in [0, 1, 3]
    ] for dataset in ['IAT', 'PS']}


class ValidationEpsilonSelectionTests(unittest.TestCase):
    def test_highest_validation_score_wins(self):
        groups = cases()
        for rows in groups.values():
            for row in rows:
                if row['epsilon'] == .01:
                    row['AC'] = 72.
        selected, _ = select_candidate(groups)
        self.assertEqual(selected['epsilon'], .01)

    def test_positive_forgetting_is_penalized_per_domain(self):
        groups = cases()
        for rows in groups.values():
            for row in rows:
                if row['epsilon'] == .05:
                    row['AC'] = 72.
                    row['forgetting'] = 9. if row['level'] == 0 else -9.
        selected, reports = select_candidate(groups)
        self.assertEqual(selected['epsilon'], 0.)
        self.assertAlmostEqual(next(r for r in reports if r['epsilon'] == .05)['score'], 69.)

    def test_ties_use_labels_then_smaller_epsilon(self):
        groups = cases()
        for rows in groups.values():
            for row in rows:
                if row['epsilon'] in [.01, .05]:
                    row['labels_percent'] = 15.
        selected, _ = select_candidate(groups)
        self.assertEqual(selected['epsilon'], .01)

    def test_missing_validation_level_is_rejected(self):
        groups = cases()
        groups['PS'].pop()
        with self.assertRaises(ValueError):
            select_candidate(groups)

    def test_test_levels_cannot_replace_validation_levels(self):
        groups = cases()
        groups['IAT'][0]['level'] = 5
        with self.assertRaises(ValueError):
            select_candidate(groups)


if __name__ == '__main__':
    unittest.main()
