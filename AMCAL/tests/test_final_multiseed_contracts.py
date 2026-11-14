"""Coordinator contracts only: no Torch, model imports, training or inference."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'multiseed_coordinator', Path(__file__).resolve().parents[1]/'amcal_final_multiseed.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MultiseedContracts(unittest.TestCase):
    def test_duplicate_training_seeds_are_not_independent_runs(self):
        with self.assertRaises(ValueError):
            m.validate_seeds([42, 42])
        with self.assertRaises(ValueError):
            m.validate_seeds([42])
        self.assertEqual(m.validate_seeds([42, 48]), [42, 48])

    def test_training_stage_locations_do_not_share_adaptive_checkpoints(self):
        first = m.seed_paths(Path('experiment'), 42)
        second = m.seed_paths(Path('experiment'), 48)
        self.assertFalse(set(first.values()) & set(second.values()))
        self.assertEqual(set(first), set(m.STAGES) | {'evaluation'})

    def test_protocol_drift_is_rejected_instead_of_reusing_results(self):
        with self.assertRaises(ValueError):
            m.check_plan({'cnn': 'original', 'seed': 42}, {'cnn': 'light', 'seed': 42})

    def test_sd_is_sample_sd(self):
        mean, sd = m.moments([1., 2., 3.])
        self.assertEqual(mean, 2.)
        self.assertAlmostEqual(sd, 1.)
        with self.assertRaises(ValueError):
            m.moments([1.])
        with self.assertRaises(ValueError):
            m.moments([1., float('nan')])

    def test_overall_sd_is_across_training_seed_means_not_pooled_levels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'dataset': 'IAT', 'training_seeds': [42, 48]}
            m.write(root/'plan.json', plan)
            for seed, offset in [(42, 10.), (48, 20.)]:
                rows = []
                for level in m.LEVELS:
                    row = dict(dataset='IAT', training_seed=seed, evaluation_seed=42,
                               level=level, beta=1.)
                    row.update({k: offset+level for k in m.METRICS})
                    rows.append(row)
                dest = m.seed_paths(root, seed)['evaluation']
                m.write(dest/'summary_by_level.json', rows)
                m.write(dest/'completed.json', dict(training_seed=seed,
                    plan_sha256=m.digest(root/'plan.json'),
                    summary_sha256=m.digest(dest/'summary_by_level.json')))
            m.aggregate(root, plan)
            result = m.read(root/'Results'/'completed.json')['overall']
            self.assertAlmostEqual(result['AC_mean'], 15.+sum(m.LEVELS)/7)
            self.assertAlmostEqual(result['AC_sd'], math.sqrt(50.))
            self.assertEqual(result['seeds'], 2)

    def test_missing_training_seed_cannot_be_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'dataset': 'PS', 'training_seeds': [42, 48]}
            m.write(root/'plan.json', plan)
            with self.assertRaises(FileNotFoundError):
                m.aggregate(root, plan)
            self.assertFalse((root/'Results'/'completed.json').exists())


if __name__ == '__main__':
    unittest.main()
