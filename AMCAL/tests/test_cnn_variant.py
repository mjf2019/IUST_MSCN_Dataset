"""User-run checks for opt-in Light IAT checkpoint compatibility."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from enhancements.cnn_variant import expected_cnn_architecture, validate_cnn_variant


class CNNVariantTests(unittest.TestCase):
    def test_existing_defaults_are_preserved(self):
        self.assertEqual(expected_cnn_architecture('IAT'), 'legacy')
        self.assertEqual(expected_cnn_architecture('PS'), 'light_ps_cnn_v1')

    def test_opt_in_iat_uses_identical_architecture_id_to_ps(self):
        self.assertEqual(expected_cnn_architecture('IAT', True), expected_cnn_architecture('PS'))
        validate_cnn_variant('IAT', 'light_ps_cnn_v1', True)

    def test_checkpoints_cannot_silently_cross_variants(self):
        for dataset, architecture, light in [('IAT', 'legacy', True),
                ('IAT', 'light_ps_cnn_v1', False), ('PS', 'legacy', False),
                ('IAT', 'unknown', False)]:
            with self.subTest(dataset=dataset, architecture=architecture, light=light):
                with self.assertRaisesRegex(ValueError, 'matching source/teacher'):
                    validate_cnn_variant(dataset, architecture, light)

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError): expected_cnn_architecture('other', True)


if __name__ == '__main__': unittest.main()
