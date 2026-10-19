"""Mapping/protocol tests plus optional PyTorch attack checks on tiny models."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import adversarial_domains as domains
import amcal_adversarial_protocol as protocol

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch


class DomainTests(unittest.TestCase):
    def test_level_two_is_second_positive_intensity(self):
        self.assertEqual(domains.strength('FGSM', 2), .05)
        self.assertEqual(domains.strength('PGD', 2), .05)
        self.assertEqual(domains.strength('CW', 2), .5)
        self.assertEqual(domains.strength('DeepFool', 2), .5)
        self.assertEqual(domains.strength('CW', 0), 0.)
        for attack, level in [('FGSM', 6), ('CW', -1), ('fake', 2)]:
            with self.assertRaises(ValueError):domains.strength(attack, level)

    def test_domain_export_preserves_explicit_ids_and_original_labels(self):
        source = pd.DataFrame({'x': [10., 20., 30.], 'z': [1., 2., 3.], 'label': ['A', 'B', 'A']})
        artifact = dict(feature_names=['x', 'z'], label_name='label',
                        scaler=SimpleNamespace(inverse_transform=lambda x: x * 2))
        with tempfile.TemporaryDirectory() as directory:
            record = domains.export_domain(directory, 2, source, [2, 0], [[1., 2.], [3., 4.]], artifact)
            path = Path(directory) / domains.domain_name(2)
            frame = pd.read_csv(path)
            mapping = pd.read_csv(path.with_name(path.stem + '_rows.csv'))
            self.assertEqual(mapping.source_row.tolist(), [2, 0])
            self.assertEqual(frame.label.tolist(), ['A', 'A'])
            np.testing.assert_equal(frame[['x', 'z']].to_numpy(), [[2., 4.], [6., 8.]])
            self.assertEqual(record['csv_sha256'], domains.digest(path))
            self.assertEqual(record['rows'], 2)

    def test_mapping_rejects_duplicate_fractional_and_out_of_bounds_ids(self):
        for ids in ([0, 0], [.5], [-1], [3], []):
            with self.assertRaises(ValueError):domains.validate_ids(ids, 3)

    def test_existing_cache_is_validated_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / domains.domain_name(2)
            path.write_text('changed')
            mapping = path.with_name(path.stem + '_rows.csv')
            mapping.write_text('source_row\n0\n')
            manifest = {'domains': {'2': dict(csv_sha256='old', row_map_sha256=domains.digest(mapping), rows=1)}}
            with self.assertRaises(ValueError):
                protocol.make_attack(SimpleNamespace(), None, None, None, None, None,
                                     Path(directory), 2, [0], manifest)

    def test_all_attack_dispatch_preserves_seed_and_has_independent_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'new'
            a = SimpleNamespace(command='train', output=root, data_dir=Path(directory),
                                source_file='source.csv', clean_run=Path(directory), seed=42, device='cpu',
                                levels=list(range(6)), context_lr=.0005, budget_fraction=.2,
                                selection_cost=.0001, gain_weight=1., retention_weight=1., gamma=.95,
                                eval_epsilon=.05, stage1_epochs=12, cnn_epochs=30, source_epochs=30,
                                bank_epochs=12, pgd_steps=20, cw_steps=100, cw_search=3, deepfool_steps=25)
            with patch.object(protocol.subprocess, 'run') as run:
                protocol.all_attacks(a)
            self.assertEqual(run.call_count, 4)
            for call, attack in zip(run.call_args_list, domains.LEVELS):
                argv = call.args[0]
                self.assertEqual(argv[argv.index('--attack')+1], attack)
                self.assertEqual(argv[argv.index('--output')+1], str(root / attack))
                self.assertEqual(argv[argv.index('--eval-epsilon')+1], '0.05')
                self.assertEqual(argv[argv.index('--seed')+1], '42')
                self.assertTrue(call.kwargs['check'])

    def test_csv_aggregation_distinguishes_seen_from_unseen(self):
        rows = []
        for level in (0, 2, 3):
            for mode in ('learned', 'random'):
                rows.append(dict(Attack='FGSM', Level=str(level), Strength=str(domains.strength('FGSM', level)),
                                 Domain='Unseen' if level == 3 else 'Seen', Mode=mode,
                                 Frozen='50', Online='60' if mode == 'learned' else '55',
                                 Labels='10', RefDrop='1'))
        with redirect_stdout(io.StringIO()) as output:protocol.display(rows)
        self.assertIn('Unseen', output.getvalue())
        self.assertIn('Mean', output.getvalue())
        self.assertIn('60.00', output.getvalue())

    def test_training_levels_never_default_to_a_test_intensity(self):
        # The wrapper calls the existing engines, rather than reimplementing
        # policy decisions; inspect command construction without importing Torch.
        import ast
        tree = ast.parse(Path(protocol.__file__).read_text())
        training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        text = ast.get_source_segment(Path(protocol.__file__).read_text(), training)
        self.assertIn("artifact['train_ids'], artifact['val_ids']", text)
        self.assertNotIn("artifact['test_ids']", text.split('make_attack(a,')[1].split('original_train =')[0])
        self.assertIn("'--multishift-levels', 0, 2", text)
        self.assertIn("'--balanced-reward-reference'", text)
        self.assertNotIn("'--decoupled-query'", text)
        self.assertNotIn("'--budget-aware-policy'", text)


@unittest.skipUnless(HAS_TORCH, 'PyTorch not installed in this environment')
class AttackTensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2, bias=False)).eval()
        with torch.no_grad():self.model[1].weight.copy_(torch.tensor([[1., 1., 1., 1.], [-1., -1., -1., -1.]]))
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.inputs = torch.tensor([[[.2, .2, .2, .2]], [[-.2, -.2, -.2, -.2]]])
        self.labels = torch.tensor([0, 1])

    def test_fgsm_and_pgd_are_linf_bounded_and_increase_loss(self):
        for attack in ('FGSM', 'PGD'):
            before = self.model[1].weight.detach().clone()
            output = domains.perturb(torch, self.model, self.inputs, self.labels, attack, .5)
            self.assertLessEqual(float((output-self.inputs).abs().max()), .50001)
            self.assertGreater(float(torch.nn.functional.cross_entropy(self.model(output), self.labels)),
                               float(torch.nn.functional.cross_entropy(self.model(self.inputs), self.labels)))
            self.assertTrue(torch.equal(self.model[1].weight, before))

    def test_deepfool_crosses_a_linear_boundary_within_l2_radius(self):
        output = domains.perturb(torch, self.model, self.inputs, self.labels, 'DeepFool', 1.)
        self.assertTrue((self.model(output).argmax(1) != self.labels).all())
        self.assertTrue(((output-self.inputs).flatten(1).norm(dim=1) <= 1.00001).all())

    def test_projected_cw_obeys_l2_budget_and_crosses_a_linear_boundary(self):
        output = domains.perturb(torch, self.model, self.inputs, self.labels, 'CW', 1., cw_steps=120, cw_search=4)
        self.assertTrue((self.model(output).argmax(1) != self.labels).all())
        self.assertTrue(((output-self.inputs).flatten(1).norm(dim=1) <= 1.00001).all())

    def test_zero_radius_preserves_inputs(self):
        for attack in domains.LEVELS:
            output = domains.perturb(torch, self.model, self.inputs, self.labels, attack, 0.)
            self.assertTrue(torch.equal(output, self.inputs))


if __name__ == '__main__':unittest.main()
