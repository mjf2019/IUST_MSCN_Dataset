"""Small numerical contracts, no real datasets or training runs."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires the existing PyTorch experiment environment')
class StandardizedActorTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT))

    def tearDown(self):
        sys.path.pop(0)

    def test_natural_moments_recover_reservoir_prevalence(self):
        from enhancements.standardized_actor import fit_train_spec
        samples = [dict(state=torch.tensor([value, 7.]), score=sign)
            for value, sign in [(10., 1.), (0., -1.)] for _ in range(8)]
        spec = fit_train_spec(samples, {'positive':10, 'negative':90})
        self.assertAlmostEqual(spec['mean'][0], 1.)
        self.assertAlmostEqual(spec['scale'][0], 3.)
        self.assertAlmostEqual(spec['mean'][1], 7.)
        self.assertEqual(spec['scale'][1], 1.)
        self.assertEqual(spec['near_constant_dimensions'], 1)
        self.assertEqual(spec['positive_fraction'], .1)

    def test_zero_logit_penalty_matches_original_update(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer, regularized_actor_step
        from enhancements.minimax_query import maximize_step
        torch.manual_seed(42)
        model = StandardizedQueryMaximizer(2, dict(mean=[1., 2.], scale=[2., 3.]))
        twin = deepcopy(model)
        opt1 = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
        opt2 = torch.optim.Adam(twin.parameters(), lr=.001, weight_decay=1e-4)
        states = torch.tensor([[2., 3.], [-2., -4.]], requires_grad=True)
        targets = torch.tensor([.2, -.8], requires_grad=True)
        for _ in range(3):
            maximize_step(model, opt1, states, targets)
            regularized_actor_step(twin, opt2, states, targets, 0.)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v, twin.state_dict()[k])
        self.assertIsNone(states.grad); self.assertIsNone(targets.grad)

    def test_logit_penalty_restores_saturated_outputs_without_changing_value_update(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer, regularized_actor_step
        from enhancements.minimax_query import maximize_step
        for logit in (-20., 20.):
            model = StandardizedQueryMaximizer(2, dict(mean=[0., 0.], scale=[1., 1.]))
            with torch.no_grad():
                for p in model.actor.parameters(): p.zero_()
                model.actor[-1].bias.fill_(logit)
            twin = deepcopy(model)
            buffers = {k:v.clone() for k,v in model.named_buffers()}
            states = torch.ones(1, 2, requires_grad=True)
            targets = torch.tensor([0.], requires_grad=True)
            opt1 = torch.optim.SGD(model.parameters(), lr=.1)
            opt2 = torch.optim.SGD(twin.parameters(), lr=.1)
            maximize_step(twin, opt2, states, targets)
            result = regularized_actor_step(model, opt1, states, targets, .1)
            self.assertLess(abs(float(model.actor[-1].bias)), abs(logit))
            self.assertEqual(float(twin.actor[-1].bias), logit)
            self.assertAlmostEqual(result['actor_score_loss'], 0.)
            self.assertAlmostEqual(result['actor_logit_penalty_loss'], 20.)
            for k,v in model.value.state_dict().items(): torch.testing.assert_close(v, twin.value.state_dict()[k])
            for k,v in model.named_buffers(): torch.testing.assert_close(v, buffers[k])
            self.assertIsNone(states.grad); self.assertIsNone(targets.grad)

    def test_invalid_logit_penalty_is_rejected(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer, regularized_actor_step
        head = StandardizedQueryMaximizer(2, dict(mean=[0., 0.], scale=[1., 1.]))
        optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        for penalty in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                regularized_actor_step(head, optimizer, torch.ones(1, 2), torch.ones(1), penalty)

    def test_forward_changes_actor_only_and_checkpoint_roundtrip(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer
        spec = dict(mean=[2., 4.], scale=[2., 1.])
        model = StandardizedQueryMaximizer(2, spec)
        with torch.no_grad(): model.actor[-1].weight.fill_(.1)
        states = torch.tensor([[4., 7.]])
        actor, value = model(states)
        torch.testing.assert_close(actor, model.actor(torch.tensor([[1., 3.]])).squeeze(-1))
        torch.testing.assert_close(value, model.value(states).squeeze(-1))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'checkpoint.pt'
            torch.save(model.state_dict(), path)
            restored = StandardizedQueryMaximizer(2, dict(mean=[0., 0.], scale=[1., 1.]))
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(states)[0], actor)

    def test_balanced_step_at_equal_prevalence_matches_original(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer, balanced_actor_step
        from enhancements.minimax_query import maximize_step
        torch.manual_seed(42)
        first = StandardizedQueryMaximizer(2, dict(mean=[1., 2.], scale=[2., 3.]))
        second = deepcopy(first)
        opt1 = torch.optim.Adam(first.parameters(), lr=.001, weight_decay=1e-4)
        opt2 = torch.optim.Adam(second.parameters(), lr=.001, weight_decay=1e-4)
        states = torch.tensor([[2., 3.], [-2., -4.]], requires_grad=True)
        targets = torch.tensor([.2, -.8], requires_grad=True)
        initial_buffers = {k:v.clone() for k,v in first.named_buffers()}
        for _ in range(3):
            maximize_step(first, opt1, states, targets)
            balanced_actor_step(second, opt2, states, targets, .5)
        for k, v in first.state_dict().items(): torch.testing.assert_close(v, second.state_dict()[k])
        for k, v in first.named_buffers(): torch.testing.assert_close(v, initial_buffers[k])
        self.assertIsNone(states.grad); self.assertIsNone(targets.grad)

    def test_invalid_normalizer_and_insufficient_cache_are_rejected(self):
        from enhancements.standardized_actor import StandardizedQueryMaximizer, fit_train_spec
        with self.assertRaises(ValueError):
            StandardizedQueryMaximizer(2, dict(mean=[0., 0.], scale=[1., 0.]))
        with self.assertRaises(ValueError): fit_train_spec([], {'positive':1, 'negative':1})


if __name__ == '__main__': unittest.main()

