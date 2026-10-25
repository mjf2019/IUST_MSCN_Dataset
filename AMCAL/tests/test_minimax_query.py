"""CPU unit checks only: no datasets, downloads or model training runs."""
import importlib.util
from pathlib import Path
import unittest
from copy import deepcopy

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location('minimax_query',
        Path(__file__).resolve().parents[1]/'enhancements'/'minimax_query.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@unittest.skipIf(torch is None, 'Requires the existing experiment environment with PyTorch')
class MinimaxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_actor_ascends_signed_score(self):
        for sign in [1., -1.]:
            head = module.QueryMaximizer(3)
            optimizer = torch.optim.SGD(head.parameters(), lr=.1)
            states = torch.zeros(4, 3)
            before = head(states)[0].sigmoid().mean().item()
            module.maximize_step(head, optimizer, states, torch.full((4,), sign))
            after = head(states)[0].sigmoid().mean().item()
            self.assertGreater(sign*(after-before), 0)
            self.assertFalse({id(p) for p in head.actor.parameters()} &
                             {id(p) for p in head.value.parameters()})

    def test_counterfactual_preserves_live_weights_adam_and_rng(self):
        context = nn.Linear(2, 2, bias=False).eval()
        cnn = nn.Linear(2, 2, bias=False).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(context.parameters(), lr=.01)
        current = (torch.tensor([[1., .2]]), torch.tensor([1]))
        paid = [(torch.tensor([[.3, -.8]]), torch.tensor([0]))]
        replay = (torch.tensor([[-1., .7]]), torch.tensor([0]))
        future = (torch.tensor([[.8, -.2], [-.4, .9]]), torch.tensor([1, 0]))
        reference = (torch.tensor([[.1, -1.], [-.3, -.7]]), torch.tensor([0, 1]))
        # Populate Adam moments before cloning, rather than testing an empty state.
        module.context_min_step(context, optimizer, cnn, current, paid, replay, .1)
        weights = deepcopy(context.state_dict())
        adam = deepcopy(optimizer.state_dict())
        rng = torch.random.get_rng_state().clone()
        candidate, candidate_optimizer, target = module.counterfactual_step(
            context, optimizer, cnn, current, paid, replay, future, reference,
            .1, 2., 3., .05)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key, value in weights.items(): self.assertTrue(torch.equal(value, context.state_dict()[key]))
        after_adam = optimizer.state_dict()
        for key, state in adam['state'].items():
            for field, value in state.items():
                self.assertTrue(torch.equal(value, after_adam['state'][key][field]))
        self.assertIsNot(candidate, context)
        self.assertIsNot(candidate_optimizer, optimizer)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(context.parameters(), candidate.parameters())))
        copied_parameters = {id(p) for group in candidate_optimizer.param_groups for p in group['params']}
        self.assertEqual(copied_parameters, {id(p) for p in candidate.parameters()})
        self.assertAlmostEqual(target['score'], target['gap']+2*target['gain']-3*target['harm']-.05)
        expected_gain = (module.row_losses(cnn, context, future) -
                         module.row_losses(cnn, candidate, future)).mean().item()
        self.assertAlmostEqual(target['gain'], expected_gain)
        self.assertGreaterEqual(target['harm'], 0)
        self.assertTrue(all(p.grad is None for p in cnn.parameters()))

    def test_nonfinite_targets_are_rejected(self):
        head = module.QueryMaximizer(3)
        optimizer = torch.optim.Adam(head.parameters())
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            module.maximize_step(head, optimizer, torch.zeros(1, 3), torch.tensor([float('nan')]))

    def test_measured_score_is_a_detached_target(self):
        head = module.QueryMaximizer(3)
        optimizer = torch.optim.Adam(head.parameters())
        measured_score = torch.tensor([.2], requires_grad=True)
        module.maximize_step(head, optimizer, torch.zeros(1, 3), measured_score)
        self.assertIsNone(measured_score.grad)


if __name__ == '__main__': unittest.main()
