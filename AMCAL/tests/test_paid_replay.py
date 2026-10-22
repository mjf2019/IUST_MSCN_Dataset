"""CPU checks: python -m unittest discover -s AMCAL/tests -p test_paid_replay.py -v"""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required')
class PaidReplayTests(unittest.TestCase):
    def test_steps_and_batch_match_explicit_updates(self):
        # Load isolated helper with the real production Context objective/step.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from copy import deepcopy
        from enhancements.paid_replay import paid_context_step
        from enhancements.counterfactual import context_step
        from enhancements.config import EnhancementConfig
        cnn = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        initial = torch.nn.Linear(2, 2).eval()
        for mode in ('sequential', 'batch', 'batch_all', 'batch5', 'batch8'):
            size = {'batch5': 5, 'batch8': 8}.get(mode, 3)
            rows = [(torch.tensor([[float(i), .4]]), torch.tensor([i % 2])) for i in range(size)]
            context = deepcopy(initial); reference = deepcopy(initial)
            optimizer = torch.optim.Adam(context.parameters(), lr=.0005, weight_decay=1e-4)
            expected = torch.optim.Adam(reference.parameters(), lr=.0005, weight_decay=1e-4)
            a = types.SimpleNamespace(paid_replay_mode=mode, context_alpha=.8, config=EnhancementConfig())
            frozen = cnn(rows[0][0]).detach()
            before = deepcopy(cnn.state_dict())
            paid_context_step(a, cnn, context, optimizer, *rows[0], frozen, None, rows[1:])
            if mode == 'sequential':
                for x, y in rows: context_step(a, cnn, reference, expected, x, y, cnn(x).detach(), None)
            else:
                import amcal_burst_structured_bank_protocol as legacy
                x = torch.cat([r[0] for r in rows]); y = torch.cat([r[1] for r in rows])
                z = reference(x)
                loss = legacy.context_objective(torch.nn.functional.cross_entropy(cnn(z), y),
                    torch.nn.functional.cross_entropy(cnn(x), y).detach(), .8)
                loss = loss + a.config.residual_weight * (z-x).square().mean()
                expected.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1); expected.step()
            for k, v in context.state_dict().items(): torch.testing.assert_close(v, reference.state_dict()[k])
            for k, v in cnn.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
            self.assertEqual(int(next(iter(optimizer.state.values()))['step']), 3 if mode == 'sequential' else 1)

    def test_paid_sampler_excludes_source_current_and_duplicates(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import numpy as np
        from enhancements.memory import ProtectedMemory
        from enhancements.config import EnhancementConfig
        config = EnhancementConfig(source_capacity=1, recent_capacity=2, historical_capacity=4)
        source = torch.tensor([[[-99.]]])
        memory = ProtectedMemory((source, np.array([0])), 2, config, 42)
        memory.paid_replay_rng = np.random.default_rng(55)
        current = torch.tensor([[[10.]]])
        self.assertEqual(memory.paid_replay(current), [])
        for value in (1., 2., 3., 10.): memory.add(torch.tensor([[[value]]]), torch.tensor([0]))
        memory_rng = deepcopy_state(memory.rng.bit_generator.state)
        replay_rng = deepcopy_state(memory.replay_rng.bit_generator.state)
        all_sample = memory.paid_replay(current, count=None)
        all_values = [float(x.item()) for x, y in all_sample]
        self.assertEqual(set(all_values), {1., 2., 3.})
        self.assertEqual(len(all_values), 3)
        for count in (4, 7):
            short = memory.paid_replay(current, count=count)
            self.assertEqual({float(x.item()) for x, y in short}, {1., 2., 3.})
            self.assertEqual(len(short), 3)
        sample = memory.paid_replay(current)
        values = [float(x.item()) for x, y in sample]
        self.assertEqual(len(values), 2); self.assertEqual(len(set(values)), 2)
        self.assertTrue(set(values) <= {1., 2., 3.})
        self.assertEqual(memory_rng, memory.rng.bit_generator.state)
        self.assertEqual(replay_rng, memory.replay_rng.bit_generator.state)


def deepcopy_state(state):
    from copy import deepcopy
    return deepcopy(state)


if __name__ == '__main__': unittest.main()
