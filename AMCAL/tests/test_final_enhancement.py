"""CPU invariants for enhanced routing, memory, sparse execution and real CLI IO."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_burst_structured_bank_protocol as legacy
import amcal_final_enhancement as runner
from enhancements.agent import EfficientQ, EnhancementAgent, DeviceReliability, ranking_loss
from enhancements.config import EnhancementConfig
from enhancements.memory import ProtectedMemory, transfer
from enhancements.models import ResidualContext, residual_bank, BankExecutor
from enhancements.stream import episode
from test_structured_bank_protocol import TinyCNN, TinyContext


class FinalEnhancementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(12)

    def config(self, profile='accuracy', **changes):
        return replace(EnhancementConfig.preset(profile), source_capacity=8,
                       historical_capacity=8, recent_capacity=16, reference_size=8,
                       memory_replay_size=2, **changes).validated()

    def fixture(self, config=None, query='uniform', policy='fixed'):
        config = config or self.config()
        cnn = TinyCNN().eval().requires_grad_(False)
        bank = [TinyContext().eval() for _ in range(3)]
        initial = deepcopy(bank[0].state_dict())
        for context in bank: context.load_state_dict(initial)
        args = SimpleNamespace(config=config, seed=42, classes=['0', '1'], context_lr=.001,
            context_alpha=.8, budget_fraction=.5, query_schedule=query, update_policy=policy,
            train_epsilon=.2, eval_epsilon=0, gain_weight=1., retention_weight=1.,
            selection_cost=.0001, balanced_reward_reference=True, bank_utility=False)
        x = torch.randn(16, 1, 8); y = np.arange(16) % 2
        anchors = torch.randn(12, 1, 8) + 5, np.arange(12) % 2
        agent = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, config)
        return cnn, bank, initial, args, (x, y, np.arange(16)), anchors, agent

    def test_single_pass_scorer_matches_original_and_gradients(self):
        old = legacy.StructuredQ(34)
        new = EfficientQ(34); new.load_state_dict(old.state_dict())
        x = torch.randn(8, 139)
        q, logits = new.forward_with_logits(x)
        self.assertTrue(torch.allclose(q, old(x), atol=1e-7))
        self.assertTrue(torch.allclose(logits, old.route_logits(x), atol=1e-7))

    def test_ranking_ignores_uninformative_rows_and_has_correct_direction(self):
        z = torch.zeros(3, 4, requires_grad=True)
        y = torch.tensor([[1., 0, 1, 0], [1., 1, 1, 1], [0., 0, 0, 0]])
        loss = ranking_loss(z, y)
        self.assertEqual(float(loss[1]), 0); self.assertEqual(float(loss[2]), 0)
        loss.sum().backward()
        self.assertLess(float(z.grad[0, 0]), 0)
        self.assertGreater(float(z.grad[0, 1]), 0)
        self.assertEqual(float(z.grad[1:].abs().sum()), 0)

    def test_td_and_pairing_cannot_train_correctness(self):
        _, _, _, a, _, _, agent = self.fixture()
        x = torch.randn(1, 139); x[:, -3] = -1; x[:, -2] = .5
        before = deepcopy(agent.policy.correctness.state_dict())
        for _ in range(40):
            agent.push(x, 4, .5, x, True)
            agent.push_pair(x, 4, .5, x, x, True)
        agent.learn_td()
        self.assertEqual(agent.td_updates, 1)
        self.assertIn('paired', agent.last_losses)
        for name, value in before.items(): self.assertTrue(torch.equal(value, agent.policy.correctness.state_dict()[name]))
        q = agent.policy(x)
        self.assertGreater(float(q[0, 4] - q[0, 1]), 0)

    def test_version_change_downweights_stale_replay(self):
        *_, agent = self.fixture()
        x = torch.zeros(1, 139)
        agent.push(x, 0, 0, x, False)
        _, fresh = agent._sample(agent.td_memory, 128)
        agent.versions[:] = 32
        _, old = agent._sample(agent.td_memory, 128)
        self.assertLess(float(old[0]), float(fresh[0]))

    def test_device_reliability_matches_existing_semantics(self):
        old = legacy.Reliability(2); new = DeviceReliability(2, 'cpu')
        for t in range(8):
            probabilities = torch.randn(1, 4, 2).softmax(2)
            old.observe_paid(probabilities, t % 2); new.observe_paid(probabilities, t % 2)
            old.updated(1, t+1, .2); new.updated(1, t+1, .2)
            self.assertTrue(torch.allclose(old.features(probabilities, 8, t, 16),
                                           new.features(probabilities, 8, t, 16), atol=1e-6))

    def test_memory_is_bounded_retains_rare_class_and_excludes_current(self):
        c = replace(self.config(), source_capacity=4, historical_capacity=4, recent_capacity=4)
        anchors = torch.arange(32, dtype=torch.float32).reshape(4, 1, 8) - 100, np.array([0, 1, 0, 1])
        memory = ProtectedMemory(anchors, 2, c, 42)
        for i in range(40): memory.add(torch.full((1, 1, 8), float(i)), torch.tensor([1 if i == 0 else 0]))
        counts = memory.counts()
        self.assertLessEqual(sum(counts[key] for key in ('source', 'historical', 'recent')), 12)
        self.assertGreater(counts['historical_per_class'][1], 0)
        current = torch.full((1, 1, 8), 39.)
        batch = memory.sample(current)
        self.assertFalse(any(torch.equal(row, current[0]) for row in batch.x))
        self.assertEqual(len({row.numpy().tobytes() for row in batch.x}), len(batch.x))
        self.assertLessEqual(len(batch.x), c.reference_size)

    def test_protected_damage_cannot_be_cancelled_by_recent_improvement(self):
        c = self.config()
        from enhancements.memory import ReferenceBatch
        batch = ReferenceBatch(torch.zeros(2, 1, 8), torch.tensor([0, 0]), np.array([0, 2]))
        utility, gain, harm = transfer(np.array([1., 1.]), np.array([2., .5]), batch, 1., 1., .0001, c)
        self.assertGreater(gain, 0); self.assertGreater(harm, 0)
        self.assertLess(utility, -.1)

    def test_compact_context_is_identity_and_residual_bounded(self):
        c = self.config('compact')
        context = ResidualContext(8, c).eval()
        x = torch.randn(3, 1, 8)
        self.assertTrue(torch.equal(context(x), x))
        with torch.no_grad(): context.adapter.output.bias.fill_(100)
        self.assertLessEqual(float((context(x) - x).abs().max()), c.residual_bound)
        self.assertLess(sum(p.numel() for p in context.parameters()), 100000)

    def test_shared_trunk_runs_once_and_only_selected_adapter_updates(self):
        c = self.config('shared')
        bank = residual_bank(8, c)
        cnn = TinyCNN().eval().requires_grad_(False)
        x = torch.randn(2, 1, 8)
        self.assertIs(bank[0].trunk, bank[1].trunk)
        self.assertFalse(any(p.requires_grad for p in bank[0].trunk.parameters()))
        executor = BankExecutor(cnn, bank)
        out = executor.outputs(x, cnn(x))
        with torch.no_grad(): expected = torch.stack([cnn(x), *[cnn(context(x)) for context in bank]], 1)
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-6))
        self.assertEqual(executor.trunk_forwards, len(x))
        before = [deepcopy(context.state_dict()) for context in bank]
        optimizer = torch.optim.Adam([p for p in bank[1].parameters() if p.requires_grad], lr=.01)
        F = torch.nn.functional
        F.cross_entropy(cnn(bank[1](x)), torch.tensor([0, 1])).backward(); optimizer.step()
        for slot in (0, 2):
            for key, value in before[slot].items(): self.assertTrue(torch.equal(value, bank[slot].state_dict()[key]))
        self.assertTrue(any(not torch.equal(value, bank[1].state_dict()[key]) for key, value in before[1].items()))

    def test_vmap_refresh_uses_current_slot_weights(self):
        cnn, bank, _, _, domain, _, _ = self.fixture()
        executor = BankExecutor(cnn, bank, 'vmap')
        x = domain[0][:2]
        with torch.no_grad(): bank[1].layer.bias.add_(.2)
        executor.refresh(1)
        out = executor.outputs(x, cnn(x))
        with torch.no_grad(): expected = torch.stack([cnn(x), *[cnn(context(x)) for context in bank]], 1)
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-6))

    def test_frozen_only_shared_path_does_not_execute_the_trunk(self):
        bank = residual_bank(8, self.config('shared'))
        cnn = TinyCNN().eval().requires_grad_(False); x = torch.randn(2, 1, 8)
        executor = BankExecutor(cnn, bank)
        out = executor.outputs(x, cnn(x), ())
        self.assertEqual(executor.context_forwards, 0); self.assertEqual(executor.trunk_forwards, 0)
        self.assertTrue(torch.isnan(out[0][:, 1:]).all())

    def test_actual_compact_vmap_agrees_with_current_modules(self):
        config = self.config('compact')
        bank = residual_bank(8, config)
        with torch.no_grad(): bank[2].adapter.output.bias.fill_(.3)
        cnn = TinyCNN().eval().requires_grad_(False); x = torch.randn(2, 1, 8)
        executor = BankExecutor(cnn, bank, 'vmap')
        out = executor.outputs(x, cnn(x))
        with torch.no_grad(): expected = torch.stack([cnn(x), *[cnn(context(x)) for context in bank]], 1)
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-5))

    def test_matched_feedback_keeps_banks_identical_and_cnn_frozen(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        snapshot = deepcopy(agent.policy.state_dict()); cnn_before = deepcopy(cnn.state_dict())
        learned, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        first = [deepcopy(context.state_dict()) for context in bank]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=snapshot)
        random, _ = episode(a, cnn, bank, initial, other, domain, anchors, mode='random')
        self.assertEqual(learned['labels'], 8); self.assertEqual(random['labels'], 8)
        self.assertEqual(learned['slot_queries'], random['slot_queries'])
        for before, context in zip(first, bank):
            for name, value in before.items(): self.assertTrue(torch.equal(value, context.state_dict()[name]))
        for name, value in cnn_before.items(): self.assertTrue(torch.equal(value, cnn.state_dict()[name]))
        self.assertTrue(all(row['labels_used'] <= 8 for row in trace))

    def test_unqueried_labels_cannot_change_learning_or_actions(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        snapshot = deepcopy(agent.policy.state_dict())
        _, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        weights = deepcopy(agent.policy.state_dict()); context_weights = [deepcopy(c.state_dict()) for c in bank]
        queried = {row['source_row'] for row in trace if row['queried']}
        y = domain[1].copy()
        for i in range(len(y)):
            if i not in queried: y[i] = 1-y[i]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=snapshot)
        _, second = episode(a, cnn, bank, initial, other, (domain[0], y, domain[2]), anchors)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in second])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in second])
        for name, value in weights.items(): self.assertTrue(torch.equal(value, other.policy.state_dict()[name]))
        for before, context in zip(context_weights, bank):
            for name, value in before.items(): self.assertTrue(torch.equal(value, context.state_dict()[name]))

    def test_zero_budget_never_acquires_or_updates(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.budget_fraction = 0
        m, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        self.assertEqual(m['labels'], 0); self.assertEqual(m['route_updates'], 0); self.assertEqual(m['td_updates'], 0)
        self.assertTrue(all(not row['queried'] for row in trace))

    def test_sparse_execution_never_invents_routes_targets_or_complete_oracle(self):
        c = self.config('cascade', cascade_audit_fraction=0., cascade_max_gap=100)
        cnn, bank, initial, a, domain, anchors, agent = self.fixture(c, query='policy', policy='learned')
        with torch.no_grad():
            for p in agent.pre_router.parameters(): p.zero_()
            agent.pre_router.network[-1].bias.copy_(torch.tensor([10., -10., -10., -10.]))
        m, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        self.assertEqual(m['full_bank_fraction'], 0)
        self.assertIsNone(m['oracle_all']); self.assertIsNone(m['oracle_on_full_rows'])
        self.assertEqual(m['stream_context_forwards'], 0)
        self.assertEqual(m['route_updates'], 0)
        self.assertTrue(all(row['predictions'][1:] == [None, None, None] for row in trace))

    def test_routed_utility_and_observers_do_not_use_unpaid_labels(self):
        c = self.config(routed_utility=True)
        cnn, bank, initial, a, domain, anchors, agent = self.fixture(c)
        before = deepcopy(agent.policy.state_dict())
        _, trace = episode(a, cnn, bank, initial, agent, domain, anchors, same_level=True, reference=anchors)
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, c, policy_state=before)
        _, second = episode(a, cnn, bank, initial, other, domain, anchors)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in second])
        self.assertEqual([r['reward'] for r in trace], [r['reward'] for r in second])

    def test_configuration_invalid_values_and_fingerprint(self):
        c = self.config()
        self.assertNotEqual(c.fingerprint, c.without(['ranking']).fingerprint)
        with self.assertRaises(ValueError): replace(c, source_capacity=257).validated()
        with self.assertRaises(ValueError): replace(c, route_lr_online=float('nan')).validated()

    def test_real_cli_training_evaluation_serialization_and_persistent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; models = source/'Models'; models.mkdir(parents=True)
            burst = root/'burst'; burst.mkdir()
            rng = np.random.default_rng(42); features = [f'f{i}' for i in range(8)]
            raw = rng.normal(size=(80, 8)).astype(np.float32); labels = np.arange(80) % 2
            for level in (0, 5, 7):
                path = burst/f'AdvBurstTrainedBurst_size_{level}_Step_2000.csv'
                frame = pd.DataFrame(raw + level*.02, columns=features); frame['label'] = labels
                frame.to_csv(path, index=False)
                pd.DataFrame({'source_row': np.arange(80)}).to_csv(path.with_name(path.stem+'_rows.csv'), index=False)
            artifact = dict(feature_names=features, label_name='label', source_rows=80, source_labels=labels,
                train_ids=np.arange(40), val_ids=np.arange(40, 60), test_ids=np.arange(60, 80),
                scaler=StandardScaler().fit(raw[:40]), label_encoder=LabelEncoder().fit(labels),
                source_sha256=hashlib.sha256((burst/'AdvBurstTrainedBurst_size_0_Step_2000.csv').read_bytes()).hexdigest())
            joblib.dump(artifact, models/'protocol_preprocessing.joblib')
            cnn = legacy.core.CNN1D(8, 2)
            context = legacy.core.ContextAwareNetwork(8)
            torch.save({'model_state_dict': cnn.state_dict()}, models/'initial_cnn_model.pth')
            torch.save({'context_model_state_dict': context.state_dict()}, models/'adversarial_best_context.pth')
            output = root/'enhanced'
            common = ['--output', str(output), '--burst-dir', str(burst), '--device', 'cpu']
            with redirect_stdout(io.StringIO()):
                runner.main(['train', *common, '--source-run', str(source), '--profile', 'cascade',
                    '--epochs', '1', '--context-epochs', '1', '--warmup-episodes', '1', '--levels', '0'])
                runner.main(['evaluate', *common, '--levels', '5', '7', '--modes', 'learned', '--continual'])
            results = next((output/'Results').glob('evaluation_*'))
            m = json.loads((results/'learned_level5_metrics.json').read_text())
            self.assertTrue(m['classifier_unchanged'])
            self.assertLessEqual(m['labels'], m['label_cap'])
            self.assertTrue((results/'sequential_summary.csv').is_file())
            config = json.loads((results/'execution.json').read_text())
            audit = set(config['audit_source_ids'])
            self.assertFalse(audit & set(m['source_ids']))
            self.assertEqual(m['config_fingerprint'], EnhancementConfig(**config['config']).fingerprint)
            # Preexisting results and training runs are protected from accidental overwrite.
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                runner.main(['evaluate', *common, '--levels', '5', '7', '--modes', 'learned', '--continual'])


if __name__ == '__main__': unittest.main()
