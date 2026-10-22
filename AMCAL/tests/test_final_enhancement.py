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
from unittest.mock import patch
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
from enhancements.agent import EfficientQ, EnhancementAgent, DeviceReliability, ranking_loss, routing_objective, load_policy_state
from enhancements.config import EnhancementConfig
from enhancements.memory import ProtectedMemory, transfer
from enhancements.models import ResidualContext, residual_bank, BankExecutor, PreRouter
from enhancements.stream import episode
from enhancements.sequence import split_validation, fit_sequence, validate_sequence
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

    def test_joint_router_is_neutral_at_start_and_strictly_loads_old_and_new_policies(self):
        old = EfficientQ(34)
        joint = EfficientQ(34, joint_router=True)
        load_policy_state(joint, old.state_dict())
        x = torch.randn(5, 139)
        self.assertTrue(torch.equal(old.route_logits(x), joint.route_logits(x)))
        self.assertTrue(torch.equal(old(x), joint(x)))
        saved = deepcopy(joint.state_dict())
        restored = EfficientQ(34, joint_router=True)
        load_policy_state(restored, saved)
        self.assertTrue(torch.equal(joint(x), restored(x)))
        broken = dict(saved); del broken[next(k for k in broken if k.startswith('comparison.'))]
        with self.assertRaises(ValueError): load_policy_state(restored, broken)
        broken = dict(old.state_dict()); del broken['correctness.0.weight']
        with self.assertRaises(ValueError): load_policy_state(restored, broken)
        with self.assertRaises(ValueError): load_policy_state(old, saved)

    def test_joint_router_compares_other_routes_and_is_permutation_equivariant(self):
        model = EfficientQ(34, joint_router=True).eval()
        with torch.no_grad(): model.comparison.output[-1].weight.normal_(0, .1)
        x = torch.randn(3, 139); altered = x.clone()
        altered[:, 34:68] += torch.randn(3, 34)*3
        self.assertFalse(torch.allclose(model.route_logits(x)[:, 0], model.route_logits(altered)[:, 0]))
        permutation = torch.tensor([0, 3, 1, 2])
        swapped = torch.cat([x[:, :136].reshape(-1, 4, 34)[:, permutation].flatten(1), x[:, -3:]], 1)
        self.assertTrue(torch.allclose(model.route_logits(swapped), model.route_logits(x)[:, permutation], atol=1e-6))

    def test_joint_router_learns_from_route_labels_and_is_isolated_from_td(self):
        *_, agent = self.fixture(self.config(joint_router=True))
        x = torch.randn(1, 139); x[:, -3] = -1
        agent.observe_routes(x, torch.tensor([1., 0, 0, 0]))
        self.assertGreater(float(agent.policy.comparison.output[-1].weight.abs().sum()), 0.)
        before = {k: v.detach().clone() for k, v in agent.policy.named_routing_parameters()}
        for _ in range(40):
            agent.push(x, 4, .5, x, True)
            agent.push_pair(x, 4, .5, x, x, True)
        agent.learn_td()
        self.assertEqual(agent.td_updates, 1)
        for k, v in agent.policy.named_routing_parameters(): self.assertTrue(torch.equal(v, before[k]))

    def test_joint_router_does_not_learn_from_unqueried_labels(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture(self.config(joint_router=True))
        start = deepcopy(agent.policy.state_dict())
        m, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        after = deepcopy(agent.policy.state_dict())
        weights = [deepcopy(c.state_dict()) for c in bank]
        paid_ids = set(m['queried_source_ids'])
        labels = domain[1].copy()
        for i, source_id in enumerate(domain[2]):
            if source_id not in paid_ids: labels[i] = 1-labels[i]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=start)
        _, repeated = episode(a, cnn, bank, initial, other, (domain[0], labels, domain[2]), anchors)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in repeated])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in repeated])
        for k, v in after.items(): self.assertTrue(torch.equal(v, other.policy.state_dict()[k]))
        for saved, context in zip(weights, bank):
            for k, v in saved.items(): self.assertTrue(torch.equal(v, context.state_dict()[k]))

    def test_paid_only_training_cannot_use_unqueried_train_labels(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        start = deepcopy(agent.policy.state_dict())
        m, trace = episode(a, cnn, bank, initial, agent, domain, anchors,
                           training=True, paid_supervision=True)
        self.assertEqual(m['route_updates'], m['labels'])
        self.assertEqual(m['supervision_protocol'], 'paid feedback only')
        after = deepcopy(agent.policy.state_dict()); bank_after = [deepcopy(c.state_dict()) for c in bank]
        paid = set(m['queried_source_ids']); labels = domain[1].copy()
        for i, source in enumerate(domain[2]):
            if source not in paid: labels[i] = 1-labels[i]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=start)
        _, second = episode(a, cnn, bank, initial, other, (domain[0], labels, domain[2]), anchors,
                            training=True, paid_supervision=True)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in second])
        for k, v in after.items(): self.assertTrue(torch.equal(v, other.policy.state_dict()[k]))
        for saved, context in zip(bank_after, bank):
            for k, v in saved.items(): self.assertTrue(torch.equal(v, context.state_dict()[k]))

    def test_sequence_validation_audit_is_disjoint_and_observer_labels_cannot_change_learning(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.levels = [0, 1]; a.audit_fraction = .2
        domains = {0: domain, 1: (domain[0]+.1, domain[1], domain[2])}
        streams, audits, clean, ids = split_validation(domains, domain, 42, .2)
        self.assertEqual(set(ids), set(np.random.default_rng(42+8121).permutation(16)[:3]))
        for d in streams.values(): self.assertFalse(set(ids) & set(d[2]))
        start = deepcopy(agent.policy.state_dict())
        result = validate_sequence(a, cnn, bank, initial, agent, streams, audits, clean, anchors)
        after = deepcopy(agent.policy.state_dict()); bank_after = [deepcopy(c.state_dict()) for c in bank]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=start)
        flipped = {level: (x, 1-y) for level, (x, y) in audits.items()}
        validate_sequence(a, cnn, bank, initial, other, streams, flipped, (clean[0], 1-clean[1]), anchors)
        for k, v in after.items(): self.assertTrue(torch.equal(v, other.policy.state_dict()[k]))
        for saved, context in zip(bank_after, bank):
            for k, v in saved.items(): self.assertTrue(torch.equal(v, context.state_dict()[k]))
        self.assertEqual(len(result['audit_accuracy_by_stage']), 3)
        self.assertEqual(result['labels'], sum(int(a.budget_fraction*len(d[1])) for d in streams.values()))
        scores = np.asarray(result['audit_accuracy_by_stage'])
        self.assertAlmostEqual(result['accuracy'], scores[-1].mean())
        self.assertAlmostEqual(result['forgetting_pp'], 100*(scores[1:, 0].max()-scores[-1, 0]))
        self.assertAlmostEqual(result['bwt_pp'], 100*(scores[-1, 0]-scores[1, 0]))

    def test_training_sequence_keeps_state_and_uses_exact_paid_caps(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.levels = [0, 1]
        result = fit_sequence(a, cnn, bank, initial, agent, {0: domain, 1: domain}, anchors, 42)
        self.assertEqual(result['labels'], 16)
        self.assertEqual(agent.route_updates, 16)
        self.assertEqual([r['route_updates'] for r in result['stages']], [8, 8])
        # Exactly one bank-generation reset plus the 16 paid updates.
        self.assertEqual(int(agent.versions.sum()), 3*32+16)

    def test_ranking_ignores_uninformative_rows_and_has_correct_direction(self):
        z = torch.zeros(3, 4, requires_grad=True)
        y = torch.tensor([[1., 0, 1, 0], [1., 1, 1, 1], [0., 0, 0, 0]])
        loss = ranking_loss(z, y)
        self.assertEqual(float(loss[1]), 0); self.assertEqual(float(loss[2]), 0)
        loss.sum().backward()
        self.assertLess(float(z.grad[0, 0]), 0)
        self.assertGreater(float(z.grad[0, 1]), 0)
        self.assertEqual(float(z.grad[1:].abs().sum()), 0)

    def test_informative_weight_preserves_calibration_and_emphasizes_route_choices(self):
        y = torch.tensor([[1., 0, 1, 0], [0., 0, 0, 0]])
        z = torch.zeros(2, 4, requires_grad=True)
        baseline = routing_objective(z, y, torch.ones(2), .2)
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none').mean(1)
                    + .2*ranking_loss(z, y)).mean()
        self.assertTrue(torch.equal(baseline, expected))
        g0 = torch.autograd.grad(baseline, z)[0]
        boosted = routing_objective(z, y, torch.ones(2), .2, 3.)
        g1 = torch.autograd.grad(boosted, z)[0]
        self.assertTrue(torch.allclose(g1[0], 1.5*g0[0]))
        self.assertTrue(torch.allclose(g1[1], .5*g0[1]))
        self.assertGreater(float(g1[1].abs().sum()), 0)
        for target in (torch.zeros_like(y), torch.ones_like(y)):
            self.assertTrue(torch.equal(routing_objective(z, target, torch.ones(2), .2),
                                        routing_objective(z, target, torch.ones(2), .2, 3.)))

    def test_informative_routing_leaves_td_heads_and_query_objective_independent(self):
        *_, agent = self.fixture()
        agent.informative_weight = 3.
        states = torch.randn(1, 139)
        before = [deepcopy(h.state_dict()) for h in (agent.policy.future, agent.policy.update_value)]
        agent.observe_routes(states, torch.tensor([1., 0, 1, 0]))
        for saved, head in zip(before, (agent.policy.future, agent.policy.update_value)):
            for k, v in saved.items(): self.assertTrue(torch.equal(v, head.state_dict()[k]))
        correctness = deepcopy(agent.policy.correctness.state_dict())
        states[:, -3] = -1
        for _ in range(40):
            agent.push(states, 4, .5, states, True)
            agent.push_pair(states, 4, .5, states, states, True)
        agent.learn_td()
        self.assertEqual(agent.td_updates, 1)
        for k, v in correctness.items(): self.assertTrue(torch.equal(v, agent.policy.correctness.state_dict()[k]))

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
        a.router_refresh = True
        snapshot = deepcopy(agent.policy.state_dict()); cnn_before = deepcopy(cnn.state_dict())
        learned, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        first = [deepcopy(context.state_dict()) for context in bank]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=snapshot)
        random, _ = episode(a, cnn, bank, initial, other, domain, anchors, mode='random')
        self.assertEqual(learned['labels'], 8); self.assertEqual(random['labels'], 8)
        self.assertEqual(learned['fresh_route_updates'], 8)
        self.assertEqual(random['fresh_route_updates'], 0)
        self.assertEqual(learned['slot_queries'], random['slot_queries'])
        for before, context in zip(first, bank):
            for name, value in before.items(): self.assertTrue(torch.equal(value, context.state_dict()[name]))
        for name, value in cnn_before.items(): self.assertTrue(torch.equal(value, cnn.state_dict()[name]))
        self.assertTrue(all(row['labels_used'] <= 8 for row in trace))

    def test_unqueried_labels_cannot_change_learning_or_actions(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.router_refresh = True
        agent.informative_weight = 3.
        snapshot = deepcopy(agent.policy.state_dict())
        _, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        weights = deepcopy(agent.policy.state_dict()); context_weights = [deepcopy(c.state_dict()) for c in bank]
        queried = {row['source_row'] for row in trace if row['queried']}
        y = domain[1].copy()
        for i in range(len(y)):
            if i not in queried: y[i] = 1-y[i]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, policy_state=snapshot,
                                 informative_weight=3.)
        _, second = episode(a, cnn, bank, initial, other, (domain[0], y, domain[2]), anchors)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in second])
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in second])
        for name, value in weights.items(): self.assertTrue(torch.equal(value, other.policy.state_dict()[name]))
        for before, context in zip(context_weights, bank):
            for name, value in before.items(): self.assertTrue(torch.equal(value, context.state_dict()[name]))

    def test_zero_budget_never_acquires_or_updates(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.router_refresh = True
        a.budget_fraction = 0
        m, trace = episode(a, cnn, bank, initial, agent, domain, anchors)
        self.assertEqual(m['labels'], 0); self.assertEqual(m['route_updates'], 0); self.assertEqual(m['td_updates'], 0)
        self.assertEqual(m['fresh_route_updates'], 0)
        self.assertTrue(all(not row['queried'] for row in trace))

    def test_refresh_relabels_current_bank_and_only_changes_routing_parameters(self):
        cnn, bank, initial, a, domain, anchors, agent = self.fixture()
        a.router_refresh = True
        sample_original = ProtectedMemory.sample
        refresh_original = agent.refresh_routes
        captured = {}; checked = []
        def sample(memory, current):
            batch = sample_original(memory, current)
            captured['batch'] = batch
            return batch
        def refresh(states, correct):
            batch = captured['batch']
            with torch.no_grad():
                expected = torch.stack([cnn(batch.x), *[cnn(c(batch.x)) for c in bank]], 1).argmax(2)
            self.assertTrue(torch.equal(correct, expected == batch.y[:, None]))
            self.assertFalse(states.requires_grad)
            before = [deepcopy(c.state_dict()) for c in bank]
            heads = [deepcopy(agent.policy.future.state_dict()), deepcopy(agent.policy.update_value.state_dict())]
            route_before = deepcopy(agent.policy.correctness.state_dict())
            refresh_original(states, correct)
            self.assertTrue(any(not torch.equal(v, agent.policy.correctness.state_dict()[k]) for k, v in route_before.items()))
            for saved, context in zip(before, bank):
                for k, v in saved.items(): self.assertTrue(torch.equal(v, context.state_dict()[k]))
            for saved, head in zip(heads, [agent.policy.future, agent.policy.update_value]):
                for k, v in saved.items(): self.assertTrue(torch.equal(v, head.state_dict()[k]))
            checked.append(len(states))
        with patch.object(ProtectedMemory, 'sample', sample), patch.object(agent, 'refresh_routes', refresh):
            m, _ = episode(a, cnn, bank, initial, agent, domain, anchors)
        self.assertEqual(len(checked), 8)
        self.assertEqual(m['fresh_route_rows'], sum(checked))

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
            for level in (0, 1, 5, 7):
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
            old_router = legacy.StructuredQ(128 + 4*2 + 14)
            torch.save(dict(version=1, route_width=150, policy_state_dict=old_router.state_dict(),
                model_hashes={name: legacy.core.file_sha256(models/name) for name in runner.SOURCE_FILES}),
                models/'structured_bank_ddqn.pth')
            output = root/'enhanced'
            common = ['--output', str(output), '--burst-dir', str(burst), '--device', 'cpu']
            with redirect_stdout(io.StringIO()):
                runner.main(['train', *common, '--source-run', str(source), '--profile', 'cascade',
                    '--warm-start-selector', '--epochs', '1', '--context-epochs', '1',
                    '--warmup-episodes', '1', '--levels', '0'])
                runner.main(['evaluate', *common, '--levels', '5', '7', '--modes', 'learned', '--continual', '--router-refresh', '--query-schedule', 'uniform'])
            results = next((output/'Results').glob('evaluation_*'))
            m = json.loads((results/'learned_level5_metrics.json').read_text())
            self.assertTrue(m['classifier_unchanged'])
            self.assertLessEqual(m['labels'], m['label_cap'])
            self.assertTrue((results/'sequential_summary.csv').is_file())
            config = json.loads((results/'execution.json').read_text())
            self.assertEqual(config['router_refresh']['extra_label_capacity'], 0)
            self.assertGreater(m['fresh_route_updates'], 0)
            audit = set(config['audit_source_ids'])
            self.assertFalse(audit & set(m['source_ids']))
            expected_audit = set(np.random.default_rng(42+8121).permutation(np.arange(60, 80))[:4])
            self.assertEqual(audit, expected_audit)
            later = json.loads((results/'learned_level7_metrics.json').read_text())
            self.assertFalse(audit & set(later['source_ids']))
            matrix = pd.read_csv(results/'sequential_audit_matrix.csv')
            self.assertEqual(set(matrix.Stage), {0, 1, 2})
            self.assertEqual(set(matrix.AuditLevel), {0, 5, 7})
            self.assertEqual(len(matrix), 9)
            summary = pd.read_csv(results/'sequential_summary.csv').iloc[0]
            clean_scores = matrix[matrix.AuditLevel == 0].sort_values('Stage').Accuracy.to_numpy()
            self.assertAlmostEqual(summary.CleanDrop, clean_scores[0]-clean_scores[-1])
            first_scores = matrix[matrix.AuditLevel == 5].sort_values('Stage').Accuracy.to_numpy()
            self.assertAlmostEqual(summary.Forgetting, max(first_scores[1:])-first_scores[-1])
            self.assertAlmostEqual(summary.BWT, first_scores[-1]-first_scores[1])
            self.assertEqual(summary.Labels, m['labels']+later['labels'])
            self.assertEqual(m['config_fingerprint'], EnhancementConfig(**config['config']).fingerprint)
            # Preexisting results and training runs are protected from accidental overwrite.
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                runner.main(['evaluate', *common, '--levels', '5', '7', '--modes', 'learned', '--continual', '--router-refresh', '--query-schedule', 'uniform'])
            with redirect_stdout(io.StringIO()):
                runner.main(['evaluate', *common, '--levels', '5', '7', '--modes', 'learned',
                    '--continual', '--query-schedule', 'uniform', '--router-informative-weight', '3'])
            extra = next(p for p in (output/'Results').glob('evaluation_*') if p != results)
            meta = json.loads((extra/'execution.json').read_text())
            self.assertEqual(meta['router_informative_weight'], 3.)
            self.assertNotIn('router_refresh', meta)
            measured = json.loads((extra/'learned_level5_metrics.json').read_text())
            self.assertEqual(measured['fresh_route_updates'], 0)
            self.assertEqual(measured['labels'], measured['label_cap'])
            self.assertTrue(measured['classifier_unchanged'])
            source_hash = legacy.core.file_sha256(output/'Models'/'final_enhancement.pth')
            joint_output = root/'joint'
            joint_common = ['--output', str(joint_output), '--burst-dir', str(burst), '--device', 'cpu']
            with redirect_stdout(io.StringIO()):
                runner.main(['train', *joint_common, '--source-run', str(output), '--profile', 'cascade',
                    '--joint-router', '--warm-start-enhancement', '--epochs', '1', '--levels', '0'])
                runner.main(['evaluate', *joint_common, '--levels', '5', '7', '--modes', 'learned',
                    '--continual', '--query-schedule', 'uniform', '--update-policy', 'learned'])
            checkpoint = torch.load(joint_output/'Models'/'final_enhancement.pth', weights_only=False)
            self.assertTrue(checkpoint['config']['joint_router'])
            self.assertEqual(checkpoint['parent_checkpoint_sha256'], source_hash)
            self.assertIn(checkpoint['selected_epoch'], (0, 1))
            self.assertTrue(any(k.startswith('comparison.') for k in checkpoint['policy_state_dict']))
            self.assertEqual(legacy.core.file_sha256(output/'Models'/'enhanced_context.pth'),
                             legacy.core.file_sha256(joint_output/'Models'/'enhanced_context.pth'))
            self.assertEqual(legacy.core.file_sha256(output/'Models'/'final_enhancement.pth'), source_hash)
            training = json.loads((joint_output/'enhancement_training.json').read_text())
            self.assertEqual(training['epochs'][0]['epoch'], 0)
            self.assertEqual(training['epochs'][0]['train_labels'], 0)
            joint_results = next((joint_output/'Results').glob('evaluation_*'))
            joint_metrics = json.loads((joint_results/'learned_level5_metrics.json').read_text())
            self.assertTrue(joint_metrics['classifier_unchanged'])
            self.assertEqual(joint_metrics['labels'], joint_metrics['label_cap'])
            self.assertEqual(joint_metrics['fresh_route_updates'], 0)
            sequence_source = root/'sequence_source'
            aligned_output = root/'aligned'
            calibration_sizes = []; original_calibration = PreRouter.calibrate
            def calibrated(model, train_h, val_h):
                calibration_sizes.append(len(val_h))
                return original_calibration(model, train_h, val_h)
            with redirect_stdout(io.StringIO()), patch.object(PreRouter, 'calibrate', calibrated):
                runner.main(['train', '--source-run', str(source), '--output', str(sequence_source),
                    '--burst-dir', str(burst), '--device', 'cpu', '--profile', 'accuracy',
                    '--warm-start-selector', '--epochs', '1', '--warmup-episodes', '0', '--levels', '0', '1'])
                runner.main(['train', '--source-run', str(sequence_source), '--output', str(aligned_output),
                    '--burst-dir', str(burst), '--device', 'cpu', '--profile', 'accuracy',
                    '--warm-start-enhancement', '--aligned-sequence', '--query-schedule', 'uniform',
                    '--epochs', '1', '--levels', '0', '1'])
                runner.main(['evaluate', '--output', str(aligned_output), '--burst-dir', str(burst),
                    '--device', 'cpu', '--continual', '--query-schedule', 'uniform',
                    '--modes', 'learned', '--levels', '5', '7'])
            aligned = json.loads((aligned_output/'enhancement_training.json').read_text())
            self.assertEqual(calibration_sizes, [20, 16])
            protocol = aligned['training_protocol']
            self.assertEqual(protocol['mode'], 'aligned_sequence')
            audit_ids = set(protocol['validation_audit_source_ids'])
            for adaptation_ids in protocol['validation_adaptation_source_ids'].values():
                self.assertFalse(audit_ids & set(adaptation_ids))
            self.assertFalse(aligned['config']['joint_router'])
            self.assertEqual(aligned['epochs'][0]['train_labels'], 0)
            for stage in aligned['epochs'][1]['train_stages']:
                self.assertEqual(stage['labels'], stage['cap'])
                self.assertEqual(stage['route_updates'], stage['labels'])
            saved = torch.load(aligned_output/'Models'/'final_enhancement.pth', weights_only=False)
            self.assertEqual(saved['training_protocol'], protocol)
            best_row = next(row for row in aligned['epochs'] if row['epoch'] == saved['selected_epoch'])
            self.assertAlmostEqual(saved['selection_score'], best_row['selection_score'])
            audit_scores = np.asarray(best_row['validation_sequence']['audit_accuracy_by_stage'])
            self.assertAlmostEqual(saved['validation_accuracy'], audit_scores[-1].mean())
            result_dir = next((aligned_output/'Results').glob('evaluation_*'))
            execution = json.loads((result_dir/'execution.json').read_text())
            self.assertEqual(execution['training_protocol'], protocol)
            # Calibrate using TRAIN only, without persisting temporary adaptation.
            calibration_path = sequence_source/'Calibration'/'utility_train_seed42.json'
            checkpoint_path = sequence_source/'Models'/'final_enhancement.pth'
            checkpoint_hash = legacy.core.file_sha256(checkpoint_path)
            read_splits = []; original_read = legacy.read_domain
            def read_train(*args, **kwargs):
                read_splits.append(args[3])
                return original_read(*args, **kwargs)
            utility_common = ['--output', str(sequence_source), '--burst-dir', str(burst),
                '--device', 'cpu', '--query-schedule', 'uniform', '--update-policy', 'learned',
                '--utility-calibration', str(calibration_path)]
            with redirect_stdout(io.StringIO()), patch.object(legacy, 'read_domain', read_train):
                runner.main(['calibrate-utility', *utility_common, '--levels', '0', '1'])
            self.assertEqual(set(read_splits), {'train_ids'})
            calibration = json.loads(calibration_path.read_text())
            self.assertEqual(calibration['scope'], 'TRAIN')
            self.assertEqual(calibration['levels'], [0,1])
            self.assertEqual(calibration['rows'], 80)
            self.assertEqual(calibration['checkpoint_sha256'], checkpoint_hash)
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                runner.main(['calibrate-utility', *utility_common, '--levels', '0', '1'])
            with redirect_stdout(io.StringIO()):
                runner.main(['evaluate', *utility_common, '--continual', '--modes', 'learned',
                    '--levels', '5', '7'])
            normalized_results = next((sequence_source/'Results').glob('evaluation_*'))
            metadata = json.loads((normalized_results/'execution.json').read_text())
            self.assertEqual(metadata['utility_normalization']['calibration'], calibration)
            trace = pd.read_csv(normalized_results/'learned_level5_trace.csv')
            self.assertIn('raw_update_values', trace.columns)
            self.assertIn('selection_scores', trace.columns)
            metrics = json.loads((normalized_results/'learned_level5_metrics.json').read_text())
            self.assertEqual(metrics['labels'], metrics['label_cap'])
            self.assertEqual(legacy.core.file_sha256(checkpoint_path), checkpoint_hash)
            cf_output = root/'counterfactual'
            with redirect_stdout(io.StringIO()):
                runner.main(['train', '--source-run', str(sequence_source), '--output', str(cf_output),
                    '--burst-dir', str(burst), '--device', 'cpu', '--profile', 'accuracy',
                    '--warm-start-enhancement', '--aligned-sequence', '--counterfactual-utility',
                    '--query-schedule', 'uniform', '--epochs', '1', '--levels', '0', '1'])
                runner.main(['evaluate', '--output', str(cf_output), '--burst-dir', str(burst),
                    '--device', 'cpu', '--continual', '--query-schedule', 'uniform',
                    '--modes', 'learned', '--levels', '5', '7'])
            cf_log = json.loads((cf_output/'enhancement_training.json').read_text())
            self.assertEqual(cf_log['training_protocol']['utility_supervision']['alternatives'],3)
            self.assertGreater(cf_log['epochs'][1]['counterfactual_updates_total'],0)
            for stage in cf_log['epochs'][1]['train_stages']:
                self.assertEqual(stage['counterfactual_context_steps'],3*stage['labels'])
                self.assertEqual(stage['route_updates'],0)
                self.assertEqual(stage['td_updates'],0)
            cf_checkpoint = torch.load(cf_output/'Models'/'final_enhancement.pth',weights_only=False)
            cf_source = torch.load(checkpoint_path,weights_only=False)
            for key,value in cf_checkpoint['policy_state_dict'].items():
                if not key.startswith('update_value.'):
                    self.assertTrue(torch.equal(value,cf_source['policy_state_dict'][key]))
            self.assertEqual(cf_checkpoint['parent_checkpoint_sha256'],checkpoint_hash)
            cf_results = next((cf_output/'Results').glob('evaluation_*'))
            cf_metrics = json.loads((cf_results/'learned_level5_metrics.json').read_text())
            self.assertNotIn('counterfactual_rows',cf_metrics)
            self.assertEqual(cf_metrics['labels'],cf_metrics['label_cap'])
            # The trained utility model supports policy-query RL in both tests.
            cf_hash = legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth')
            existing = set((cf_output/'Results').glob('evaluation_*'))
            policy_common = ['evaluate','--output',str(cf_output),'--burst-dir',str(burst),
                '--device','cpu','--query-schedule','policy','--update-policy','learned',
                '--modes','learned','--seed','42','--levels','5','7']
            original_episode = runner.episode; starts = []
            def policy_episode(*args,**kwargs):
                agent = args[4]; continuing = bool(kwargs.get('continuation'))
                starts.append(continuing)
                self.assertEqual(agent.counterfactual_updates,0)
                if not continuing:
                    self.assertEqual(len(agent.route_memory),0)
                    self.assertEqual(len(agent.td_memory),0)
                    for key,value in agent.policy.state_dict().items():
                        self.assertTrue(torch.equal(value,cf_checkpoint['policy_state_dict'][key]))
                return original_episode(*args,**kwargs)
            with redirect_stdout(io.StringIO()), patch.object(runner,'episode',policy_episode), \
                    patch('enhancements.stream.alternatives',side_effect=AssertionError('No TEST simulation')):
                runner.main([*policy_common,'--continual'])
                runner.main(policy_common)
            self.assertEqual(starts,[False,True,False,False])
            policy_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(policy_results),2)
            kinds=set()
            for result in policy_results:
                execution=json.loads((result/'execution.json').read_text())
                self.assertEqual(execution['query_schedule'],'policy')
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertNotIn('utility_normalization',execution)
                kinds.add(execution['continual'])
                self.assertEqual((result/'sequential_summary.csv').exists(),execution['continual'])
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertLessEqual(measured['labels'],measured['label_cap'])
                    self.assertNotIn('counterfactual_rows',measured)
                    self.assertEqual(measured['samples'],16 if execution['continual'] else 20)
            self.assertEqual(kinds,{True,False})
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Actual two-Context ablation uses the SAME shared checkpoint.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            with redirect_stdout(io.StringIO()), \
                    patch('enhancements.stream.alternatives',side_effect=AssertionError('No TEST simulation')):
                runner.main([*policy_common,'--context-count','2','--continual'])
                runner.main([*policy_common,'--context-count','2'])
            reduced=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(reduced),2)
            for result in reduced:
                execution=json.loads((result/'execution.json').read_text())
                self.assertEqual(execution['context_count'],2)
                self.assertEqual(execution['action_count'],5)
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertEqual(len(measured['slot_queries']),2)
                    self.assertEqual(len(measured['action_counts']),5)
                    self.assertLessEqual(measured['labels'],measured['label_cap'])
                traces=pd.read_csv(result/'learned_level5_trace.csv')
                self.assertTrue(traces.action.between(0,4).all())
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Single Context, uniform feedback: same checkpoint, both protocols.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            single_common=['evaluate','--output',str(cf_output),'--burst-dir',str(burst),
                '--device','cpu','--query-schedule','uniform','--update-policy','learned',
                '--modes','learned','--seed','42','--levels','5','7','--context-count','1']
            with redirect_stdout(io.StringIO()), \
                    patch('enhancements.stream.alternatives',side_effect=AssertionError('No TEST simulation')):
                runner.main([*single_common,'--continual'])
                runner.main(single_common)
            single_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(single_results),2)
            kinds=set()
            for result in single_results:
                execution=json.loads((result/'execution.json').read_text())
                self.assertEqual(execution['context_count'],1)
                self.assertEqual(execution['action_count'],3)
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertEqual(execution['query_schedule'],'uniform')
                kinds.add(execution['continual'])
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertEqual(len(measured['slot_queries']),1)
                    self.assertEqual(len(measured['action_counts']),3)
                    self.assertEqual(measured['labels'],measured['label_cap'])
                    self.assertEqual(measured['samples'],16 if execution['continual'] else 20)
                traces=pd.read_csv(result/'learned_level5_trace.csv')
                self.assertTrue(traces.action.between(0,2).all())
                self.assertTrue(all(len(json.loads(x))==2 for x in traces.predictions))
                self.assertTrue(all(len(json.loads(x))==5 for x in traces.scores))
            self.assertEqual(kinds,{True,False})
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Random selection: both real CLI protocols, no selector execution or learning.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            random_common=single_common.copy()
            random_common[random_common.index('uniform')]='policy'
            random_common[random_common.index('--modes')+1]='random'
            with redirect_stdout(io.StringIO()), \
                    patch.object(EfficientQ,'forward_for_action',side_effect=AssertionError('No RL action')), \
                    patch.object(EfficientQ,'route_logits',side_effect=AssertionError('No RL observer')):
                runner.main(random_common)
                runner.main([*random_common,'--continual'])
            random_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(random_results),2)
            for result in random_results:
                execution=json.loads((result/'execution.json').read_text())
                self.assertEqual(execution['random_control']['version'],2)
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertEqual((result/'sequential_summary.csv').exists(),execution['continual'])
                for level in (5,7):
                    measured=json.loads((result/f'random_level{level}_metrics.json').read_text())
                    self.assertEqual(measured['route_updates'],0)
                    self.assertEqual(measured['td_updates'],0)
                    self.assertLessEqual(measured['labels'],measured['label_cap'])
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Expanded banks: each Context count runs both protocols, same checkpoint.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            for count in (4,5):
                expanded=['evaluate','--output',str(cf_output),'--burst-dir',str(burst),
                    '--device','cpu','--query-schedule','uniform','--update-policy','learned',
                    '--modes','learned','--seed','42','--levels','5','7','--context-count',str(count)]
                with redirect_stdout(io.StringIO()), \
                        patch('enhancements.stream.alternatives',side_effect=AssertionError('No TEST simulation')):
                    runner.main([*expanded,'--continual'])
                    runner.main(expanded)
            expanded_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(expanded_results),4)
            cases=set()
            for result in expanded_results:
                execution=json.loads((result/'execution.json').read_text())
                count=execution['context_count']; cases.add((count,execution['continual']))
                self.assertEqual(execution['action_count'],2*count+1)
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertEqual(execution['query_schedule'],'uniform')
                self.assertIn('expansion',execution['context_count_protocol'])
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertEqual(len(measured['slot_queries']),count)
                    self.assertEqual(len(measured['action_counts']),2*count+1)
                    self.assertEqual(measured['labels'],measured['label_cap'])
                    self.assertEqual(measured['samples'],16 if execution['continual'] else 20)
                traces=pd.read_csv(result/'learned_level5_trace.csv')
                self.assertTrue(traces.action.between(0,2*count).all())
                self.assertTrue(all(len(json.loads(x))==count+1 for x in traces.predictions))
                self.assertTrue(all(len(json.loads(x))==3*count+2 for x in traces.scores))
            self.assertEqual(cases,{(4,True),(4,False),(5,True),(5,False)})
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Isolated online replay-weight override: same source, both protocols.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            with redirect_stdout(io.StringIO()):
                runner.main([*single_common,'--online-memory-ce-weight','0.3','--continual'])
                runner.main([*single_common,'--online-memory-ce-weight','0.3'])
            retention_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(retention_results),2)
            kinds=set()
            for result in retention_results:
                execution=json.loads((result/'execution.json').read_text())
                kinds.add(execution['continual'])
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertEqual(execution['config']['memory_ce_weight'],.3)
                original_config=dict(cf_checkpoint['config']); effective_config=dict(execution['config'])
                effective_config['memory_ce_weight']=original_config['memory_ce_weight']
                self.assertEqual(effective_config,original_config)
                override=execution['online_adaptation_override']['memory_ce_weight']
                self.assertEqual(override,{'checkpoint':.1,'online':.3})
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    self.assertEqual(measured['labels'],measured['label_cap'])
                    self.assertEqual(len(measured['slot_queries']),1)
                    self.assertTrue(measured['classifier_unchanged'])
                    self.assertEqual(measured['config_fingerprint'],EnhancementConfig(**execution['config']).fingerprint)
            self.assertEqual(kinds,{True,False})
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            # Protected-reference acceptance: both query schedules and both protocols.
            existing=set((cf_output/'Results').glob('evaluation_*'))
            with redirect_stdout(io.StringIO()):
                runner.main([*single_common,'--context-damage-guard','--continual'])
                runner.main([*single_common,'--context-damage-guard'])
                rl_single=[*policy_common,'--context-count','1','--context-damage-guard']
                runner.main([*rl_single,'--continual'])
                runner.main(rl_single)
            guarded_results=set((cf_output/'Results').glob('evaluation_*'))-existing
            self.assertEqual(len(guarded_results),4)
            kinds=set()
            schedules=set()
            for result in guarded_results:
                execution=json.loads((result/'execution.json').read_text())
                kinds.add(execution['continual'])
                schedules.add((execution['query_schedule'],execution['continual']))
                self.assertEqual(execution['checkpoint_sha256'],cf_hash)
                self.assertEqual(execution['config'],cf_checkpoint['config'])
                self.assertEqual(execution['config']['memory_ce_weight'],.1)
                self.assertNotIn('online_adaptation_override',execution)
                self.assertEqual(execution['context_damage_guard']['step_scales'],[1.,.5,.25])
                for level in (5,7):
                    measured=json.loads((result/f'learned_level{level}_metrics.json').read_text())
                    guard=measured['damage_guard']
                    self.assertEqual(guard['accepted']+guard['rejected'],measured['labels'])
                    self.assertLessEqual(measured['labels'],measured['label_cap'])
                    if execution['query_schedule']=='uniform':
                        self.assertEqual(measured['labels'],measured['label_cap'])
                    self.assertLessEqual(guard['max_accepted_harm'],guard['max_harm'])
                    self.assertTrue(measured['classifier_unchanged'])
                traces=pd.read_csv(result/'learned_level5_trace.csv')
                self.assertIn('damage_guard',traces.columns)
            self.assertEqual(schedules,{('uniform',True),('uniform',False),('policy',True),('policy',False)})
            self.assertEqual(kinds,{True,False})
            self.assertEqual(legacy.core.file_sha256(cf_output/'Models'/'final_enhancement.pth'),cf_hash)
            self.assertEqual(legacy.core.file_sha256(checkpoint_path),checkpoint_hash)


if __name__ == '__main__': unittest.main()
