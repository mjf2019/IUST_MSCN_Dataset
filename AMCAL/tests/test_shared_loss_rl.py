"""Small CPU software checks only; no dataset experiment or installation."""
from copy import deepcopy
from dataclasses import asdict, replace
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amcal_final_enhancement as base
import amcal_shared_loss_rl as runner
from enhancements.agent import EnhancementAgent
from enhancements.config import EnhancementConfig
from enhancements.paid_replay import paid_context_step


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = self.conv_block2 = self.conv_block3 = nn.Identity()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 2))

    def forward(self, x): return self.dense_layers(self.flatten(x))


class TinyContext(nn.Module):
    def __init__(self):
        super().__init__(); self.layer = nn.Linear(8, 8)
        nn.init.eye_(self.layer.weight); nn.init.zeros_(self.layer.bias)

    def forward(self, x): return self.layer(x.squeeze(1)).unsqueeze(1)


class SharedLossRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); base.legacy.core.device = torch.device('cpu')

    def setUp(self): torch.manual_seed(42)

    def config(self):
        return replace(EnhancementConfig.preset('accuracy'), shared_loss_rl=True, paired_weight=0.,
            source_capacity=8, historical_capacity=8, recent_capacity=16,
            reference_size=8, memory_replay_size=2).validated()

    def fixture(self, n=64):
        cnn = TinyCNN().eval().requires_grad_(False); context = TinyContext().eval()
        a = base.parser().parse_args(['train', '--output', 'unused', '--burst-dir', 'unused'])
        a.config = self.config(); a.seed = 42; a.classes = ['0', '1']; a.context_count = 1
        a.context_lr = .00025; a.context_alpha = .8; a.budget_fraction = .2
        a.query_schedule = 'policy'; a.update_policy = 'learned'; a.paid_replay_mode = 'batch5'
        a.train_epsilon = .3; a.eval_epsilon = 0.; a.selection_cost = .05
        a.bank_utility = False; a.balanced_reward_reference = True
        agent = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, slots=1)
        x = torch.randn(n, 1, 8); y = np.arange(n) % 2
        anchors = torch.randn(12, 1, 8)+5, np.arange(12) % 2
        return a, cnn, context, agent, (x, y, np.arange(n)), anchors

    def constants(self, agent, future, advantage):
        for head, value in [(agent.policy.future, future), (agent.policy.update_value, advantage)]:
            for p in head.parameters(): p.data.zero_()
            head[-1].bias.data.fill_(value)
        agent.target.load_state_dict(agent.policy.state_dict())

    def test_old_serialization_fingerprint_and_q_semantics_are_unchanged(self):
        config = EnhancementConfig.preset('accuracy')
        expected = asdict(config); expected.pop('shared_loss_rl')
        self.assertEqual(config.as_dict(), expected)
        self.assertEqual(config.fingerprint, hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12])
        old = base.legacy.StructuredQ(34)
        from enhancements.agent import EfficientQ
        new = EfficientQ(34); new.load_state_dict(old.state_dict())
        state = torch.randn(3, 139)
        torch.testing.assert_close(new(state), old(state), rtol=0, atol=0)

    def test_query_return_does_not_inherit_the_context_route_quality_offset(self):
        a, cnn, context, agent, data, anchors = self.fixture()
        self.constants(agent, 1., 2.)
        states = torch.zeros(2, 71); states[:, -3] = -1; states[:, -2] = 1
        with patch.object(agent.policy, 'route_logits', return_value=torch.tensor([[9., -9.], [-9., 9.]])):
            scores, _, raw, update = agent.policy.forward_for_action(states)
        torch.testing.assert_close(raw, torch.tensor([[1., 1., 3.], [1., 1., 3.]]))
        self.assertTrue(torch.isneginf(scores[0, 1])); self.assertTrue(torch.isneginf(scores[1, 0]))
        self.assertTrue(all(p.requires_grad for p in agent.policy.future.parameters()))
        self.assertTrue(all(p.requires_grad for p in agent.policy.update_value.parameters()))

    def test_double_dqn_chooses_with_policy_and_evaluates_with_target_and_masks_cap(self):
        _, _, _, agent, _, _ = self.fixture()
        self.constants(agent, 1., 1.)
        for head, value in [(agent.target.future, 10.), (agent.target.update_value, -8.)]:
            for p in head.parameters(): p.data.zero_()
            head[-1].bias.data.fill_(value)
        states = torch.zeros(2, 71); states[:, -3] = -1; states[:, -2] = torch.tensor([1., 0.])
        torch.testing.assert_close(agent._next_values(states), torch.tensor([2., 10.]))

    def test_terminal_td_has_no_bootstrap_and_both_value_heads_learn_without_routing(self):
        _, _, _, agent, _, _ = self.fixture()
        state = torch.zeros(1, 71); state[:, -3] = -1; state[:, -2] = 1
        original = {k:v.clone() for k,v in agent.policy.named_routing_parameters()}
        for p in agent.target.parameters(): p.data.fill_(50.)
        for _ in range(32): agent.push(state, 2, 1., state, True)
        agent.learn_td()
        self.assertAlmostEqual(agent.last_losses['td'], .5, places=6)
        self.assertGreater(float(agent.policy.future[-1].bias.detach()), 0.)
        self.assertGreater(float(agent.policy.update_value[-1].bias.detach()), 0.)
        for k,v in agent.policy.named_routing_parameters(): torch.testing.assert_close(v, original[k], rtol=0, atol=0)
        self.assertFalse(agent.pair_memory)

    def test_reported_payoff_is_the_actual_pre_update_loss_and_preserves_old_adam_update(self):
        a, cnn, context, _, _, _ = self.fixture()
        for size in (1, 5):
            first, second = deepcopy(context), deepcopy(context)
            opt1 = torch.optim.Adam(first.parameters(), lr=a.context_lr, weight_decay=1e-4)
            opt2 = torch.optim.Adam(second.parameters(), lr=a.context_lr, weight_decay=1e-4)
            rows = [(torch.randn(1, 1, 8), torch.tensor([i % 2])) for i in range(size)]
            replay = torch.randn(2, 1, 8), torch.tensor([0, 1])
            x = torch.cat([r[0] for r in rows]); y = torch.cat([r[1] for r in rows])
            with torch.no_grad():
                rx = torch.cat([x, replay[0]]); ry = torch.cat([y, replay[1]])
                transformed = first(rx)
                ce = torch.nn.functional.cross_entropy(cnn(transformed), ry, reduction='none')
                frozen_ce = torch.nn.functional.cross_entropy(cnn(x), y)
                expected = base.legacy.context_objective(ce[:size].mean(), frozen_ce, .8)
                expected += a.config.memory_ce_weight*ce[size:].mean()
                expected += a.config.residual_weight*(transformed-rx).square().mean()
            loss = paid_context_step(a, cnn, first, opt1, *rows[0], cnn(rows[0][0]).detach(), replay, rows[1:], return_loss=True)
            old = paid_context_step(a, cnn, second, opt2, *rows[0], cnn(rows[0][0]).detach(), replay, rows[1:])
            self.assertIsNone(old); self.assertAlmostEqual(loss, float(expected), places=6)
            for k,v in first.state_dict().items(): torch.testing.assert_close(v, second.state_dict()[k], rtol=0, atol=0)

    def test_stream_uses_only_real_transitions_shared_reward_and_never_exceeds_cap(self):
        a, cnn, context, agent, data, anchors = self.fixture()
        self.constants(agent, 0., 1.)
        initial = deepcopy(context.state_dict()); frozen = deepcopy(cnn.state_dict())
        result, trace = base.episode(a, cnn, [context], initial, agent, data, anchors)
        self.assertEqual(len(agent.td_memory), len(data[1]))
        self.assertFalse(agent.pair_memory)
        self.assertLessEqual(result['labels'], result['label_cap'])
        for row, transition in zip(trace, agent.td_memory):
            self.assertEqual(row['action'], transition[1])
            self.assertEqual(row['td_reward'], transition[2])
            if row['queried']: self.assertAlmostEqual(row['td_reward'], row['shared_loss_before']-.05)
            else: self.assertEqual(row['td_reward'], 0.)
        self.assertTrue(trace[-1]['td_terminal'])
        for k,v in cnn.state_dict().items(): torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)

    def test_changing_unqueried_truth_changes_no_action_context_or_td_weight(self):
        a, cnn, context, agent, data, anchors = self.fixture()
        self.constants(agent, 0., 1.)
        saved = deepcopy(agent.policy.state_dict()); initial = deepcopy(context.state_dict())
        first, trace = base.episode(a, cnn, [context], initial, agent, data, anchors)
        paid = {r['source_row'] for r in trace if r['queried']}
        changed = data[1].copy()
        for i in range(len(changed)):
            if i not in paid: changed[i] = 1-changed[i]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, slots=1, policy_state=saved)
        second_context = TinyContext().eval()
        _, repeated = base.episode(a, cnn, [second_context], initial, other, (data[0], changed, data[2]), anchors)
        self.assertEqual([r['action'] for r in trace], [r['action'] for r in repeated])
        for k,v in agent.policy.state_dict().items(): torch.testing.assert_close(v, other.policy.state_dict()[k], rtol=0, atol=0)
        for k,v in context.state_dict().items(): torch.testing.assert_close(v, second_context.state_dict()[k], rtol=0, atol=0)

    def test_snapshot_restores_td_sampler_target_optimizer_and_both_heads(self):
        _, _, _, agent, _, _ = self.fixture()
        states = torch.randn(40, 71); states[:, -3] = -1; states[:, -2] = 1
        for state in states: agent.push(state[None], 2, .3, state[None], True)
        agent.learn_td(); saved = runner.snapshot(agent)
        _, _, _, other, _, _ = self.fixture(); runner.restore(other, saved)
        agent.learn_td(); other.learn_td()
        for k,v in agent.policy.state_dict().items(): torch.testing.assert_close(v, other.policy.state_dict()[k], rtol=0, atol=0)

    def test_current_paid_label_cannot_change_its_own_prediction_or_action(self):
        a, cnn, context, agent, data, anchors = self.fixture()
        self.constants(agent, 0., 1.)
        initial = deepcopy(context.state_dict()); saved = deepcopy(agent.policy.state_dict())
        _, trace = base.episode(a, cnn, [context], initial, agent, data, anchors)
        first_id = trace[0]['source_row']; changed = data[1].copy(); changed[first_id] = 1-changed[first_id]
        other = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, a.config, slots=1, policy_state=saved)
        _, repeated = base.episode(a, cnn, [TinyContext().eval()], initial, other, (data[0], changed, data[2]), anchors)
        self.assertTrue(trace[0]['queried'])
        self.assertEqual(trace[0]['action'], repeated[0]['action'])
        self.assertEqual(trace[0]['prediction'], repeated[0]['prediction'])

    def test_full_runner_factory_training_checkpoint_reuse_and_all_controls_with_real_tiny_networks(self):
        from enhancements.config import EnhancementConfig
        a, cnn, context, _, _, _ = self.fixture()
        old_config = replace(a.config, shared_loss_rl=False, paired_weight=.5)
        source_agent = EnhancementAgent(34, 12, 2, torch.device('cpu'), 42, old_config, slots=1)
        artifact = dict(label_encoder=SimpleNamespace(classes_=np.array(['0', '1'])))
        domains = {}
        for split, size, offset in [('train_ids', 120, 0), ('val_ids', 60, 200), ('test_ids', 60, 400)]:
            x = torch.randn(size, 1, 8)+offset/20
            for level in LEVELS_ALL:
                domains[level, split] = x+level*.01, np.arange(size)%2, np.arange(offset, offset+size)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'source'; models = source/'Models'; models.mkdir(parents=True)
            torch.save(dict(architecture='legacy', model_state_dict=cnn.state_dict()), models/'initial_cnn_model.pth')
            torch.save(dict(initial_state_dict=context.state_dict()), models/'enhanced_context.pth')
            hashes = {name:runner.digest(models/name) for name in ('initial_cnn_model.pth', 'enhanced_context.pth')}
            checkpoint = dict(format_version=base.FORMAT_VERSION, config=old_config.as_dict(), training_levels=[0,1,3],
                model_hashes=hashes, training_domain_sha256={}, policy_state_dict=source_agent.policy.state_dict(),
                prerouter_state_dict=source_agent.pre_router.state_dict())
            torch.save(checkpoint, models/'final_enhancement.pth')
            args = SimpleNamespace(source=source, burst_dir=root, seed=42, device='cpu', cost=.05,
                output=root/'output', dataset='IAT', pilot_rows=32, epochs=1)
            def build_bank(*args, **kwargs): return [deepcopy(context)]
            with patch.object(base, 'load_artifact', return_value=artifact), patch.object(base, 'load_cnn', return_value=cnn), \
                    patch.object(base, 'build_bank', side_effect=build_bank), \
                    patch.object(base.legacy, 'read_domain', side_effect=lambda root, artifact, level, split:domains[level, split]), \
                    redirect_stdout(io.StringIO()):
                experiment = runner.Experiment(args)
                protocol = dict(version=1, audits=deepcopy(experiment.audit_info))
                runner.train(experiment, protocol)
                complete = args.output/'training_complete.json'
                self.assertTrue(complete.exists())
                runner.train(experiment, protocol)
                saved = torch.load(args.output/'Models'/'shared_loss_rl.pth', weights_only=False)
                for k,v in source_agent.policy.state_dict().items():
                    if k.startswith('correctness.'):
                        torch.testing.assert_close(saved['policy_state_dict'][k], v, rtol=0, atol=0)
                rows = runner.evaluate(experiment, saved, 'test_ids', [5], args.output/'Results')
            self.assertEqual(len(rows), 6)
            for row in rows: self.assertLessEqual(row['labels_percent'], 20.)
            test_ids = set(experiment.audit_info['test_ids']['ids'])
            self.assertFalse(test_ids.intersection(experiment.domains['test_ids'][5][2]))
            histories = json.loads((args.output/'training.json').read_text())['epochs']
            self.assertTrue(all(r['skipped_pair_buffer']==0 for r in histories[0]['training']))


LEVELS_ALL = [0, 1, 3, 5, 7, 10, 12, 15, 17, 20]
if __name__ == '__main__': unittest.main()
