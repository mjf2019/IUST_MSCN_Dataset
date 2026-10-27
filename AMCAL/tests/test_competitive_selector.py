import ast
from copy import deepcopy, copy
import contextlib
import io
import json
import hashlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires existing PyTorch; nothing is installed')
class CompetitiveSelectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_compact_features_include_uncertainty_paid_novelty_and_coverage(self):
        from enhancements.competitive_selector import query_features
        logits = torch.tensor([[[3., 0.], [0., 1.]]])
        args = ((logits, torch.zeros(1, 2)), torch.tensor([[1., 0.]]))
        initial = query_features(*args, [], [], 0, 2, 0, 10, 0)
        paid = query_features(*args, [args[1]], [0], 1, 2, 3, 10, 2)
        self.assertEqual(initial.shape, (1, 22))
        self.assertGreater(initial[0, 2*2+4], 0.)
        self.assertEqual(float(initial[0, -8]), 1.)
        self.assertAlmostEqual(float(paid[0, -8]), 0.)
        self.assertAlmostEqual(float(paid[0, -6]), .5)
        torch.testing.assert_close(initial[:, :4], paid[:, :4])

    def test_hard_cap_is_optional_and_no_tail_is_forced(self):
        from enhancements.competitive_selector import select_feedback
        self.assertFalse(select_feedback(.1, 0, 2))
        self.assertFalse(select_feedback(.99, 2, 2))
        self.assertTrue(select_feedback(.6, 1, 2))
        self.assertFalse(select_feedback(.6, 1, 2, training=True, draw=.7))
        self.assertTrue(select_feedback(.6, 1, 2, training=True, draw=.5))

    def test_signed_components_keep_gap_gain_harm_and_do_not_center(self):
        from enhancements.competitive_selector import competitive_score
        scales = dict(gap=2., weighted_gain=4., weighted_harm=.5)
        row = dict(gap=1., weighted_gain=2., weighted_harm=.1)
        self.assertAlmostEqual(competitive_score(row, scales, .05), .75)
        self.assertLess(competitive_score(dict(gap=-1., weighted_gain=0., weighted_harm=0.), scales, .05), 0)

    def test_original_paper_reward_matches_all_four_pre_feedback_cases(self):
        from enhancements.competitive_selector import original_paper_reward
        self.assertEqual(original_paper_reward(1, 1, 0), 3.)
        self.assertEqual(original_paper_reward(1, 0, 0), 2.)
        self.assertEqual(original_paper_reward(0, 1, 0), -2.)
        self.assertEqual(original_paper_reward(0, 0, 0), 0.)

    def test_actor_max_gradient_is_separate_from_diagnostic_regression(self):
        from enhancements.competitive_selector import CompetitiveSelector
        from enhancements.minimax_query import maximize_step
        head = CompetitiveSelector(dict(mean=[0.]*22, scale=[1.]*22))
        state = torch.ones(2, 22)
        actor = list(head.actor.parameters())
        value = list(head.value.parameters())
        logits, estimate = head(state)
        objective = -(logits.sigmoid()*torch.ones(2)).mean()
        gradients = torch.autograd.grad(objective, actor+value, allow_unused=True)
        self.assertTrue(all(g is None for g in gradients[len(actor):]))
        optimizer = torch.optim.Adam(head.parameters(), lr=.01)
        before = float(head(state)[0].detach().mean())
        maximize_step(head, optimizer, state, torch.ones(2))
        self.assertGreater(float(head(state)[0].detach().mean()), before)

    def test_logits_stay_bounded_and_normalizer_is_checkpointed(self):
        from enhancements.competitive_selector import CompetitiveSelector
        spec = dict(mean=[1.]*22, scale=[2.]*22)
        head = CompetitiveSelector(spec)
        for parameter in head.actor.parameters(): parameter.data.fill_(100.)
        logits, _ = head(torch.ones(3, 22)*1e10)
        self.assertTrue((logits.abs() <= 4).all())
        clone = CompetitiveSelector(dict(mean=[0.]*22, scale=[1.]*22))
        clone.load_state_dict(head.state_dict())
        torch.testing.assert_close(head(torch.zeros(1, 22))[0], clone(torch.zeros(1, 22))[0])

    def stream_fixture(self):
        from enhancements.minimax_query import context_min_step, counterfactual_step, CounterfactualWorkspace, query_target_allowed
        from enhancements.competitive_selector import CompetitiveSelector
        path = Path(__file__).resolve().parents[1]/'amcal_minimax_paironly.py'
        names = ('feature_key', 'feature_keys', 'balanced_draw', 'validate_trace', 'stream', 'prediction_route_scores')
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
        class History:
            def __init__(self, *args, **kwargs): pass
            def observe_paid(self, *args): pass
            def updated(self, *args): pass
        class Memory:
            def __init__(self, anchors, seed, *args):
                self.rng = np.random.default_rng(seed); self.anchors = anchors; self.rows = []
            def draw(self, *args, **kwargs):
                at = self.rng.choice(len(self.anchors[1]), 2, replace=False)
                return self.rows[-4:], (self.anchors[0][at], torch.tensor(self.anchors[1][at]))
            def add(self, i, x, y): self.rows.append((x.clone(), torch.tensor([y])))
        class Executor:
            def __init__(self, cnn, contexts): self.cnn, self.context = cnn, contexts[0]
            def outputs(self, x, z):
                return torch.stack([z, self.cnn(self.context(x))], 1), x.new_zeros(len(x), 2)
        cnn = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        context = torch.nn.Linear(2, 2).eval()
        query = CompetitiveSelector(dict(mean=[0.]*22, scale=[1.]*22))
        with torch.no_grad(): query.actor[-1].bias.fill_(.1)
        agent = SimpleNamespace(policy=SimpleNamespace(future=torch.nn.Linear(1, 1), update_value=torch.nn.Linear(1, 1)),
                                versions=np.array([0]), clock=0)
        def cache(cnn, x):
            with torch.no_grad(): return cnn(x), x
        def metrics(y, predictions, classes):
            return dict(AC=100*float(np.mean(np.asarray(y) == predictions)), PR=0., RC=0., F1=0.)
        def endpoint(a, cnn, context, agent, domain, *args):
            with torch.no_grad(): return metrics(domain[1], cnn(context(domain[0])).argmax(1).numpy(), 2)
        ns = dict(torch=torch, np=np, math=math, deepcopy=deepcopy, copy=copy,
            DeviceReliability=History, Replay=Memory, CounterfactualWorkspace=CounterfactualWorkspace,
            BankExecutor=Executor, build_state=lambda *args:torch.zeros(1, 19),
            base=SimpleNamespace(legacy=SimpleNamespace(core=object(), base_cache=cache)),
            endpoint=endpoint, metrics=metrics, query_target_allowed=query_target_allowed,
            context_min_step=context_min_step, counterfactual_step=counterfactual_step,
            normalized_entropy=lambda logits: np.zeros(logits.shape[:-1]),
            uniform_query_at=lambda pos,n,cap: (pos+1)*cap//n > pos*cap//n)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        x = torch.randn(30, 2)
        data = x, np.array([0, 1]*15), np.arange(30)
        anchors = torch.randn(8, 2)+10, np.array([0, 1]*4), np.arange(100, 108)
        reference = torch.randn(8, 2)+20, np.array([0, 1]*4), np.arange(200, 208)
        a = SimpleNamespace(classes=['0', '1'], config=object(), budget_fraction=.2,
            eval_query_schedule='uniform', train_query_schedule='uniform', eval_routing='policy',
            budget_actor_mode=True, competitive_selector=True, collect_query_targets_only=True,
            competitive_scales=dict(gap=1., weighted_gain=1., weighted_harm=1.),
            context_lr=.001, query_lr=.001, replay_weight=.1, horizon=2, reference_size=4,
            gain_weight=1., forgetting_weight=1., cost=.05, progress_every=100)
        return ns['stream'], a, (cnn, context, agent, query), data, anchors, reference

    def test_train_probes_do_not_change_real_update_or_sampler_compared_to_evaluation(self):
        stream, a, live, data, anchors, reference = self.stream_fixture()
        original = deepcopy(live)
        with contextlib.redirect_stdout(io.StringIO()):
            a.eval_query_schedule = 'policy'
            trained, result, trace = stream(a, deepcopy(original), data, anchors, reference, reference, 42,
                training=True, target_endpoints=False)
            a.eval_query_schedule = 'uniform'
            tested, evaluated, other = stream(a, deepcopy(original), data, anchors, None, reference, 42,
                training=False, target_endpoints=False)
        self.assertEqual(result['labels'], 6)
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in other])
        for key, value in trained[1].state_dict().items():
            torch.testing.assert_close(value, tested[1].state_dict()[key], rtol=0, atol=0)
        for key, value in original[0].state_dict().items():
            torch.testing.assert_close(value, trained[0].state_dict()[key], rtol=0, atol=0)

    def test_unpaid_truth_cannot_change_actor_decisions_or_context(self):
        stream, a, live, data, anchors, reference = self.stream_fixture()
        a.eval_query_schedule = 'policy'
        with torch.no_grad(): live[3].actor[-1].bias.fill_(-1.)
        changed = data[0], 1-data[1], data[2]
        with contextlib.redirect_stdout(io.StringIO()):
            first, result, trace = stream(a, deepcopy(live), data, anchors, None, reference, 42, target_endpoints=False)
            second, _, other = stream(a, deepcopy(live), changed, anchors, None, reference, 42, target_endpoints=False)
        self.assertEqual(result['labels'], 0)
        self.assertEqual([r['query_probability'] for r in trace], [r['query_probability'] for r in other])
        self.assertTrue(all(r['route'] == 1 for r in trace))
        for key, value in first[1].state_dict().items(): torch.testing.assert_close(value, second[1].state_dict()[key])

    def test_paper_reward_training_uses_no_future_probe_and_keeps_same_paid_updates(self):
        from enhancements.competitive_selector import original_paper_reward
        stream, a, live, data, anchors, reference = self.stream_fixture()
        a.original_paper_reward = True
        def forbidden(*args, **kwargs): raise AssertionError('Future reward path was called')
        stream.__globals__['counterfactual_step'] = forbidden
        stream.__globals__['query_target_allowed'] = forbidden
        a.eval_query_schedule = 'policy'
        with contextlib.redirect_stdout(io.StringIO()):
            trained, result, trace = stream(a, deepcopy(live), data, anchors, reference, reference,
                42, training=True, target_endpoints=False)
            a.eval_query_schedule = 'uniform'
            tested, _, other = stream(a, deepcopy(live), data, anchors, None, reference,
                42, training=False, target_endpoints=False)
        self.assertEqual(result['labels'], 6)
        measured = [row for row in trace if row['score'] is not None]
        self.assertGreater(len(measured), 0)
        for row in measured:
            self.assertEqual(row['score'], original_paper_reward(row['frozen_prediction'],
                row['context_prediction'], row['truth']))
            self.assertIsNone(row['gain']); self.assertIsNone(row['harm'])
        self.assertTrue(all(row['score'] is None for row in other))
        self.assertEqual([r['prediction'] for r in trace], [r['prediction'] for r in other])
        for key, value in trained[1].state_dict().items():
            torch.testing.assert_close(value, tested[1].state_dict()[key], rtol=0, atol=0)

    def test_round_training_resume_and_selected_checkpoint_hash_with_tiny_network(self):
        from enhancements.competitive_selector import CompetitiveSelector
        from enhancements.minimax_query import maximize_step
        from enhancements.budget_actor_math import fit_signal_scales
        path = Path(__file__).resolve().parents[1]/'amcal_competitive_selector.py'
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                 and n.name in ('save_model', 'train')]
        def write_json(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        def normalizer(states):
            return dict(mean=states.mean(0).tolist(), scale=states.std(0).clamp_min(.01).tolist())
        ns = dict(torch=torch, np=np, deepcopy=deepcopy, json=json,
            CompetitiveSelector=CompetitiveSelector, maximize_step=maximize_step,
            fit_signal_scales=fit_signal_scales, LEVELS=(0, 1, 3),
            old=SimpleNamespace(normalizer=normalizer),
            pilot=SimpleNamespace(write_json=write_json,
                digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest(),
                base=SimpleNamespace(legacy=SimpleNamespace(core=SimpleNamespace(
                    device='cpu', set_seed=torch.manual_seed)))))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        states = torch.randn(32, 22)
        rows = [dict(score=float(i-16), gap=float(i)/10, weighted_gain=float(i%5)/10,
                     weighted_harm=float(i%3)/10) for i in range(32)]
        class Experiment:
            def __init__(self): self.calls = []; self.fail = True
            def collect(self, state, schedule, directory, round_index):
                self.calls.append(round_index)
                return states, torch.linspace(-1, 1, 32), rows
            def evaluate(self, state, split, levels, root):
                if self.fail and root.name.endswith('02'): raise RuntimeError('Simulated interruption')
                return [dict(method='Actor', AC=70., forgetting=0., labels_percent=10.)]*3
        with tempfile.TemporaryDirectory() as temporary:
            a = SimpleNamespace(output=Path(temporary)/'run', seed=42, actor_lr=.001,
                rounds=2, steps=2, classes=['0', '1'])
            experiment = Experiment()
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'interruption'): ns['train'](a, experiment, dict(version=1))
                experiment.fail = False
                ns['train'](a, experiment, dict(version=1))
                calls = list(experiment.calls)
                ns['train'](a, experiment, dict(version=1))
            self.assertEqual(calls, [0, 1, 2, 2])
            self.assertEqual(experiment.calls, calls)
            saved = json.loads((a.output/'training_complete.json').read_text())
            self.assertEqual(saved['sha256'], hashlib.sha256((a.output/'Models'/'selector.pth').read_bytes()).hexdigest())
            self.assertEqual(len(json.loads((a.output/'training.json').read_text())['history']), 2)


if __name__ == '__main__': unittest.main()
