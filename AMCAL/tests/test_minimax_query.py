"""CPU unit checks only: no datasets, downloads or model training runs."""
import importlib.util
from pathlib import Path
import unittest
from copy import deepcopy
import ast
import types
import contextlib
import io
import math

try:
    import numpy as np
except ImportError:
    np = None

helper_source = Path(__file__).resolve().parents[1]/'enhancements'/'minimax_query.py'
gate_ast = next(node for node in ast.parse(helper_source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'query_target_allowed')
gate_namespace = {}
exec(compile(ast.Module(body=[gate_ast], type_ignores=[]), str(helper_source), 'exec'), gate_namespace)
runner_source = Path(__file__).resolve().parents[1]/'amcal_minimax_paironly.py'
trace_ast = next(node for node in ast.parse(runner_source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'validate_trace')
exec(compile(ast.Module(body=[trace_ast], type_ignores=[]), str(runner_source), 'exec'), gate_namespace)
uniform_ast = next(node for node in ast.parse(runner_source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == 'uniform_query_at')
exec(compile(ast.Module(body=[uniform_ast], type_ignores=[]), str(runner_source), 'exec'), gate_namespace)


class UniformScheduleTests(unittest.TestCase):
    def test_exact_cap_and_even_prefix_consumption(self):
        schedule = gate_namespace['uniform_query_at']
        for count in range(1, 101):
            for cap in range(count+1):
                used = 0
                for pos in range(count):
                    used += int(schedule(pos, count, cap))
                    self.assertLessEqual(abs(used-(pos+1)*cap/count), .5+1e-9)
                self.assertEqual(used, cap)
        self.assertEqual([i for i in range(10) if schedule(i, 10, 2)], [2, 7])

    def test_invalid_schedule_arguments_are_rejected(self):
        for args in [(0, 0, 0), (-1, 10, 2), (10, 10, 2), (0, 10, 11), (0, 10, -1)]:
            with self.assertRaises(ValueError): gate_namespace['uniform_query_at'](*args)


class BudgetContractTests(unittest.TestCase):
    def test_exhausted_budget_never_creates_training_targets(self):
        gate = gate_namespace['query_target_allowed']
        self.assertFalse(gate(True, 2, 2, 0, 10, 2))
        self.assertFalse(gate(True, 0, 0, 0, 10, 2))
        self.assertTrue(gate(True, 1, 2, 0, 10, 2))

    def test_test_and_validation_never_create_future_targets(self):
        self.assertFalse(gate_namespace['query_target_allowed'](False, 0, 2, 0, 10, 2))

    def test_full_future_block_is_required(self):
        gate = gate_namespace['query_target_allowed']
        self.assertTrue(gate(True, 0, 2, 7, 10, 2))
        self.assertFalse(gate(True, 0, 2, 8, 10, 2))


class TraceContractTests(unittest.TestCase):
    def row(self, step, queried, used):
        return dict(step=step, route=0, queried=queried, action=2 if queried else 0,
            labels_used=used, frozen_route_score=1., context_route_score=0.,
            frozen_prediction=1, context_prediction=0, prediction=1)

    def test_legal_pre_feedback_stream(self):
        gate_namespace['validate_trace']([self.row(1, True, 1), self.row(2, False, 1)], 1)

    def test_illegal_feedback_accounting_is_rejected(self):
        for rows in ([self.row(1, True, 1), self.row(2, True, 2)], [self.row(1, False, 1)]):
            with self.assertRaises(RuntimeError): gate_namespace['validate_trace'](rows, 1)

    def test_wrong_route_or_post_feedback_prediction_is_rejected(self):
        wrong_prediction = self.row(1, True, 1); wrong_prediction['prediction'] = 0
        wrong_route = self.row(1, True, 1); wrong_route['context_route_score'] = 2.
        for row in (wrong_prediction, wrong_route):
            with self.assertRaises(RuntimeError): gate_namespace['validate_trace']([row], 1)

@unittest.skipIf(np is None, 'Requires the existing NumPy dependency')
class ProbePartitionTests(unittest.TestCase):
    def functions(self):
        names = ('balanced_draw', 'reference_partition', 'subset', 'pilot_indices')
        nodes = [n for n in ast.parse(runner_source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name in names]
        ns = dict(np=np, torch=types.SimpleNamespace(as_tensor=lambda data, **kwargs: np.asarray(data)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner_source), 'exec'), ns)
        return ns

    def test_balanced_probe_is_unique_and_covers_available_classes(self):
        draw = self.functions()['balanced_draw']
        labels = np.array([0]+[1]*3+[2]*6)
        indices = draw(labels, 6, np.random.default_rng(42))
        self.assertEqual(len(indices), 6)
        self.assertEqual(len(set(indices)), 6)
        self.assertEqual(set(labels[indices]), {0, 1, 2})
        self.assertEqual(len(draw(labels, 100, np.random.default_rng(42))), len(labels))
        self.assertEqual(draw(labels, 0, np.random.default_rng(42)), [])

    def test_pilot_sampling_is_fixed_unique_bounded_and_keeps_classes(self):
        sample = self.functions()['pilot_indices']
        labels = np.repeat(np.arange(6), [90,30,15,9,6,3])
        first, repeated = sample(labels, 60, 42), sample(labels, 60, 42)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(first), 60)
        self.assertEqual(len(set(first)), 60)
        self.assertTrue(all((labels[first] == cls).sum() >= 3 for cls in range(6)))
        self.assertTrue(np.all(first[:-1] < first[1:]))
        np.testing.assert_array_equal(sample(labels, 0, 42), np.arange(len(labels)))
        np.testing.assert_array_equal(sample(labels, 999, 42), np.arange(len(labels)))
        with self.assertRaises(ValueError): sample(labels, 10, 42)
        with self.assertRaises(ValueError): sample(labels, -1, 42)

    def test_reference_ids_are_stratified_and_excluded_from_train(self):
        class Array(np.ndarray):
            @property
            def device(self): return 'cpu'
        x = np.arange(120).reshape(60, 2).view(Array)
        labels = np.repeat(np.arange(3), 20)
        ids = np.arange(100, 160)
        train, held, reserved = self.functions()['reference_partition']((x, labels, ids), 42)
        self.assertEqual(len(held[1]), 6)
        self.assertEqual(set(held[1]), {0, 1, 2})
        self.assertEqual(set(map(int, held[2])), reserved)
        self.assertFalse(set(map(int, train[2])) & reserved)
        self.assertEqual(set(train[2]) | reserved, set(ids))


try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    torch.set_num_threads(1)
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

    def test_batch5_has_equal_weights_and_separate_protected_replay(self):
        context = nn.Linear(2, 2, bias=False).eval()
        expected = deepcopy(context)
        cnn = nn.Linear(2, 2, bias=False).eval().requires_grad_(False)
        x = torch.tensor([[1., .2], [.3, -.8], [-1., .7], [.8, -.2], [-.4, .9]])
        y = torch.tensor([1, 0, 0, 1, 0])
        replay = torch.tensor([[.1, -1.]]), torch.tensor([0])
        actual_optimizer = torch.optim.SGD(context.parameters(), lr=.01)
        expected_optimizer = torch.optim.SGD(expected.parameters(), lr=.01)
        expected_optimizer.zero_grad()
        expected_gap = torch.nn.functional.cross_entropy(cnn(expected(x)), y) - torch.nn.functional.cross_entropy(cnn(x), y).detach()
        loss = expected_gap + .1*torch.nn.functional.cross_entropy(cnn(expected(replay[0])), replay[1])
        loss.backward()
        nn.utils.clip_grad_norm_(expected.parameters(), 1.)
        expected_optimizer.step()
        gap, _ = module.context_min_step(context, actual_optimizer, cnn, (x[:1], y[:1]),
            [(x[i:i+1], y[i:i+1]) for i in range(1, 5)], replay, .1)
        self.assertAlmostEqual(gap, float(expected_gap.detach()), places=6)
        for actual, wanted in zip(context.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted)

    def test_reused_shadow_matches_fresh_branch_and_survives_commit(self):
        context = nn.Linear(2, 2, bias=False).eval()
        cnn = nn.Linear(2, 2, bias=False).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(context.parameters(), lr=.01)
        current = torch.tensor([[1., .2]]), torch.tensor([1])
        future = torch.tensor([[.3, -.8]]), torch.tensor([0])
        reference = torch.tensor([[-1., .7]]), torch.tensor([0])
        args = (cnn, current, [], None, future, reference, .1, 1., 1., .05)
        workspace = module.CounterfactualWorkspace()
        for commit in (False, True, False, True):
            wanted, _, wanted_target = module.counterfactual_step(context, optimizer, *args)
            live = deepcopy(context.state_dict())
            actual, actual_optimizer, actual_target = module.counterfactual_step(context, optimizer, *args, workspace=workspace)
            for key, value in live.items(): torch.testing.assert_close(context.state_dict()[key], value)
            for actual_p, wanted_p in zip(actual.parameters(), wanted.parameters()):
                torch.testing.assert_close(actual_p, wanted_p)
            self.assertAlmostEqual(actual_target['score'], wanted_target['score'])
            live_ptrs = {p.data_ptr() for p in context.parameters()}
            self.assertFalse(live_ptrs & {p.data_ptr() for p in actual.parameters()})
            if commit:
                old = context, optimizer
                context, optimizer = actual, actual_optimizer
                workspace.recycle(*old)

    @unittest.skipIf(np is None, 'Requires NumPy')
    def test_replay_excludes_probe_ids_features_and_aliases(self):
        nodes = [n for n in ast.parse(runner_source.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and n.name in ('Replay', 'feature_key', 'feature_keys', 'balanced_draw')]
        class Storage:
            def __init__(self, anchors, *args):
                x, y = anchors
                self.source = [(x[i:i+1], int(y[i]), x[i:i+1].numpy().tobytes()) for i in range(len(y))]
                self.recent, self.historical = [], [[]]
            def add(self, x, y):
                self.recent.append((x.clone(), int(y[0]), x.numpy().tobytes()))
        ns = dict(np=np, torch=torch, ProtectedMemory=Storage)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner_source), 'exec'), ns)
        x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        x[6] = x[2]  # One feature hash has two different source IDs.
        anchors = x[:8], np.arange(8) % 2, np.arange(100, 108)
        replay = ns['Replay'](anchors, 42, types.SimpleNamespace(memory_replay_size=8), 2)
        for i in range(4): replay.add(200+i, x[8+i:9+i], i % 2)
        key = ns['feature_key']
        paid, protected = replay.draw(100, x[8:9], future_ids=[106, 202],
            reference_keys=[key(x[1:2])], current_key=key(x[8:9]), future_keys=[key(x[3:4])])
        blocked = {key(x[i:i+1]) for i in (0, 1, 2, 3, 6, 8, 10)}
        paid_keys = {key(row[0]) for row in paid}
        protected_keys = {key(row[None]) for row in protected[0]}
        self.assertFalse(paid_keys & blocked)
        self.assertFalse(protected_keys & blocked)
        self.assertFalse(paid_keys & protected_keys)
        self.assertEqual(len(paid_keys), 2)
        self.assertEqual(len(protected_keys), 3)


@unittest.skipIf(torch is None or np is None, 'Requires the existing PyTorch/NumPy experiment environment')
class StreamContractTests(unittest.TestCase):
    """Exercise the real stream with tiny networks and in-memory infrastructure."""
    def fixture(self, training=False):
        torch.manual_seed(19)
        calls = []
        class TinyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.routing = nn.Linear(6, 2)
                self.future = nn.Linear(1, 1)
                self.update_value = nn.Linear(1, 1)
                with torch.no_grad():
                    self.routing.weight.zero_(); self.routing.bias.copy_(torch.tensor([1., 0.]))
            def route_logits(self, states): return self.routing(states)
        class TinyAgent:
            def __init__(self):
                self.policy = TinyPolicy()
                self.clock = 0
                self.versions = np.zeros(1, dtype=np.int64)
                self.feedback = []
            def observe_routes(self, state, correct, **kwargs): self.feedback.append(correct.clone())
        class TinyHistory:
            def __init__(self, *args, **kwargs): pass
            def observe_paid(self, *args): pass
            def updated(self, *args): pass
        class TinyMemory:
            def __init__(self, *args): pass
            def draw(self, *args, **kwargs): return [], None
            def add(self, *args): pass
        class TinyExecutor:
            def __init__(self, cnn, bank): self.cnn, self.context = cnn, bank[0]
            def outputs(self, x, z):
                return torch.stack([z, self.cnn(self.context(x))], 1), torch.zeros(len(x), 2)
        def cache(cnn, x):
            with torch.no_grad(): return cnn(x), x
        def state(outputs, embedding, history, used, cap, position, count, gate):
            return torch.cat([outputs[0].flatten(1), embedding.new_tensor(
                [[(cap-used)/max(cap,1), position/max(count,1)]]).expand(len(embedding), -1)], 1)
        def metric(y, predicted, classes):
            ac = 100*float((np.asarray(y) == predicted).mean())
            return dict(AC=ac, PR=ac, RC=ac, F1=ac)
        def endpoint(*args): return dict(AC=0., PR=0., RC=0., F1=0.)
        def counterfactual(*args, **kwargs):
            if not training: raise AssertionError('TEST requested future labels.')
            calls.append(1)
            return module.counterfactual_step(*args, **kwargs)
        path = Path(__file__).resolve().parents[1]/'amcal_minimax_paironly.py'
        functions = [node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in ('stream', 'feature_keys', 'feature_key', 'balanced_draw', 'validate_trace', 'uniform_query_at')]
        namespace = dict(torch=torch, np=np, deepcopy=deepcopy, math=math,
            base=types.SimpleNamespace(legacy=types.SimpleNamespace(core=object(), base_cache=cache)),
            DeviceReliability=TinyHistory, Replay=TinyMemory, BankExecutor=TinyExecutor,
            build_state=state, endpoint=endpoint, metrics=metric,
            CounterfactualWorkspace=module.CounterfactualWorkspace,
            query_target_allowed=module.query_target_allowed, maximize_step=module.maximize_step,
            context_min_step=module.context_min_step, counterfactual_step=counterfactual)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        cnn = nn.Linear(2, 2).eval().requires_grad_(False)
        context = nn.Linear(2, 2).eval()
        agent = TinyAgent()
        query = module.QueryMaximizer(6)
        with torch.no_grad(): query.actor[-1].bias.fill_(80.)
        x = torch.randn(10, 2); y = np.array([0, 1]*5); ids = np.arange(10)
        domain = x, y, ids
        reference = torch.randn(4, 2), np.array([0, 1, 0, 1]), np.arange(100, 104)
        a = types.SimpleNamespace(budget_fraction=.2, config=object(), classes=['a','b'],
            context_lr=.001, query_lr=.001, horizon=2, reference_size=2,
            replay_weight=.1, gain_weight=1., forgetting_weight=1., cost=.05, progress_every=100)
        return namespace['stream'], a, (cnn, context, agent, query), domain, reference, calls

    def test_test_never_probes_future_and_respects_hard_cap(self):
        run, a, runtime, domain, reference, calls = self.fixture()
        initial = {k: v.clone() for k, v in runtime[3].state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            updated, result, trace = run(a, runtime, domain, domain, None, domain, 42)
        self.assertEqual(calls, [])
        self.assertEqual(result['labels'], 2)
        self.assertEqual(sum(row['queried'] for row in trace), 2)
        self.assertEqual(len(updated[2].feedback), 2)
        for row in trace[2:]:
            self.assertFalse(row['queried'])
            self.assertEqual(row['query_probability'], 0.)
        for key, value in initial.items(): torch.testing.assert_close(updated[3].state_dict()[key], value)

    def test_uniform_evaluation_overrides_collapsed_actor_causally(self):
        run, a, runtime, domain, reference, calls = self.fixture()
        a.eval_query_schedule = 'uniform'
        with torch.no_grad(): runtime[3].actor[-1].bias.fill_(-80.)
        query_before = deepcopy(runtime[3].state_dict())
        context_before = deepcopy(runtime[1].state_dict())
        twin = deepcopy(runtime)
        flipped = domain[0], 1-domain[1], domain[2]
        with contextlib.redirect_stdout(io.StringIO()):
            updated, result, trace = run(a, runtime, domain, domain, None, domain, 42)
            _, _, altered = run(a, twin, flipped, domain, None, domain, 42)
        self.assertEqual(calls, [])
        self.assertEqual(result['labels'], 2)
        self.assertEqual([r['step'] for r in trace if r['queried']], [3, 8])
        self.assertEqual([r['queried'] for r in trace], [r['queried'] for r in altered])
        # Prediction and route at the first feedback event precede its label.
        self.assertEqual(trace[2]['prediction'], altered[2]['prediction'])
        self.assertEqual(trace[2]['route'], altered[2]['route'])
        self.assertEqual(len(updated[2].feedback), 2)
        self.assertTrue(any(not torch.equal(v, updated[1].state_dict()[k]) for k,v in context_before.items()))
        for k,v in query_before.items(): torch.testing.assert_close(v, updated[3].state_dict()[k])

    def test_uniform_control_cannot_enter_training(self):
        run, a, runtime, domain, reference, _ = self.fixture(training=True)
        a.eval_query_schedule = 'uniform'
        with self.assertRaises(ValueError):
            run(a, runtime, domain, domain, reference, domain, 42, training=True)

    def test_uniform_train_collection_does_not_force_val_requests(self):
        run, a, runtime, domain, reference, calls = self.fixture(training=True)
        a.train_query_schedule = 'uniform'
        with torch.no_grad(): runtime[3].actor[-1].bias.fill_(-80.)
        initial_value = deepcopy(runtime[3].value.state_dict())
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            trained, result, trace = run(a, runtime, domain, domain, reference, domain, 42,
                training=True, training_state=state)
        self.assertEqual(result['labels'], 2)
        self.assertEqual([r['step'] for r in trace if r['queried']], [3, 8])
        self.assertEqual(len(calls), 8)
        self.assertTrue(any(not torch.equal(v, trained[3].value.state_dict()[k]) for k,v in initial_value.items()))
        # VAL gets a fresh independent runtime snapshot, and uses the Actor.
        validation = deepcopy(trained)
        with torch.no_grad(): validation[3].actor[-1].weight.zero_(); validation[3].actor[-1].bias.fill_(-80.)
        calls_before = len(calls)
        with contextlib.redirect_stdout(io.StringIO()):
            _, val, val_trace = run(a, validation, domain, domain, None, domain, 42)
        self.assertEqual(val['labels'], 0)
        self.assertFalse(any(r['queried'] for r in val_trace))
        self.assertEqual(len(calls), calls_before)

    def test_training_stops_forks_at_cap_and_preserves_adam_between_domains(self):
        run, a, runtime, domain, reference, calls = self.fixture(training=True)
        state = {}
        with contextlib.redirect_stdout(io.StringIO()):
            updated, _, trace = run(a, runtime, domain, domain, reference, domain, 42,
                training=True, training_state=state)
            query_optimizer = state['query_optimizer']
            context_optimizer = state['context_optimizer']
            step = next(iter(context_optimizer.state.values()))['step'].item()
            self.assertEqual(len(calls), 2)
            updated, _, _ = run(a, updated, domain, domain, reference, domain, 43,
                training=True, training_state=state)
        self.assertEqual(len(calls), 4)
        self.assertIs(state['query_optimizer'], query_optimizer)
        later_step = next(iter(state['context_optimizer'].state.values()))['step'].item()
        self.assertGreater(later_step, step)
        self.assertEqual(sum(row['queried'] for row in trace), 2)

    def test_current_prediction_does_not_depend_on_current_label(self):
        run, a, runtime, domain, reference, _ = self.fixture()
        twin = deepcopy(runtime)
        flipped = domain[0], 1-domain[1], domain[2]
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, first = run(a, runtime, domain, domain, None, domain, 42)
            _, _, second = run(a, twin, flipped, domain, None, domain, 42)
        self.assertEqual(first[0]['source_row'], second[0]['source_row'])
        self.assertEqual(first[0]['prediction'], second[0]['prediction'])
        self.assertEqual(first[0]['route'], second[0]['route'])
        self.assertEqual(first[0]['queried'], second[0]['queried'])

    def test_logging_does_not_change_training_trajectory(self):
        run, a, runtime, domain, reference, _ = self.fixture(training=True)
        twin = deepcopy(runtime)
        class Sink:
            def __init__(self): self.rows = []
            def observe(self, state, record): self.rows.append(dict(record))
        sink = Sink()
        with contextlib.redirect_stdout(io.StringIO()):
            plain, _, first = run(a, runtime, domain, domain, reference, domain, 42, training=True)
            logged, _, second = run(a, twin, domain, domain, reference, domain, 42,
                training=True, diagnostic_sink=sink)
        self.assertEqual([r['queried'] for r in first], [r['queried'] for r in second])
        self.assertEqual(len(sink.rows), len(domain[1]))
        for before, after in ((plain[1], logged[1]), (plain[3], logged[3])):
            for key, value in before.state_dict().items(): torch.testing.assert_close(value, after.state_dict()[key])
        measured = [r for r in sink.rows if r['score'] is not None]
        self.assertTrue(all(r['query_legal'] for r in measured))
        for r in measured:
            self.assertAlmostEqual(r['score'], r['gap']+r['weighted_gain']-r['weighted_harm']-r['request_cost'])


if __name__ == '__main__': unittest.main()

