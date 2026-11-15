"""Pure standard-library checks for final TEST-control dispatch."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1]/'amcal_final_multiseed_controls.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_control')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
run_control = namespace['run_control']


class FinalMultiseedControlTests(unittest.TestCase):
    def test_random_is_immediate_nontraining_on_fixed_evaluation_seed(self):
        calls = []
        def episode(*args, **kwargs):
            calls.append((args, kwargs))
            return 'metrics', 'trace'
        base = SimpleNamespace(run_episode=episode)
        def no_window(*args):
            self.fail('Random must not use window ranking')
        result = run_control(base, no_window, 'exp', 'learner', 'domain', 'reference', 'Random')
        self.assertEqual(result, ('metrics', 'trace'))
        self.assertEqual(calls[0][0], ('exp', 'learner', 'domain', 'reference', 42))
        self.assertEqual(calls[0][1], dict(training=False, mode='Random'))

    def test_uncertainty_uses_only_causal_window_ranking(self):
        calls = []
        def no_episode(*args, **kwargs):
            self.fail('Uncertainty must not use thresholded legacy runner')
        def window(*args):
            calls.append(args)
            return 'metrics', 'trace'
        base = SimpleNamespace(run_episode=no_episode)
        run_control(base, window, 'exp', 'learner', 'domain', 'reference', 'Uncertainty')
        self.assertEqual(calls, [('exp', 'learner', 'domain', 'reference',
                                  42, 'UncertaintyWindow', 5)])

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            run_control(None, None, None, None, None, None, 'unknown')


if __name__ == '__main__':
    unittest.main()
