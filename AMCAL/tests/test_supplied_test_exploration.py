"""Pure standard-library contracts for the TEST exploration adapter."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'amcal_supplied_test_exploration.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exploration_runner')
namespace = {}
exec(compile(ast.Module(body=[helper], type_ignores=[]), str(SOURCE), 'exec'), namespace)
exploration_runner = namespace['exploration_runner']


class SuppliedExplorationTests(unittest.TestCase):
    def agent(self):
        calls = []
        def choose(state, legal, epsilon, rng):
            calls.append((legal, epsilon))
            return (int(legal), [0., 1.])
        return SimpleNamespace(choose=choose), calls

    def test_eval_uses_configured_epsilon_and_records_it(self):
        learner, calls = self.agent()
        original_choose = learner.choose
        def episode(exp, agent, *args, **kwargs):
            action, _ = agent.choose(None, True, 0., None)
            return {'td_updates': 0}, [{'action': action}]
        result, trace = exploration_runner(episode, .05)(
            None, learner, 'domain', 'reference', 42, training=False)
        self.assertEqual(calls, [(True, .05)])
        self.assertIs(learner.choose, original_choose)
        self.assertEqual(result['evaluation_epsilon'], .05)
        self.assertEqual(trace[0]['epsilon'], .05)

    def test_exhausted_budget_remains_illegal(self):
        learner, calls = self.agent()
        def episode(exp, agent, *args, **kwargs):
            action, _ = agent.choose(None, False, 0., None)
            return {}, [{'action': action}]
        _, trace = exploration_runner(episode, .05)(None, learner)
        self.assertEqual(trace[0]['action'], 0)
        self.assertEqual(calls, [(False, .05)])

    def test_training_is_rejected(self):
        learner, _ = self.agent()
        wrapped = exploration_runner(lambda *a, **k: self.fail('Runner must not execute'), .05)
        with self.assertRaises(ValueError):
            wrapped(None, learner, training=True)
        with self.assertRaises(ValueError):
            wrapped(None, learner, 'domain', 'reference', 42, True)

    def test_original_method_restored_after_error(self):
        learner, _ = self.agent()
        original_choose = learner.choose
        def episode(*args, **kwargs):
            raise RuntimeError('Expected failure')
        with self.assertRaises(RuntimeError):
            exploration_runner(episode, .05)(None, learner)
        self.assertIs(learner.choose, original_choose)


if __name__ == '__main__':
    unittest.main()
