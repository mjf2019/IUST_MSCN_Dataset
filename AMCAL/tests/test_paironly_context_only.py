"""Small contracts for the original PairOnly Context-only feedback control."""
import ast
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'enhancements'/'stream.py'
NAMES = {'context_feedback_schedule', 'validate_context_only_prefix'}
nodes = [node for node in ast.parse(SOURCE.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name in NAMES]
namespace = dict(np=np)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class ContextOnlyContracts(unittest.TestCase):
    def test_frontloaded_schedule_is_the_exact_prefix(self):
        schedule = namespace['context_feedback_schedule']
        for n in (1, 5, 20, 886, 1073):
            for cap in (0, int(.2*n), n):
                self.assertEqual(schedule(n, cap, 42, True, True), set(range(cap)))

    def test_default_legacy_schedule_is_unchanged(self):
        schedule = namespace['context_feedback_schedule']
        old = set(np.random.default_rng(42+819003).choice(100,20,replace=False).tolist())
        self.assertEqual(schedule(100,20,42,True), old)
        self.assertIsNone(schedule(100,20,42,False))
        with self.assertRaises(ValueError): schedule(10,11,42,True)

    def trace(self):
        return [dict(route=1, prediction=0, predictions=[1,0], queried=(i<2), labels_used=min(i+1,2)) for i in range(10)]

    def test_live_contract_accepts_only_fixed_context_and_causal_prefix(self):
        check = namespace['validate_context_only_prefix']
        check(self.trace(), 2)
        for key, value in [('route',0), ('prediction',1), ('queried',False), ('labels_used',0)]:
            trace = self.trace(); trace[0][key] = value
            with self.assertRaises(RuntimeError): check(trace,2)
        trace = self.trace(); trace[-1]['queried'] = True
        with self.assertRaises(RuntimeError): check(trace,2)


if __name__ == '__main__': unittest.main()
