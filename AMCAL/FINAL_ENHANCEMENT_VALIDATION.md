# Enhancement validation record

Baseline code commit: `44b907e0da1238d7d75e1fdfc154f4e2fbd8b274`.
Validation date: 2026-10-21. Runtime: Python 3.12, PyTorch 2.14.1+cpu.

```text
python -m unittest discover -s AMCAL/tests -p 'test_*.py' -v
Ran 40 tests
OK
```

The scoped local checkout contains the existing structured-bank, balanced
reference and stream-timing suites plus the new enhancement suite. This command
describes those four suites; other unrelated repository test files were not
materialized or executed for this change.

The new suite executes real Transformer Contexts, a real frozen CNN and a
source-disjoint synthetic CSV training/save/reload/evaluation/persistent-audit
workflow. It verifies:

- no change to actions, learned weights or Context weights when only unqueried
  labels change;
- immutable CNN parameters/buffers and independently updated Context adapters;
- exactly matched adaptation trajectories under fixed feedback/destinations;
- hard caps including a zero-label budget;
- bounded memory, historical rare-class retention and unique references;
- gradient separation, ranking direction and paired-update credit;
- on-device reliability agreement with the original formulas;
- current-weight vectorized inference, including actual compact Transformers;
- shared-trunk execution once per bank and no trunk execution for a frozen-only
  sparse prediction;
- missing-route/null-Oracle semantics with zero hidden Context executions;
- observer isolation and independent persistent audit IDs;
- saved-config consistency and protection against overwriting results.

The old structured-bank test fixture was updated to include current CLI defaults
for Context alpha, balanced references and decoupled queries. The original model,
Context and structured-bank implementation files remain byte-for-byte identical
to the inspected baseline.

No full-dataset enhancement training, GPU benchmark or accuracy comparison was
performed. Use the launcher and compare separate profiles before claiming a
performance improvement.
