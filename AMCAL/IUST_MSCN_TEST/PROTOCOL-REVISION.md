# AMCAL protocol revision — IUST_MSCN pilot

Work branch: `experiment/amcal-protocol`. Canonical revision entry point:
`amcal_protocol.py`. The original notebook is retained as historical source;
its saved outputs are not revised evidence.

## Changes in this first patch

- Keep the original CNN, context Transformer/CNN and selector architectures.
- Group feature-identical original rows before a seeded 60/20/20 group split.
  Group percentages refer to groups, so exact row percentages may differ.
  Fit the scaler and label encoder on training only.
- Save fitted preprocessing, training class counts, feature order, split IDs,
  original labels and original CSV SHA-256 in
  `Models/protocol_preprocessing.joblib`. Every checkpoint records this
  artifact's hash. Old checkpoints without this metadata must not be reused.
- Deep-copy best weights and restore the best CNN even if training reaches
  its final epoch. Freeze CNN parameters and running statistics throughout
  context adaptation. Set context to eval mode for validation.
- Apply the same combined context objective offline and online:
  `0.8*context_ce + 0.2/(1 + original_ce/(context_ce+1e-8))`.
  The old offline maximum clamp is removed. CE-only context warm-up remains.
- Capture reward predictions before label-driven updates. Offline selector
  training uses fully supervised training labels; online learning only uses
  queried labels. Online action selection is greedy with dropout disabled;
  online replay starts empty.
- Score online predictions before querying labels and adapting. Count unique
  original source-row queries separately from context/selector optimizer steps.
  Check that the frozen classifier's tensors remain unchanged.
- Evaluate only the original held-out test IDs, including for attack variants.
  Require an explicit row map or an explicit assertion of preserved row order.
- Save per-sample traces, clean training metrics and evaluation JSON files.
  New checkpoints/results go in `protocol_runs/`, separate from historical
  `Models/` and `Results/`.

## Current limits

This is a first protocol patch, not a completed experimental revision.
No training, evaluation, syntax-check execution or tests were run by the
assistant; runtime validation is pending the author's execution.

Exact-feature grouping detects duplicates; it does not prove independence of
sessions, flows, times or near-duplicate/oversampled examples. Attack provenance
still needs verification. Matching row counts and labels cannot prove preserved
sample order. The audit reports these facts, not a leakage-free certification.

The historical selector has terminal replay targets (`done=1`), so its current
learning target is a contextual bandit. We retain that behavior explicitly;
we have not manufactured sequential transitions or resolved the manuscript's
DDQN/meta-learning/game-theory claims. The reward configuration is preserved,
including its highest reward when both classifiers are wrong.
These method questions must be resolved before final result tables.

A label budget is a maximum, not guaranteed consumption: the existing action,
disagreement and confidence-gap gates may use fewer labels. A zero query rate
is an important diagnostic, not a reason to tune on test performance.
Macro-F1 uses all training classes, including classes absent from a short pilot.
For publication comparisons, use this same split and protocol for every baseline.

## Run from repository root (PowerShell)

Dependencies: the existing PyTorch environment plus numpy, pandas,
scikit-learn and joblib. No additional model downloads are required.

First run the data audit and share `audit.json` or the printed output:

```powershell
python AMCAL/IUST_MSCN_TEST/amcal_protocol.py audit
```

The default output directory is
`AMCAL/IUST_MSCN_TEST/protocol_runs/seed42/`.
Audit checks the original CSV and enumerates attack CSVs with counts,
column-order matches and rowwise label matches.

Then a small pipeline pilot (these epochs are not publication settings):

```powershell
python AMCAL/IUST_MSCN_TEST/amcal_protocol.py train --base-epochs 5 --context-epochs 5 --pretrain-epochs 2
python AMCAL/IUST_MSCN_TEST/amcal_protocol.py evaluate --max-samples 200 --budget-fraction 0
python AMCAL/IUST_MSCN_TEST/amcal_protocol.py evaluate --max-samples 200 --budget-fraction 0.20
```

The clean evaluation does not need an attack mapping. Each evaluation reloads
the same frozen starting checkpoints; different evaluation calls are independent
runs, not a sequential forgetting experiment. Budget fraction is computed from
unique held-out source IDs in the evaluated prefix, not all rows in an attack CSV.

For an attack file, supply a CSV with exactly one zero-based `source_row` per
attack row, generated from attack provenance. Example (replace paths):

```powershell
python AMCAL/IUST_MSCN_TEST/amcal_protocol.py evaluate --attack-file AMCAL/IUST_MSCN_TEST/Dataset/FGSM/FGSM_eps_0.1.csv --row-map attack_source_rows.csv --max-samples 200 --budget-fraction 0.20
```

Only if the attack generation actually preserved every original row in the same
order may you replace `--row-map ...` with `--assume-row-aligned`. The runner
checks length and labels but records this as a user assertion, not provenance proof.

To retrain, choose a fresh directory with `--output`; the runner refuses to
overwrite an existing Models directory. Pass the same output to evaluation.
For each additional seed, specify both `--seed` and a fresh `--output`.

Share the pilot's `Results/training_clean_metrics.json`, evaluation
`*_metrics.json`, and `*_trace.csv` (especially queried rows and query counts).
Do not merge this branch into develop until the pilot has been checked.
