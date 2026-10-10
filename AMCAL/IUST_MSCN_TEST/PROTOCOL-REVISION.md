# AMCAL protocol revision v2 — IUST_MSCN pilot

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
- Score predictions before receiving labels. Separately compute the selector's
  reward and next state after updating the context on queried samples, matching
  Algorithm 1. These post-update outputs never replace scored predictions.
  Offline training uses supervised training labels; online replay starts empty.
- Align the selector with Eqs. (13)-(19): gamma=0.95, uncertainty from modulated
  inputs, zero reward for action 0, and non-terminal same-input local transitions.
  Sync the offline target every 15 epochs. During online streaming, sync after
  every 15 successful selector optimizer steps (a documented online convention).
  Greedy online selection uses eval-mode dropout; offline selection is epsilon-greedy.
- Query only when action=1 and unique-label budget remains. Remove the legacy
  disagreement/confidence-gap gates and hindsight replay, which were not defined
  in Algorithm 1. Log proposed and effective actions separately.
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

Protocol v2 restores bootstrapped DDQN targets, gamma=0.95 and the article's
same-input local transition: recompute the selected sample's modulated uncertainty
after context adaptation; for an unselected sample use a self-transition and
reward zero without accessing its label. No terminal boundary is introduced.
This reproduces the paper's continuing local-transition formulation; it does not
establish that its state is Markov or supply meta-learning/min-max guarantees.
Those theoretical questions remain open, and empirical stability must be checked.
Uncertainty is defined as scalar `1-max softmax` to retain the original selector's
feature_count+1 input width; the manuscript's vector-looking notation needs clarification.
Action-1 rewards retain Eq. (19), including highest reward when both predictions
are wrong. Action-0 reward is corrected to zero.
Protocol-v1 checkpoints are rejected; retrain into a fresh output directory.

A label budget is a maximum, not guaranteed consumption: the selector may
choose fewer samples. A zero query rate is a diagnostic, not a reason to tune
on test performance.
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
`AMCAL/IUST_MSCN_TEST/protocol_runs/seed42_v2/`.
Audit checks the original CSV and enumerates attack CSVs with counts,
column-order matches and rowwise label matches. It also compares attack-label
sequences with candidate historical 80/20 and two-stage 60/20/20 splits, with and
without stratification (using the chosen seed), and reports duplicate-group overlap
for each candidate. A matched label sequence is not proof of sample identity and
is never automatically exported or accepted as an attack source-row map.

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

## Budget handling correction

Zero-budget evaluation now freezes both context and selector. No online replay
transitions or optimizer steps are produced in that reference run.
After exhausting the label budget, both online learners also stop updating.
A proposed action 1 blocked by budget or a duplicate source ID is not converted
into a fictitious action-0 replay observation. Natural action-0 transitions may
still train the selector without labels while budget remains.
The last permitted queried transition may produce one selector update even when
that query consumes the final label. Existing v2 checkpoints remain compatible;
training does not need to be repeated for this evaluator-only correction.

The first max-samples rows are selected in original CSV order after held-out
filtering, whereas offline clean metrics use the held-out split's stored order.
Short-prefix and whole-test metrics therefore have different sample compositions.
Do not compare them as the same evaluation set.

## Online selector replay batch

Online evaluation uses a separate replay minibatch size, default 32, configurable
with `--online-selector-batch-size`. Offline selector training retains 256.
A 100-label pilot may exhaust its budget before reaching the old 256-experience
warm-up, preventing any selector optimizer update; this change allows updates
after 32 stored experiences. It does not guarantee improved accuracy or complete
budget consumption. For budgets/streams shorter than 32 stored transitions, lower
the option explicitly for a diagnostic pilot; do not tune it on final test results.

Metrics record the online minibatch size, final replay size, first/last query step
and first selector update step. Traces include replay size per sample.
Zero-budget runs still freeze both learners. Models remain compatible; retraining
is not required. The assistant has not run this evaluator or tests.
