# Two-action DDQN reward alignment

This opt-in experiment preserves the strong original IAT CNN / Light PS CNN,
healthy Context checkpoint, parent compact encoder/normalizer, fixed Context ->
Frozen prediction path, batch5, replay weight 0.1, Context LR 0.00025 and DDQN
hyperparameters. It does not use the weak Light-IAT checkpoint. Existing drivers,
checkpoints and result directories are unchanged.

## Reward

For each paid TRAIN update, observe two fixed, disjoint TRAIN probes immediately
before and after the **one actual batch5/Adam update**:

`reward = (generalization_CE_before - generalization_CE_after)
          - forgetting_weight * max(0, retention_CE_after - retention_CE_before)
          - request_cost`.

The gain remains signed. No sign-only target, reward clipping, target normalization,
copied update branch, future rollout or auxiliary prediction head is introduced.
Skip reward is zero and needs no current label. DDQN uses the unchanged actual
next-input transition and bootstrap for later decisions. The measured effect is
one update, not an overlapping multi-step rollout return.

The generalization probe has up to 24 seeded rows from each TRAIN level 0/1/3,
sampled outside the union of all 300-row adaptive TRAIN streams and the retention
reference. Each probe retains all classes and removes feature duplicates. Macro
cross entropy gives equal weight to present classes; the three generalization
levels have equal weight. Retention uses up to 32 rows from the already reserved
original-domain TRAIN reference. Probe IDs and feature duplicates are excluded
across intensities, removed from TRAIN replay anchors, and checked against every
actual update batch. Sampled adaptive streams themselves stay unchanged.

These probes are held out from this experiment's adaptive updates. They may have
appeared in historical CNN/Context/parent-encoder pretraining. Repeated TRAIN
probe observation is a training objective, not an independent test or proof of
future generalization. VAL and TEST remain separate. On VAL/TEST the reward probes
are never observed, DDQN/Actor weights stay fixed, and only Context adapts on paid
labels. Offline metrics still score all pre-feedback predictions. The paper-table
reward field in evaluation traces is observer-only and does not train any model.

## Matched comparison and VAL selection

Defaults run four utility configurations: forgetting weight 0.5 or 1.0, request
cost 0 or 0.001 (raw CE units). A paper-reward DDQN is retrained with the exact same
streams, initial weights, seeds, TRAIN probe exclusions and optimizer settings.
The existing trained Actor, uncertainty, random, front-loaded and no-query controls
are evaluated with identical original evaluation anchors, rows and stream order.
They are deployment controls, not independently retrained no-RL ablations.

Every candidate gets at most three epochs. Per-candidate epoch selection retains
the existing VAL accuracy-minus-positive-forgetting criterion. Across utility
candidates, choose lowest actual VAL label use subject to mean online accuracy
within 0.5 percentage points of the front-loaded VAL reference and mean positive
original-domain forgetting within 0.5 points of that reference. If no candidate
qualifies, report the fallback with highest VAL accuracy minus positive forgetting;
do not claim the goal was achieved. Even a qualifying candidate is marked successful
only if it actually consumes fewer labels than the reference.

The selected utility candidate and matched paper DDQN are then evaluated on the
seven independent TEST levels. TEST never chooses an epoch, cost, weight or policy.
Label saving at the same TEST accuracy/forgetting is an empirical question; VAL
qualification does not establish it. This remains a 300-row-per-domain, seed-42
pilot, not the complete multi-seed benchmark.

The competitive-gap Context descent is unchanged. The new utility reward is a
different query objective; closed-loop interaction is preserved, but a formal
common-objective min-max claim is not established by this implementation.

## Run separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_utility.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_utility.ps1 -Dataset PS
```

The matching original healthy and parent-Actor model binaries must already exist
locally, as for the preceding DDQN experiment. No CNN/healthy-Context retraining
or installation is performed. Defaults are **three epochs per candidate**, five
candidates total, with a 20% hard cap per level and the same reset protocol. Use
`-Action Evaluate` with identical settings to repeat only the selected TEST runs.
To resume, repeat the same command; DDQN resumes completed epoch boundaries.

Console output includes each candidate, actual labels/cap, online accuracy, TD
updates, signed reward means, gain/harm/cost means, positive-reward fraction,
VAL eligibility and seven-level TEST comparison summaries. Output directories are
`protocol_runs/two_action_utility_seed42_p300_e3_v1`. `Results/comparison_summary.csv`
compares UtilityDDQN, matched PaperDDQN and the five fixed query controls.

```powershell
.\AMCAL\push_amcal_two_action_utility_results.ps1 -Dataset Both
```

The helper stages only this experiment's JSON/CSV, excluding model binaries and
rejecting unrelated staged files. Its candidate logs include sufficient reward
components to diagnose an all-negative target or collapsed query policy.

Unit tests cover signed rewards, class weighting, frozen probe observation,
probe/update separation, feature-duplicate exclusion, TRAIN-only reward access,
zero-cost skip, frozen TEST Q, unchanged parent-Actor evaluation and VAL selection
with explicit failure reporting. Original DDQN contract tests are also run by the
user-invoked wrapper. No tests, imports, training or evaluation were run by the
assistant while preparing this change.
