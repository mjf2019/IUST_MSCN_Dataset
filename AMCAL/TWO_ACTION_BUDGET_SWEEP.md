# Evaluation-only budget sweep: paper DDQN versus uncertainty

Budgets: 30%, 40%, 50%, 60%, 70%, 80%. Official TEST levels:
5/7/10/12/15/17/20. No training or parameter selection is performed.

The DDQN checkpoint is the matched **original-paper reward, no synthetic drift**
candidate from the utility experiment:

`protocol_runs/two_action_utility_seed42_p300_e3_v1/Candidates/paper/Models/two_action_ddqn.pth`.

This is the PaperDDQN previously reporting mean online AC about 71.48% on IAT and
75.57% on PS at 20% budget. It is not the synthetic-drift candidate, utility policy,
or older initial DDQN pilot. Completion/model hashes, selected epoch, code/data
provenance, healthy CNN/Context hashes and parent-Actor hashes are verified.

The model was trained with 20% budget. It is now evaluated with larger hard caps
without retraining. DDQN stays frozen and greedy. Uncertainty uses the unchanged
saved TRAIN threshold, with no TEST calibration or forced consumption. Both use
fixed Context -> Frozen predictions, batch5, Context LR 0.00025 and replay weight
0.1. IAT uses the original CNN and PS uses Light CNN. All rows are predicted before
their own feedback. Only Context adapts from acquired labels; no query learner is
updated during evaluation.

Each method/budget/intensity gets a fresh copy of the same healthy Context, fresh
Adam and replay memory, empty paid history and a new `floor(budget*N)` label cap.
Frozen CNN and query policy do not carry adaptive state between cases. Input rows,
ordering, seed and source audit are matched across methods/budgets. The source
audit is observer-only; it does not enter updates or selection.

## Run IAT and PS separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_budget_sweep.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_budget_sweep.ps1 -Dataset PS
```

Default `-Method Both` evaluates both methods at all six caps (84 cases per
dataset). Defaults retain the same 300 sampled TEST rows per level as the previous
comparison. Use `-PilotRows 0` for all official TEST rows; this can take longer and
has its own output directory. Only TEST row count changes; the original training
lineage is reconstructed and checked with its saved pilot size first.

If needed, evaluate only one policy:

```powershell
.\AMCAL\run_amcal_two_action_budget_sweep.ps1 -Dataset IAT -Method DDQN
.\AMCAL\run_amcal_two_action_budget_sweep.ps1 -Dataset IAT -Method Uncertainty
```

Single-method invocations have separate default output directories. `-Budgets`
accepts a subset of the six requested caps. Change `-Output` when changing a saved
sweep's budget list or provenance. `-PaperRun`, `-ActorRun` and `-Teacher` accept
explicit matching local checkpoint folders. No model or dependency is downloaded
or installed. The existing local model binaries from the completed experiments
are required.

## Console and results

Every case prints START/DONE/REUSED with case index, dataset, full method name,
TEST budget, attack level and maximum labels. DONE/REUSED prints online AC/PR/RC/F1,
actual labels/cap and percentage, final target AC/F1 and original-domain AC
forgetting. A separate summary is printed after each method/budget and at the end.
The TRAIN budget is explicitly printed as 20%, distinct from the current TEST cap.

Default output:
`protocol_runs/two_action_paper_budget_seed42_p300_both_v1`.

- `summary_by_level.csv`: every method/budget/level, with online metrics, consumed
  labels, final target metrics and original-domain forgetting.
- `summary_mean.csv`: means across seven levels for each method and budget.
- `budget_XX/METHOD/level_X/`: execution metadata, full metrics, causal trace and
  checksummed completion record.

Repeating the same command reuses verified completed cases and reruns only partial
or missing cases. Corrupted completed results or provenance mismatches are rejected.
No old experiment outputs are overwritten. A budget is a maximum: a policy may
consume less. This remains a single-seed small-data evaluation unless full TEST is
explicitly requested; increased caps are not evidence of a retrained policy.

```powershell
.\AMCAL\push_amcal_two_action_budget_results.ps1 -Dataset Both
```

The push helper stages only the new sweep's JSON/CSV, excluding model binaries and
rejecting unrelated staged files. User-run CPU tests cover checkpoint identity,
all six caps for both policies, frozen query weights, independent resets, explicit
console labels, completion reuse and corruption detection. No tests, imports,
training or inference were run by the assistant while preparing this code.
