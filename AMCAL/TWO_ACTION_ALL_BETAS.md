# Test every beta: 0, 10, 15, 20, 100

This extends the existing original-paper-plus-gain DDQN experiment. All five
coefficients are evaluated on all seven official TEST levels 5/7/10/12/15/17/20,
including nonselected beta=100. There is no cross-beta TEST selection or exclusion
of policies that make no requests. Results are a descriptive coefficient comparison.
The original VAL selection record is never modified.

The completed parent `two_action_paper_gain_seed42_p300_e3_v1` supplies existing
beta 0/10/100 checkpoints; they are verified and never retrained or overwritten.
Only missing beta 15/20 models are trained, from the same initial healthy CNN/Context
and parent query encoder, with the same TRAIN streams/probes, seeds, update settings
and three-epoch recipe. Each model's epoch is selected on VAL, exactly as before.
TRAIN remains the parent's 300-row-per-domain pilot. CNN and initial Context are
not retrained; only the extra DDQN candidates are fitted.

Parent source-code hashes, data/model lineage and probe/replay-anchor identities
must match. The exact reward remains `paper_reward + beta * same-domain TRAIN gain`.
Context LR 0.00025, replay weight 0.1, batch5, original IAT CNN / Light PS CNN, fixed
Context -> Frozen prediction and 20% per-level cap are unchanged. Query policy
weights stay frozen during TEST. Each beta/intensity resets Context, Adam, source
and paid memory and label cap. All rows are scored before their own feedback.

## Run IAT and PS separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_all_betas.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_all_betas.ps1 -Dataset PS
```

Default: the same 300 TEST rows per intensity as the previous comparison, all
five betas, 35 TEST cases per dataset. This means complete **beta coverage**, not
full dataset size. To evaluate every official TEST row instead:

```powershell
.\AMCAL\run_amcal_two_action_all_betas.ps1 -Dataset IAT -TestRows 0
.\AMCAL\run_amcal_two_action_all_betas.ps1 -Dataset PS -TestRows 0
```

Extra candidates use a separate reusable training folder:
`protocol_runs/two_action_paper_gain_extra_seed42_p300_e3_v1/Candidates/beta_15`
and `beta_20`. Changing TEST size does not retrain these completed models.
Use `-Action Evaluate` to run only all-beta TEST after both extras are complete.
Use `-Action Train` to prepare extras without opening TEST inputs. `-ParentRun`,
`-CandidateRun`, `-Teacher`, `-ActorRun` and `-Output` accept explicit matching folders.

The default evaluation folder is
`protocol_runs/two_action_paper_gain_all_seed42_trainp300_testp300_v1`;
full TEST uses `testp0` instead. `summary_by_level.csv` includes all five betas and
seven levels. `summary_mean.csv` contains five rows with AC/PR/RC/F1, consumed labels,
final target metrics and original-domain forgetting. `test_manifest.json` records
the actual TEST data and checkpoint hashes/selected epochs. Case traces and
checksummed completion records permit resuming without rerunning completed cases.

Console headers identify dataset, beta, level, budget and row count. Every completed
case prints online metrics, actual labels/cap, final target AC/F1 and source forgetting.
No package installation occurs. Checkpoint binaries must exist locally.

```powershell
.\AMCAL\push_amcal_two_action_all_betas_results.ps1 -Dataset Both
```

The helper includes new extra-candidate TRAIN/VAL JSON/CSV and all-beta TEST results,
excluding model binaries. Existing tracked parent results produce no new diff unless
they changed. Unrelated staged files are rejected. This remains seed 42; full TEST
coverage does not imply full-data retraining or multiple seeds. Comparing all betas
on TEST is retrospective analysis, not evidence that a beta was independently chosen
on TEST. No experiments or manuscript selections are automatically changed.

CPU tests cover exact beta coverage, reuse routing for existing versus extra models,
provenance checks, inclusion of zero-query candidates, frozen TEST policies, independent
resets, console labels, cache reuse and corruption detection. Tests and code were
written without executing imports, tests, training or evaluation by the assistant.
