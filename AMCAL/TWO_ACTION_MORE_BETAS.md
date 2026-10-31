# Additional beta values: 25, 30, 35, 40

This isolated extension trains and tests only beta 25/30/35/40 for the exact same
reward `paper_reward + beta * independent same-domain TRAIN CE decrease`.
The feedback budget remains **20%**; these numbers are reward coefficients.
Previously trained beta 0/10/15/20/100 models and results are untouched.

The existing parent `two_action_paper_gain_seed42_p300_e3_v1/protocol.json` supplies
the source recipe, hashes, data/probe identities and three-epoch TRAIN settings.
CNN, healthy initial Context, parent encoder/normalizer, original IAT CNN / Light
PS CNN, batch5, Context LR 0.00025, source replay weight 0.1 and DDQN settings
remain unchanged. No synthetic drift, request cost or new forgetting penalty is
introduced. Only the four new DDQNs train, from the same initial weights/seeds.

Each beta's epoch is selected on VAL by the unchanged criterion. Every requested
beta is then tested on all official intensities 5/7/10/12/15/17/20, including any
zero-query policy. TEST does not choose a beta. Context, Adam, replay/paid memory
and cap reset for each beta/intensity; query weights stay frozen. Gain-probe labels
are accessed only in TRAIN. Metrics are pre-feedback online and final target
AC/PR/RC/F1, actual labels, source retention and forgetting.

## Run separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_more_betas.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_more_betas.ps1 -Dataset PS
```

Defaults: 300 TRAIN rows per domain from the parent recipe, three epochs per new
model, 300 TEST rows per intensity, four betas, 28 TEST cases per dataset. No CNN
or initial Context retraining and no installation occur. Each console case names
dataset, beta, intensity and actual label consumption, plus online/final metrics
and original-domain forgetting.

The four model folders are added under
`protocol_runs/two_action_paper_gain_extra_seed42_p300_e3_v1/Candidates/beta_25`
and `beta_30`, `beta_35`, `beta_40`. Existing beta 15/20 folders there are untouched.
Repeating the command verifies completed training or resumes at epoch boundaries.
`-Action Evaluate` repeats only TEST after all four candidates are trained.

Results are separate from the previous five-beta sweep:
`protocol_runs/two_action_paper_gain_more25_30_35_40_seed42_trainp300_testp300_v1`.
`summary_by_level.csv` reports 28 rows and `summary_mean.csv` four beta means.
Case traces, checksummed completion records and checkpoint/epoch manifests support
resuming without rerunning completed inference. `-TestRows 0` evaluates all official
TEST rows in a separate folder, reusing the same completed training checkpoints.

```powershell
.\AMCAL\push_amcal_two_action_more_betas_results.ps1 -Dataset Both
```

The push helper stages paper-plus-gain JSON/CSV, including the new TRAIN/VAL and
TEST files; model binaries are excluded and unrelated staged files rejected.
This is a single-seed exploratory coefficient comparison; no published manuscript
choice or existing VAL selection is automatically changed based on TEST results.

User-run tests check all four requested coefficients, isolated model locations,
provenance, inclusion of zero-query behavior, frozen TEST weights, per-case resets,
explicit console labels and verified caching. No imports, tests, training or
evaluation were executed by the assistant while preparing these files.
