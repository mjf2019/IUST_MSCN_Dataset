# Reward beta refinement, fixed 20% feedback cap

Requested reward coefficients:

- IAT: beta 26, 27, 28, 29.
- PS: beta 45, 50, 55, 60.

These are **reward beta values**, not labeling budgets. Every intensity retains
the same 20% cap. The reward remains `paper_reward + beta * independent same-domain
TRAIN CE decrease`. No cost, synthetic drift or new forgetting penalty is added.
All prior models and outputs are untouched.

The source recipe is the completed parent
`two_action_paper_gain_seed42_p300_e3_v1`. Source code/model/data/probe provenance
is checked. Each new DDQN trains from the same parent encoder and healthy initial
Context for three epochs, with 300 TRAIN rows per domain, unchanged seeds and
VAL epoch selection. CNN and initial Context are not retrained. Original IAT CNN,
Light PS CNN, Context LR 0.00025, batch5 and replay weight 0.1 remain fixed.

All four requested coefficients for each dataset are tested on official levels
5/7/10/12/15/17/20, including any coefficient with zero requests. No cross-beta
TEST selection occurs. Query policies stay frozen; Context/Adam/feedback memory
and cap reset per coefficient/intensity. Current prediction precedes its label.
Gain-probe labels are accessed only in TRAIN.

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_beta_refinement.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_beta_refinement.ps1 -Dataset PS
```

Each command automatically chooses its dataset-specific grid. Console output
explicitly prints REWARD_BETAS and LABEL_BUDGET, then dataset, beta, intensity,
actual consumption and online/final AC/PR/RC/F1 and original-domain forgetting.
Default TEST size is the same 300 rows per level as previous comparisons: four
coefficients and 28 TEST cases per dataset. Use `-TestRows 0` for every official
TEST row in a separate output folder. This does not imply full-data retraining.

New models are added to `two_action_paper_gain_extra_seed42_p300_e3_v1/Candidates`
under beta_26, beta_27, beta_28, beta_29 for IAT or beta_45, beta_50, beta_55, beta_60 for PS.
Existing model folders there are preserved. Completed training is verified/reused;
interrupted training resumes at epoch boundaries. `-Action Evaluate` repeats only
TEST after training. No packages are installed.

New result roots are
`protocol_runs/two_action_paper_gain_refine_seed42_trainp300_testp300_v1`
inside each dataset's folder. `summary_by_level.csv` has 28 cases and
`summary_mean.csv` has four beta means. Full causal traces, model hashes, selected
epochs and checksummed completion records are retained. Repeating the same command
reuses verified completed TEST cases.

```powershell
.\AMCAL\push_amcal_two_action_beta_refinement_results.ps1 -Dataset Both
```

The helper stages paper-plus-gain JSON/CSV, including new TRAIN/VAL and TEST files,
excluding binary models and rejecting unrelated staged files. This remains a
single-seed exploratory coefficient comparison; no previous VAL choice or paper
table is automatically changed using these TEST results.

User-run tests cover the two distinct beta grids, preserved 20% caps, independent
model paths, provenance, zero-query inclusion, frozen query weights, per-case reset,
console labels and completion integrity. No imports, tests, training or evaluation
were run by the assistant while preparing this code.
