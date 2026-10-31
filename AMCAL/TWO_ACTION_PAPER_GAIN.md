# DDQN with original paper reward plus same-domain update gain

This separate reward ablation implements exactly:

`query reward = original_paper_reward + beta * G`

`G = mean CE of an independent same-domain TRAIN probe before the actual update
     - mean CE of that identical probe after the actual update`.

Skip reward remains zero. The paper table is unchanged: +3 when Frozen and Context
are both wrong, +2 when only Frozen is wrong, -2 when only Context is wrong, and
0 when both are correct. All predictions are fixed before acquiring the label.
Positive and negative gains are retained without clipping, sign-only targets or
normalization. There is **no new request cost or forgetting penalty** in the query
reward. The Context's existing replay regularizer remains unchanged.

## What stays fixed

The healthy original IAT CNN / Light PS CNN, initial Context checkpoint, preprocessing,
parent compact encoder and TRAIN normalizer, two actions, DDQN settings, batch5,
Context LR 0.00025, replay weight 0.1, 20% budget and per-intensity reset protocol
are preserved. Only the query DDQN is retrained. There is no synthetic drift,
additional head, future rollout or copied-update branch. A real batch5 update is
performed once, with the same sampled paid/source replay, and measured before/after.

TRAIN and VAL remain official levels 0/1/3. TEST remains official unseen levels
5/7/10/12/15/17/20. Each TRAIN level has its own fixed probe (24 rows by default),
not a mixture of all three domains. It uses ordinary mean CE, matching the proposed
formula, rather than macro-class CE. Probe IDs/features are reserved outside the
sampled adaptive TRAIN streams and removed from TRAIN replay anchors. Probes do
not enter the actual current/paid/replay update batch. The preexisting independent
source reference stays excluded from updates but is not measured for this reward.
These are held out from the current adaptive updates, not necessarily from historical
CNN/Context/parent-Actor pretraining.

Gain observation occurs **only during TRAIN**. On VAL/TEST the query policy is frozen
and uses pre-feedback features; only Context adapts on acquired labels. TEST trace
paper rewards are observer diagnostics, with unobserved gain marked null; no query
training uses those labels. Source retention metrics remain available for evaluation.

## Matched control and beta selection

Defaults train beta=0, 10, 100, each for three epochs on 300 rows per TRAIN domain.
Beta=0 is always included: it uses exactly the paper-reward update path without
probe forward passes. All candidates start with identical model weights/seeds and
the same probe-excluded TRAIN replay anchors. Consequently the control isolates
the additive reward term, rather than comparing unrelated source checkpoints.

The beta grid is declared before training. Values 10/100 allow raw CE changes to
have an observable scale relative to rewards 2/3; they are hypotheses, not assumed
optimal values. Per-candidate epoch selection and cross-beta selection use VAL mean
online accuracy minus positive original-domain accuracy forgetting. Fewer labels
and smaller beta break ties. All-zero-request policies are excluded from the
cross-beta comparison; they are not called improved sample selection. Beta=0 can
win and then remains the selected method. No TEST results select a beta or epoch.
NoQuery VAL results, excluded betas, actual label savings and score changes are
recorded explicitly. Known-level VAL may still fail to predict unseen-level TEST
performance; this pilot does not assume the new reward will win.

The selected DDQN, matched beta=0 control, saved Actor, uncertainty, random,
front-loaded and no-query controls are evaluated on matching official TEST rows,
order, source audit and healthy initial Context. Fixed query controls are deployment
controls, not independently retrained no-RL ablations. If beta=0 wins, its TEST
inference is reused for the baseline row rather than redundantly rerun.

## Run separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_paper_gain.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_paper_gain.ps1 -Dataset PS
```

Defaults: **300 rows per domain, three epochs per beta, three candidates total**.
The completed local healthy and parent-Actor model files from the preceding strong
two-action experiment are required. There is no CNN/initial Context retraining or
package installation. Outputs use
`protocol_runs/two_action_paper_gain_seed42_p300_e3_v1`; previous runs are untouched.
Repeating the same command resumes at completed epoch boundaries. `-Action Evaluate`
with identical settings repeats selected TEST only. `-Betas` changes the grid while
always including zero; use a fresh `-Output` when changing a saved run's protocol.

Console lines identify dataset, beta, TRAIN/VAL/TEST and level. They print actual
labels, TD updates, original-paper reward mean, signed gain mean, weighted gain
mean, online metrics and original-domain forgetting. The final summary includes
selected versus beta=0 DDQN and the five fixed query controls.

```powershell
.\AMCAL\push_amcal_two_action_paper_gain_results.ps1 -Dataset Both
```

Only this variant's JSON/CSV are staged; model binaries and unrelated staged files
are excluded/rejected. Output includes per-sample TRAIN reward components and
`Results/comparison_summary.csv` for independent TEST comparison.

Closed-loop interaction and competitive-gap Context descent remain. The additive
query objective does not itself prove a common formal min-max objective.
User-run tests cover exact additive/sign behavior, beta-zero compatibility,
same-domain mapping, plain mean CE, absent retention/cost terms, probe/update
separation, TRAIN-only reward access, frozen TEST query weights and VAL-only
selection with explicit no-query rejection. No tests, imports, training or evaluation
were run by the assistant while preparing this change.
