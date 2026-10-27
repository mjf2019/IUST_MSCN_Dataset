# Fixed-Context budget Actor control

This opt-in pilot changes query-Actor training and deployment only. The full healthy
Natural checkpoint supplies the unchanged CNN, initial Context and Context update
function. Every evaluation intensity starts from those exact Context weights.
The prediction path is always Context -> frozen CNN. Routing/future heads never
learn or select predictions in this control.

## Training objective and min-max scope

The original signed TRAIN score remains `S = batch5_gap + G - F - cost`, with the
original gain/retention weights and cost. S is measured with the existing isolated
one-update TRAIN branch, future TRAIN block, and independent old-TRAIN probe.
No TEST labels enter targets or Actor fitting. All offline TRAIN labels may be
used to construct targets, as in the existing full-information Actor training.

A single small uniform-feedback collection over TRAIN 0/1/3 creates fixed states
and targets. These are temporary Context trajectories starting independently from
the healthy Context. They do not replace the saved healthy Context. Neither the
router nor query Actor learns during collection. State normalization is fitted
only to this new TRAIN collection; entropy is already present in the state.

The Actor then maximizes `mean(p * stopgrad(S))` subject to `mean(p)=q` in each
minibatch, for q=0.05 or q=0.10. An implicitly differentiated common logit offset
enforces the quota. This is an actual constrained max of the existing signed
local objective; it is not balanced-sign training, a sign-only target, an
uncertainty reward, or a logit bonus. Adding a constant cost to every S does not
change the ranking gradient under the equality constraint.

The Context min update remains the existing batch5 gap plus 0.1 protected replay.
Its gradient is paid-batch CE plus protected replay CE because frozen CE is a
constant. The new constraint changes the max problem; it does not establish exact
equivalence to the original unconstrained or on-policy multistep min-max game.
Actor fitting holds the collected Context trajectories fixed. The healthy Context
checkpoint is never retrained or overwritten. During VAL/TEST, only Context
adapts with acquired labels; the Actor stays fixed. This pilot does not prove
that the existing signed score ranks true future adaptation utility optimally.

## Causal deployment

Raw Actor logits are ranked against a fixed empirical TRAIN-logit distribution.
The cutoff tracks remaining labels / remaining rows. Equal-score ties use an
independent seeded draw. An exhausted budget blocks requests; if the remaining
budget equals the number of remaining rows, the tail is requested to satisfy the
exact budget. Forced-tail labels are reported separately. This is a causal
budget mechanism, not a claim that a raw sigmoid probability is calibrated.
No future TEST features or labels are used for request decisions. The runner
assumes the known finite stream length, as do the existing controls.

VAL chooses the Actor checkpoint by online accuracy minus positive clean-source
drop. TEST resets Context/Adam/memory per intensity and reports 5/7/10/12/15/17/20.
Matched FrontLoaded and Uncertainty (VAL q75) controls use the same healthy Context,
update, seed, pilot subset and budget. The uncertainty control may spend less than
the cap. A matched FrontLoaded20 reference also runs on the same pilot rows, so
low-budget results are not compared to a different full-data population. Clean forgetting is measured on level-0 TEST. Target final scores are
same-stream descriptive endpoints, not independent generalization measurements.

Default diagnostic: 300 rows per domain, 2 Actor epochs, 80 small fit steps per
epoch. CNN and the full healthy Context have already been trained; this is not a
from-scratch small-data experiment. Full runs can use PilotRows=0 after reviewing
the pilot. Output directories begin `budget_actor_q05_` or `budget_actor_q10_`.

```powershell
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset IAT -BudgetPercent 5
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset PS -BudgetPercent 5
.\AMCAL\push_amcal_budget_actor.ps1 -Dataset Both
```

No packages are installed. Preflight includes exact quota/gradient contracts and
small real-stream contracts in the user's existing Torch environment.

## UtilityEarly: separate experimental correction

Enable `-UtilityEarly` to create a new `_scaled_early20` run; previous runs and
the default method are preserved. The healthy Context/CNN and the batch5 update
stay identical. Only the Actor is fitted. This experiment addresses two observed
failures: raw target variance dominated by the competitive gap, and requests too
late to improve many subsequent predictions.

From measured TRAIN targets only, fit fixed positive standard deviations for
`gap`, `weighted_gain`, and `weighted_harm`. Fit the Actor on
`gap/sd_gap + weighted_gain/sd_gain - weighted_harm/sd_harm - cost`.
Constant components use scale 1. No sign truncation, TEST fitting, extra heads,
or inference bonus is added. Raw components and the actual ranking target are
both saved. The scales are written to JSON and stored in the Actor checkpoint.
Normalization balances units/variation; it does not prove that a component is
predictive, remove noise, or guarantee better selection.

Feedback is allowed only in the first `floor(0.2*N)` stream positions, for both
temporary TRAIN collection and Actor VAL/TEST. The *total* budget remains 5% or
10% of all N rows. TRAIN uniform collection uses this same window; the Actor's
within-window probability quota is therefore approximately 25% or 50%. Causal
remaining-budget ranking forces any remaining requests by the window deadline.
All N rows are scored before their own feedback. After the window there are no
requests or Context updates. The finite stream length is known, as in the
existing evaluation. Window length is fixed in advance and not chosen on TEST.

Positive scaling preserves the competitive gap and the Context descent
direction, but reweights the max objective and adds a temporal constraint. This
is a revised constrained *local* min-max surrogate, not a proof that the original
multistep game is unchanged. The Context gradient continues to be CE plus its
existing protected replay penalty. Query fitting uses fixed collected TRAIN
trajectories, rather than claiming to solve the full on-policy game.

Matched TEST controls are Actor, UniformEarly (same query window), Uniform
(full-stream spacing), FrontLoaded (same budget), Uncertainty (same cap, VAL q75),
and FrontLoaded20. UniformEarly is needed to check whether Actor ranking adds
value beyond the imposed timing restriction. Comparing to the previous run alone
cannot separate reward scaling from earlier budget consumption. Endpoint final
target AC is descriptive rescoring of the same stream; it is not a holdout score.

```powershell
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset IAT -BudgetPercent 10 -PilotRows 300 -Epochs 2 -UtilityEarly
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset PS -BudgetPercent 10 -PilotRows 300 -Epochs 2 -UtilityEarly
.\AMCAL\push_amcal_budget_actor.ps1 -Dataset Both
```

Console output reports each level's AC/PR/RC/F1, label usage, source forgetting,
and final target/source accuracy, followed by a cross-method mean table. Completed
runs show their saved summaries without rerunning the evaluation.

## No-fit ranking diagnosis

`run_amcal_actor_diagnostics.ps1 -Dataset IAT` (or PS) audits the existing
10%-budget, 300-row, two-epoch UtilityEarly checkpoint. It does not run new Actor
epochs, select a new checkpoint, refit scales/normalization, or evaluate TEST.
The Epochs argument identifies the saved run only.

On TRAIN and VAL levels 0/1/3, it constructs common, fixed uniform-early Context
trajectories starting from copies of the same healthy checkpoint. Actor, router,
future and utility heads are checked unchanged. Temporary Context/Adam updates
are discarded. TRAIN uses its original held-out source probe. VAL reserves a
separate source-VAL probe and excludes its IDs and feature duplicates from all
VAL candidate streams. Current update, replay, future block and retention probe
are separated using the existing stream contracts.

This intentionally uses full-information observer labels, including future VAL
labels, for offline two-branch diagnostics only. Therefore the observer audit is
not a deployable VAL policy and its feedback counts are not a claim about VAL
label efficiency. No observer label enters an Actor update or model selection.

For each measured candidate, record Actor logit, the exact target using the
checkpoint's frozen TRAIN scales, future CE-loss gain, old-domain CE-loss harm,
and raw signed `utility = weighted_gain - weighted_harm - cost`. The future block
undergoes no additional updates. Utility is a one-update loss probe; it is not
final target accuracy or the cumulative value of a sequence of requests.

Report within-level Pearson/Spearman correlations for Actor/target/utility and
top-k means for Actor, OracleTarget, OracleUtility and the exact expected mean
of a uniform random subset. Every selector is scored on the same candidate
states and independent probes; selectors do not produce their own adaptive
rollouts. Oracle selectors use observer targets, are never deployed, and only
bound this measured fixed-trajectory ranking problem. Report macro-means over
levels instead of pooling different intensities and selecting the easiest one.

Read Actor-target agreement to diagnose target learning, TRAIN/VAL disagreement
to inspect generalization, and target-utility agreement/OracleTarget utility to
audit the target. None of these correlations alone establishes a causal failure
or proves a policy will improve online accuracy. No new reward is chosen by
this tool. Results are saved in each existing run's `actor_diagnostics/` folder
and included by `push_amcal_budget_actor.ps1 -Dataset Both`.

## Pairwise ranking initialization, followed by the original max phase

`-RankWarmStart` implies UtilityEarly and creates a separate `_rankwarm` run.
CNN, healthy initial Context, Context batch5 update, raw components, TRAIN-only
signal scales and first-20%-window deployment remain the UtilityEarly design.
No new head, sign-only reward, or test-time rule is introduced.

Before the two normal quota-max epochs, the Actor receives two initialization
epochs (80 small steps each by default) of TRAIN-only pairwise softplus ranking.
Pairs are from the same intensity to prevent learning only a between-intensity
offset. Each pair uses the sign of the difference of the existing scaled S
targets. This auxiliary loss only initializes the Actor; it is not represented
as an exact min-max step. The value head, Context checkpoint and input tensors
receive no gradient from pairwise fitting. Large collections subsample at most
256 TRAIN rows per step to bound pair-matrix memory.

The subsequent max phase resets Adam and uses the existing signed quota-max
objective alone, including target magnitudes; there is no pairwise term in that
phase. Only checkpoints from the max phase are eligible for the deployed model,
using the same online VAL accuracy minus positive source-drop rule. Warmup
ranking results are diagnostics, not a silently selected alternative model.
Thus the local max objective is unchanged after initialization; this does not
prove a complete multistep min-max solution or guarantee improvement.

Once, before fitting, collect fixed full-information observer probes on VAL
0/1/3, with an independent source-VAL reference excluded from candidate streams.
Temporary Context updates are discarded. No VAL labels or targets enter Actor
gradients, TRAIN scaling or state normalization. Audit initial, both warmup, and
each max stage on the same fixed TRAIN and VAL states/targets. The console and
`ranking_stages.json` show within-level Spearman and top-k target regret. Undefined
initial correlation means constant logits. TRAIN fit alone is not evidence of
generalization; inspect VAL ranking as well as the ordinary online VAL metrics.
VAL observer probes are not counted as deployable budgeted evaluation.

```powershell
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset IAT -BudgetPercent 10 -PilotRows 300 -Epochs 2 -RankWarmStart
.\AMCAL\run_amcal_budget_actor.ps1 -Dataset PS -BudgetPercent 10 -PilotRows 300 -Epochs 2 -RankWarmStart
.\AMCAL\push_amcal_budget_actor.ps1 -Dataset Both
```

Epochs=2 here identifies two quota-max epochs, in addition to the two fixed
ranking-initialization epochs. TEST controls, per-level resets and console
summaries remain the same as UtilityEarly. Existing diagnostic script defaults
still address the earlier run; the new run carries its stage-wise audit inline.
