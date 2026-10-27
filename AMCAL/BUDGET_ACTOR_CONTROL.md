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
