# Compact competitive selector (opt-in)

This preserves the healthy initial Context and CNN, the batch5 update function,
Context LR 0.00025 and protected replay weight 0.1. IAT uses the original CNN;
PS uses Light CNN. Prediction always follows Context -> Frozen. This isolates
query selection; it is not a three-action routing ablation.

The old PairOnly, budget Actor and constrained-DDQN entry points keep their
existing behavior. The new stream branches are guarded by `competitive_selector`.

## Concrete changes

* Replace the redundant 335-dimensional raw state with 30 features for six classes:
  both probability vectors, confidence, margin, entropy, distortion, agreement,
  Jensen-Shannon divergence, paid-embedding novelty, paid-label coverage/support,
  remaining budget, stream position and time since update. No unqueried labels
  enter policy inputs. The Actor has one 16-unit hidden layer and 513 parameters.
* Fit feature moments and positive component SDs once on TRAIN 0/1/3. Do not
  center signed targets or discard magnitudes. No VAL/TEST normalization fitting.
* Directly ascend `mean(sigmoid(logit) * detached_score)`; the independent value
  regressor is diagnostic and does not determine the policy. Bounded logits limit
  policy confidence; they do not guarantee successful learning or prevent every
  form of saturation. There is no artificial spending quota or forced tail.
* TRAIN uses stochastic requests; deployment uses probability >= 0.5 under a hard
  20% cap. The cap is a maximum, not a required spend. No bonus or early window.
* Fix a TRAIN/TEST update mismatch: a hypothetical observer draw copies only the
  replay sampler RNG and does not advance the live sampler. Draw the real update
  batch first, then remove its features from the future observer. The real paid
  update draws normally, with the same data and Adam update law as TEST. Shadow
  Context/Adam are discarded; the measured branch never replaces live state.
* Every domain starts from the same healthy Context. After each TRAIN collection
  round, only the query head is optimized on recorded frozen-round states/targets.
  The healthy initial Context is never replaced by a temporary collection state.

## Exact scope of min-max and RL

At frozen round k, define `d_phi` as equal-weight paid-batch Context CE minus
detached Frozen CE. The measured one-update TRAIN future gain G_k and positive
old-domain harm F_k are detached. With positive fixed TRAIN scales s:

`J_k = E[p_theta * (d_phi/s_d + beta*G_k/s_G - lambda*F_k/s_F - c)]`

The Actor maximizes this frozen-round surrogate directly. Conditional on a real
query, Context minimizes the same batch gap plus protected replay. Scaling both
the Context gap and its auxiliary replay regularizer by `1/s_d` preserves the
original Context minimization direction; the implementation retains the healthy
optimizer/update function rather than rescaling Adam gradients. A positive
objective scaling is not an assertion of numerically identical Adam trajectories.

This is alternating local shared-gap min-max with a contextual-bandit query
policy. It is **not** sequential DDQN, a global on-policy saddle-point solver,
or the policy-versus-dual game from constrained RL. Only one-update future effects
are observed, with no updates inside the future block and no bootstrap. Batch
gain includes triggering replay, not solely the marginal information in one new
label. Neither better TEST accuracy nor superiority over uncertainty is assumed.

## Data, controls and selection

Uniform TRAIN initialization estimates scales; later collections use the current
Actor. Context/Adam/memory reset per level. TRAIN uses 0/1/3 only. All TRAIN
reference IDs/features are excluded from candidate streams; reference features
are also excluded from anchors. VAL and TEST reserve independent source audits;
their IDs/features are excluded at every target intensity before pilot sampling.
Scored populations therefore differ from older full-stream tables.

All five deployment controls use the same initial Context, stream order,
batch5/replay and cap: Actor, Context-entropy q75, random requests, front-loaded
feedback, and no-query Context. Routing is fixed Context in every control. The
random policy samples requests with probability 0.2, so it may underspend.
Uncertainty q75 comes from pre-feedback TRAIN entropies; no VAL/TEST fitting.
These are deployment controls, not independently retrained no-RL ablations.

Checkpoints are selected by VAL mean online AC minus positive source forgetting;
lower label use only breaks ties. This does not claim constrained feasibility.
Every scored prediction precedes its own feedback. TEST intensities
5/7/10/12/15/17/20 are read only with explicit Evaluate/All. JSON/CSV record online
AC, PR, RC, F1, labels, source forgetting and descriptive final-target endpoints.

Default entry point is `run_amcal_competitive_selector.ps1 -Dataset IAT` (Train
only). Completed rounds resume from `Models/trainer_state.pth`; interrupted
initialization is preserved and requires a fresh output. No dependencies are
installed. This implementation has software checks only; no new data experiment
has been run or requested as part of preparing it.

## Original-paper reward control

`-OriginalPaperReward` is a separate control that reuses the same healthy CNN and
initial Context. Only a new compact query Actor is fitted, with the original
paper's discrete pre-update reward: both predictions wrong +3, only Frozen wrong
+2, only Context wrong -2, both correct 0. The skip payoff is zero. This is not
"choose the highest actual TEST loss": the true label is unavailable before a
request. TRAIN full-information labels supervise the reward, and TEST query
selection still uses only the learned pre-feedback probability. All scored
predictions precede their own feedback.

The control uses no continuous gap, future gain, forgetting harm, request cost,
reward scaling, or future-label probes. It retains causal fixed-Context routing,
batch5, replay, 20% maximum budget and independent source audits. Source retention
is still evaluated and included in the existing VAL checkpoint selection rule.
It uses the original reward **table**, not a reproduction of the original DDQN
architecture/algorithm or a proof of the continuous shared-gap zero-sum game.
The original table's +2 case rewards a correct Context prediction; consequently
it should not be described as selecting only errors or the strict hardest sample.

Output directories end in `_paper_reward`, preserving previous runs. For example:

```powershell
.\AMCAL\run_amcal_competitive_selector.ps1 -Dataset IAT -Action All -OriginalPaperReward
.\AMCAL\push_amcal_competitive_selector.ps1 -Dataset Both
```
