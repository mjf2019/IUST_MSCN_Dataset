# Standardized Actor: TRAIN and independent VAL

This opt-in variant standardizes only the query Actor input. Frozen CNN,
Context updates, routing, value-head inputs, signed S, query threshold 0.5,
batch5, replay 0.1, learning rates, and the 20% per-domain cap are unchanged.
IAT requires the original CNN; PS requires Light CNN, as checked by the runner.

The normalizer is fitted once from the existing local TRAIN diagnostic cache.
The per-sign reservoir is weighted back to the full logged TRAIN sign prevalence
when computing means and population variances. Near-constant dimensions are
centered with scale one. These estimates use a short TRAIN trajectory and
nonzero targets, not the whole deployment distribution. No VAL/TEST state or
label enters preprocessing or the sign-prevalence estimate. Provenance and
cache checksums must match the dataset, seed and source checkpoint.

The natural control uses the unchanged original Actor loss with standardized
inputs. The balanced control applies fixed inverse-frequency weights only to
the Actor loss. It retains signed S and raw value targets, but changes Actor
weighting, so it is an auxiliary comparison, not the original shared min-max
objective. Context minimization is identical in both. No sign-target variant
is trained. Neither result is assumed to outperform uncertainty sampling.

Both controls start from the same original pretrained Context/source router,
fresh Actor and seed. Their later Context trajectories can differ because
their request decisions differ. Normalizer buffers are fixed during TRAIN,
VAL and TEST, saved inside the checkpoint, and require no cache for evaluation.

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset IAT
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset PS
.\AMCAL\push_amcal_standardized_actor.ps1 -Dataset Both
```

Run the two datasets in separate terminals if desired. Each command trains
natural then balanced, up to six epochs with the existing patience of three.
Default Action is Train: TRAIN levels 0/1/3, independent VAL rows at 0/1/3,
fresh model/optimizer/memory per validation intensity, 20% cap per intensity,
pre-feedback online scoring. VAL drives checkpoint selection; TEST is not run.
This is actual full experimental training, not the earlier cached Actor fit.
TRAIN counterfactual future probes are the existing offline supervision; the
20% cap limits committed feedback updates, not all offline TRAIN labels used.

Use `-Control Natural` or `-Control Balanced` to run controls separately.
Use `-CacheRun PATH` if the original diagnostic cache is elsewhere. Existing
completed matching runs are reused; partial/mismatched training is preserved.

Outputs are separate `minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_*`
directories. `actor_experiment.json` records normalization and control details;
`training.json` contains epoch TRAIN label consumption and per-level online VAL
metrics/forgetting; `TrainDiagnostics` contains actual target traces and their
summary. Only JSON/CSV outside Models are sent by the push helper; no cache,
weights or datasets are sent. No manuscript or previous results are changed.

After reviewing TRAIN/VAL, explicitly evaluate a chosen control if needed:

```powershell
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset IAT -Control Natural -Action Evaluate
```

## Uniform TRAIN collection, Actor-policy VAL

For a quick small-data pilot, cap each TRAIN, VAL and TEST domain at 300 rows
and use two epochs:

```powershell
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset IAT -Control Natural -UniformTraining -PilotRows 300 -Epochs 2 -Action All
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset PS -Control Natural -UniformTraining -PilotRows 300 -Epochs 2 -Action All
.\AMCAL\push_amcal_standardized_actor.ps1 -Dataset Both
```

This uses a separate `_pilot300_ep2` directory. The deterministic sampling is
applied inside each existing split before streaming, so the 20% cap is based
on the actual sampled stream (60 labels for 300 rows), not the original full
domain. TRAIN/VAL/TEST remain disjoint. Source TRAIN anchors/reference and the
clean VAL/TEST forgetting observers are also sampled with fixed role seeds.
Up to three examples per available class are retained; remaining slots are
sampled uniformly. This changes the evaluated sample, so pilot metrics must
not be substituted into full-benchmark tables. TRAIN selected IDs and roles
are recorded in `pilot_sampling.json`; TEST IDs appear in the normal traces.
The checkpoint stores the pilot protocol and evaluation restores its row cap.
Use the same PilotRows/Epochs flags to locate this run in a later Evaluate call.
Pretrained CNN/Context/router and the previously fitted TRAIN-only normalizer
are reused; only this experimental training/evaluation stage is reduced.

The original full-data experiment remains available without PilotRows:

```powershell
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset IAT -Control Natural -UniformTraining
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset PS -Control Natural -UniformTraining
.\AMCAL\push_amcal_standardized_actor.ps1 -Dataset Both
```

This is a separate training experiment, stored with the `_trainuniform` suffix.
It restarts from the same existing CNN, initial pretrained Context and source
router, fits the same fixed TRAIN-only Actor normalizer, and creates a fresh
Actor. CNN is frozen, never retrained. Uniform requests commit exactly 20%
(after flooring) feedback updates per TRAIN level. Context and routing use only
this acquired feedback; independent future/reference TRAIN probes still supply
the original signed Actor target at eligible positions. These offline targets
are full-information supervision, not counted as acquired online labels.

The Actor's original per-state max loss and Context min update are unchanged.
Because TRAIN feedback is now supplied by a fixed behavior schedule, the state
distribution differs from on-policy learning. This is a surrogate-learning
control on behavior-generated states, not an unbiased off-policy policy-gradient
claim. No importance correction is needed for the explicit per-state
full-information loss, and no rollout-return equivalence is claimed.
Uniform behavior also removes Actor control of TRAIN query locations. Retaining
the local loss formulas does not by itself establish equivalence to the original
joint on-policy min-max training game; this experiment tests experience
collection before making any such theoretical claim.

VAL always uses the Actor threshold 0.5, with fresh models/optimizers/memory at
each 0/1/3 intensity and the unchanged 20% cap. It does not use uniform requests.
Default Action Train therefore runs TRAIN and VAL only; previous checkpoints
and TEST evaluations are preserved. Six epochs and patience three are unchanged.
Use Natural alone for the first run; balanced weighting is not the chosen method.
Training logs, actual target distributions and checkpoint selection are sent by
the same push helper. Uniform TRAIN is opt-in and does not alter existing runs.

To evaluate the selected new checkpoint later, retain `-UniformTraining` to select
the new run directory and add `-Action Evaluate`. TEST then uses Actor requests
unless `-UniformQueries` is explicitly provided as a separate diagnostic control.

## Uniform-query TEST diagnostic control, no retraining

```powershell
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset IAT -Action Evaluate -UniformQueries
.\AMCAL\run_amcal_standardized_actor.ps1 -Dataset PS -Action Evaluate -UniformQueries
.\AMCAL\push_amcal_standardized_actor.ps1 -Dataset Both
```

Both previously selected natural/balanced checkpoints are evaluated by default;
`-Control Natural` can restrict this to the natural checkpoint. This option is
rejected for Train/All, so it cannot change TRAIN or VAL checkpoint selection.
Only the request decision is overridden. Requests are spread deterministically
over the entire shuffled stream by nearest-integer cumulative budget progress.
They use stream position/known length only and consume exactly floor(0.2*N)
labels per level (214 for current IAT and 177 for current PS). Prediction and
route selection still precede label access. Context adapts with the unchanged
batch5/protected replay update, and routing learns from acquired feedback just
as in the policy run. The Actor/value heads remain fixed; logged Actor request
probabilities are observational and do not drive the uniform control.

Schedule metadata changes the evaluation fingerprint, preserving policy-run
results in separate evaluation directories. The same existing push helper
includes the new JSON/CSV results and never sends models. This intervention
tests the effect of supplying distributed feedback; it does not equate label
counts to the collapsed Actor or prove optimal RL selection. Low forgetting
with zero feedback is not evidence for successful retention under adaptation.

Evaluation uses the checkpoint normalizer, resets all models at each of TEST levels
5/7/10/12/15/17/20 and preserves the 20% cap. TEST must not guide subsequent
threshold, reward or control selection. The original v2 entry point remains
unchanged when the new Actor flags are absent. Nothing is installed by scripts.

