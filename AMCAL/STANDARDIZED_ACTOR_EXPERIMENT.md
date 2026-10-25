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

## Uniform-query diagnostic control, no retraining

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

