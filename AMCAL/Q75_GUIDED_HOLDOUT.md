# q75-guided learned DDQN: independent target TEST only

This opt-in experiment compares three immediate-feedback request policies:
LearnedBetaDDQN, Q75GuidedDDQN and Q75Only. It preserves the existing learned
beta/query checkpoint, initial CNN/Context, reward and all update settings.
No training, reward fitting, threshold recalibration or package installation is
performed. The authoring assistant did not run tests, imports or inference.

## Decision rule

Q75GuidedDDQN requests only if:
1. feedback budget remains;
2. the arriving sample's Context entropy is at least the stored TRAIN q75;
3. the unchanged learned DDQN chooses query over skip.

This is a hard query-eligibility mask, not a new reward or a soft Q bonus.
It targets low-entropy requests. It does not fix DDQN rejecting high-entropy
samples and does not force the quota to be filled. Improvement is a hypothesis.

Q75Only uses the first two conditions, without consulting DDQN values/actions.
The original learned DDQN reference uses its original request decisions.
All three predict before immediate feedback; none buffers or ranks a window.
The previously reported window control can be compared separately, retaining
its feedback-delay distinction.

The threshold is read exactly from the existing source Actor checkpoint:
the TRAIN75th percentile at development levels0/1/3. It is fixed across TEST
levels. The entropy used here is the raw Context-entropy coordinate of the
compact DDQN state (float32, normalized by log(classes)), before Q feature
standardization. This matches the representation used for TRAIN percentile
calibration. The older q75 evaluator computed entropy in float64 from logits;
near-boundary floating-point decisions need not be bit-identical. Both new
guided and q75-only policies use the same representation and threshold.

## Data, updates and metrics

Both IAT and PS use all eligible target TEST rows at levels5/7/10/12/15/17/20.
The exact same group-disjoint20% target holdout as the previous independent
evaluation is reused: common flow IDs/processed features are grouped across
levels. Where previous manifests exist, equality is required before evaluation.
Source replay purging and the source checkpoint must match the prior run too.

Only the remaining approximately80% is the online adaptation stream. Its
feedback cap is20%, nominally16% of all target rows. All rows are allocated;
there is no pilot truncation. Initial Context/Adam/feedback memory/cap reset
per method/intensity. Existing Context batch5/replay .1/LR .00025 stay unchanged.
Only Context adapts; CNN, learned Q and learned beta stay fixed.

Online AC/PR/RC/F1 are pre-feedback stream metrics. Target final metrics and
gains are on the independent same-level holdout, which never enters updates,
paid memory, source replay, threshold fitting or checkpoint selection.
Source forgetting is measured separately against the original source audit.
An earlier pretrained Context is shared across controls; these are query-policy
replacement tests, not independently retrained complete AL architectures.

A gate at inference changes the behavior distribution relative to original
DDQN training. This is a controlled diagnostic, not proof that q75 guidance
will improve performance or retain an identical unconstrained optimization
problem. The original reward, competitive Context update and frozen policy
are untouched; no new exact min-max equivalence claim is made.

## User-run commands

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_q75_guided_holdout.ps1 -Dataset IAT
    .\AMCAL\run_amcal_q75_guided_holdout.ps1 -Dataset PS

Each command runs small CPU contract tests, then21 TEST cases (three policies,
seven levels), without training. Outputs are separate:
protocol_runs/q75_guided_target_holdout20_seed42_full_v1.

Trace/metrics include the exact threshold, entropy eligibility, original DDQN
decision, final guided decision, veto count and high-entropy rejections.
Console summaries print online/final accuracy, label use under both
denominators, cap utilization and source forgetting.

    .\AMCAL\push_amcal_q75_guided_results.ps1 -Dataset Both

Only JSON/CSV manifests/results are staged. Original models, result directories
and paper files are unchanged. Completed cases resume after provenance and
metrics/trace checksum validation.

