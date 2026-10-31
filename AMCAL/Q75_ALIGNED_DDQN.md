# q75-aligned DDQN continuation

Separate experimental continuation of the existing learned-beta query policy.
No original source files, models, data or earlier results are changed.
No tests, imports, training or inference were executed by the authoring assistant.

## Minimal method change

The same TRAIN q75 gate is now present in:
* greedy request selection;
* epsilon exploration (illegal queries are never explored);
* the successor action mask, applied before DDQN online argmax/target gather.

A low-entropy query cannot be stored as a real TRAIN action. A below-q75 next
state cannot bootstrap through the query action even if that action has huge Q.
No invalid-action reward or additional coefficient is introduced.

The selected learned beta is frozen. Reward remains original paper reward plus
that beta times measured same-domain independent TRAIN gain, with zero skip
reward. Original Context gap minimization, batch5 replay, LR .00025 and source
replay weight .1 remain unchanged. CNN and initial Context weights are fixed.
Temporary Contexts adapt during each TRAIN episode and are discarded afterward.

Q starts from the existing learned checkpoint; the target network is synchronized.
Q Adam and replay start fresh, avoiding illegal/stale transitions from the prior
behavior. All prior source TRAIN streams, probes, feature exclusions and reserved
meta rows/replay pool are reproduced and hash-checked. No new META trajectories
or beta updates run. TRAIN remains the original300 rows per level0/1/3.
Default continuation: three additional Q epochs, same DDQN LR .0001, gamma .95,
batch32, target interval15, replay4096 and epsilon .30 to .05.

## Selection and resumption

VAL selects the epoch by mean pre-feedback online AC minus positive original
source forgetting, with fewer labels breaking ties. The current inference-only
hybrid is included as epoch0. If continuation does not improve that VAL ranking,
epoch0 is retained. SELECTED_EPOCH=0 is printed and saved explicitly; it is not
presented as a successful training improvement.

Train actions never open TEST data. Completed epoch snapshots include the exact
q75 mask, Q optimizer, target, replay and RNG. Mismatched resume masks are rejected.
Original checkpoint files remain read-only throughout.

## Independent target evaluation

Default All then compares:
* AlignedQ75DDQN: VAL-selected masked continuation;
* InferenceOnlyQ75: original learned Q with the inference-only gate;
* LearnedBetaDDQN: original learned Q without the gate;
* Q75Only: saved TRAIN threshold without a Q decision.

All have immediate feedback after prediction. CNN/initial Context/source replay
and data order match. Each level/method resets Context/Adam/feedback memory/cap.
The same group-disjoint20% same-level holdout is reused; remaining80% is the
adaptation stream, with feedback cap20% of stream (nominal16% of all rows).
All eligible rows are allocated; TEST is not truncated. Held-out IDs/features
cannot enter streams or replay across intensities. Where prior manifests exist,
target split, source checkpoint and replay equality are checked.
Online metrics describe the stream; final metrics/gains describe only independent
target holdout. Original-domain forgetting is measured separately.

This tests train/eval action-mask alignment. Additional Q optimization also changes
the experiment; causal attribution solely to the mask would need a matched
unmasked continuation control. Existing local competitive Context learning and
closed-loop DDQN remain, without an exact common saddle-equivalence claim.
The hard gate can still exclude useful low-entropy samples; gains are not guaranteed.

## Commands on the user's machine

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_q75_aligned_ddqn.ps1 -Dataset IAT
    .\AMCAL\run_amcal_q75_aligned_ddqn.ps1 -Dataset PS

Each wrapper runs small CPU contract tests, then short Q training/VAL and28
independent TEST cases. No CNN retraining. Default output:
protocol_runs/q75_aligned_ddqn_seed42_e3_full_v1.

Optional separate phases (same Epochs value):

    .\AMCAL\run_amcal_q75_aligned_ddqn.ps1 -Dataset IAT -Action Train
    .\AMCAL\run_amcal_q75_aligned_ddqn.ps1 -Dataset IAT -Action Evaluate

    .\AMCAL\push_amcal_q75_aligned_results.ps1 -Dataset Both

Only JSON/CSV diagnostics/results are staged; Models binaries stay local.
Console prints TRAIN/VAL progress, selected epoch, exact q75, beta, online/holdout
accuracy, label use under both denominators and source forgetting.

