# Original Burst notebook reproduction

amcal_burst_legacy.py combines source cells 2 (training definitions), 7 (online
test definitions), and 9 (214-label test configuration) from AMCAL.ipynb.
Training and evaluation must be invoked separately, reproducing notebook order.

Legacy behavior is intentionally preserved: gamma 0.2, terminal replay,
label-dependent skip reward, training loss clamp, inverse online loss ratio,
post-update scoring, online epsilon 1, training replay loading, refitted scaler
on the PER=0 evaluation CSV, original shallow checkpoint snapshots, and original
validation mode behavior. This is diagnostic reproduction, not scientific
validation of those choices. The recorded old outputs cannot certify which
kernel state/checkpoint bytes produced them; numerical identity is not guaranteed.
The original notebook sets no random seed.

Only operational edits: disambiguated function names to avoid notebook cell
redefinition collisions; absolute data paths and separate output folder; explicit
torch.load(weights_only=False) for locally generated legacy replay checkpoints
under PyTorch 2.7; plot windows closed after saving to avoid blocking.
No experiments or training were run by the assistant.

Train: base max200/patience30; Context max300/patience30; pretraining20.
Evaluate: all ten PER levels, each from the initial saved models, lr0.25,
threshold0.005, max1073 rows, budget214, minibatch128, replay capacity500.
Output: protocol_runs/notebook_original, separate from corrected runs.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate
```


## Observational diagnostics before changing RL

Evaluate one level with existing legacy checkpoints:

```powershell
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --per 20 --seed 42
```

--per limits evaluation; --seed optionally fixes Python/NumPy/Torch random draws
for that invocation (default remains unseeded). Restricting to one level changes
the prior random-draw history relative to evaluating all ten levels; comparing
runs requires identical arguments. This is not a guarantee of bitwise GPU determinism.
No retraining is needed. Learning actions, reward, loss, online input/scaler,
updates and scoring remain the legacy implementation.

Observers add frozen CNN predictions using the legacy scaler and a scaler fit
on the oversampled train CSV; AMCAL predictions before and after each sample's
update; per-class reports; query steps; inference-mode learned-policy Q values
and greedy action. Policy observers temporarily disable dropout and restore its
mode, without consuming random draws or changing executed actions. The CNN
observer is in eval/no-grad mode and does not change the learner's inputs.

Results/legacy_PER_20_diagnostics.json and legacy_PER_20_trace.csv provide the
comparison. The existing post-update result table is also retained. A high
legacy accuracy with epsilon=1 cannot establish learned selector effectiveness:
actions then come from random exploration, independently of Q values.
These observer additions were reviewed statically; no experiments were run.


## Random versus learned online selector

No retraining is required. Both modes load the same original checkpoints and
replay, and preserve scaler, online loss, query gate, budget and online learner
updates. Only the source of online actions changes. Random uses the original
epsilon=1 path; learned uses argmax Q in eval mode (no dropout). DDQN replay
optimization remains active in both modes. Thus random is not a frozen-selector
ablation: its learned Q values are updated but do not choose actions.

```powershell
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --per 20 --seed 42 --selector-mode random
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --per 20 --seed 42 --selector-mode learned
```

Results are separated in Results/random_seed42 and Results/learned_seed42.
Compare amcal_accuracy_before_update and before_weighted_f1 in diagnostics.json.
Different actions change later replay and RNG history despite a common initial
seed; this is a pilot comparison, not a multi-seed effectiveness claim. Selecting
only PER20 has a different draw history from running all ten levels.
No training or evaluation was run by the assistant.


## Isolated action-0 reward correction

--reward-mode original (default) retains +1.5/-1 for skip. --reward-mode paper
sets skip reward to 0 in BOTH training and online reward definitions. Action-1
reward values, architecture, scaler, transition semantics, exploration schedule,
Context loss, threshold, budget, replay and checkpoint selection are unchanged.
This name refers only to the paper's skip reward, not full paper conformity.

Paper reward defaults to protocol_runs/notebook_skip_zero, preserving the
original models. Training is required because selector targets and stored replay
rewards differ. Model checkpoints record selector_reward_mode; evaluation
rejects a different requested mode. Untagged historical checkpoints are original.

```powershell
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --reward-mode paper --seed 42
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --per 20 --seed 42 --selector-mode learned
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --per 20 --seed 42 --selector-mode random
```

Compare learned and random using the SAME new checkpoints, pre-update accuracy,
weighted F1, executed actions and labels consumed. The previous unseeded training
run is not a paired seed-42 training control; a causal claim about reward alone
would require original-reward training with the same seed/configuration too.
Zero skip reward does not by itself prove that the learned selector will improve.
All other legacy limitations documented above remain. No training/evaluation
was run by the assistant.


## Same-input local DDQN transitions

The previous replay used done=1 for every row, so the future term in the DDQN
target was always zero regardless of gamma. The opt-in --transition-mode local
stores nonterminal training transitions: apply goes to the existing same-input
post-Context state, while skip retains its existing self-state. It does not use
the next traffic row or imply forecasting future traffic.

Online real apply transitions are terminal when the label budget is exhausted
or the stream ends. Existing counterfactual skip rows are terminal only at the
stream end: that hypothetical skip would not consume a label. The learner still
has no remaining-budget feature; this is local bootstrapping with a terminal
mask, not a complete finite-budget Markov model. Legacy online skip feedback
and hindsight construction are unchanged in this isolated experiment.

--gamma is explicit, defaults to the original 0.2, and can be set to the paper's
0.95. The default transition mode is terminal; existing commands/models keep
their original behavior. With terminal transitions changing gamma has no effect
on the target. For this new trial use local with gamma 0.95 and skip reward zero.
All three checkpoints record both settings. Evaluation rejects mismatched
checkpoints, including old terminal replay. No replay is silently relabeled.

Default new output: protocol_runs/notebook_skip_zero_local_gamma0.95.
Retraining is required. The old notebook_skip_zero models/results are preserved.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --reward-mode paper --transition-mode local --gamma 0.95 --seed 42
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode learned
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode random
```

Both evaluations reload the same new checkpoints. Compare pre-update accuracy,
weighted F1, labels consumed, action counts and query timing. Diagnostics also
record gamma, transition mode, replay terminal/nonterminal counts and
bootstrap_optimizer_steps (online optimization batches with at least one
nonterminal transition and positive gamma). This counter verifies that the
future term is enabled, not that estimates or selections are accurate.

Context architecture/loss, scalers, threshold 0.005, online LR 0.25, exploration,
target synchronization, replay capacity, validation and checkpoint selection
remain unchanged. This is one diagnostic change to replay-target semantics,
not full paper conformity or a guarantee of improvement. In particular, base
CNN uncertainty remains the legacy input state, and online natural skips are
still not inserted into replay. No training or evaluation was run by the
assistant; the source and transition/checkpoint paths were reviewed statically.


## Remove the online confidence-gap threshold first

Online evaluation now defaults to --query-gate no-threshold. A query/update
requires action=1, CNN/Context prediction disagreement, and remaining budget.
The confidence-gap check is disabled with threshold=None, not reduced to an
arbitrary smaller value. Confidence gaps are still recorded for observation.

Architecture, reward, transition mode, gamma, learning rates, saved replay and
all training behavior are unchanged. Reuse the existing local/gamma0.95,
skip-zero checkpoints; no retraining is needed. This change tests the effect of
the filter without changing the initial models. It does not remove the
prediction-disagreement gate. The unrelated threshold argument of offline
evaluate_model is also unchanged.

The previous online gate remains available explicitly with --query-gate legacy.
Earlier evaluation commands in this document used the legacy threshold; append
that flag to reproduce their gate after this change.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode learned --query-gate no-threshold
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode random --query-gate no-threshold
```

New results are saved in Results/learned_seed42_no_threshold and
Results/random_seed42_no_threshold, preserving the prior thresholded results.
Diagnostics record query_gate and confidence_gap_threshold (null when disabled).
Evaluation manifests now include mode, seed and gate in the filename to avoid
overwriting the other evaluation's settings. No training or evaluation was run
by the assistant; changes were reviewed statically.


## Align Context training and online objectives

The opt-in --context-loss-mode aligned corrects two Context loss inconsistencies:
both train and online use Lo/(Le+1e-8), and the training loss is no longer
clamped to max=1. One shared helper computes:

    0.8*Le + 0.2/(1 + Lo/(Le+1e-8))

Lo is frozen base-CNN cross-entropy on raw inputs; Le is frozen-CNN
cross-entropy on Context outputs. The Context optimizer minimizes this loss.
The pretraining CE-only stage is unchanged. In legacy mode, the helper retains
the old training cap and the reversed online ratio, exactly as before.
No gradient clipping or substitute loss saturation is silently added.

In particular, clamping a scalar loss at 1 made its gradient zero when the raw
loss exceeded 1. Removing that cap permits high-loss selected samples to train
Context. A lower numerical training-loss value after capping was not evidence
of successful learning; aligned and capped loss magnitudes are not directly
comparable. Assess accuracy/F1 and uncapped task performance instead.

The selector reward, DDQN MSE/targets, local transitions, gamma, architecture,
initialization procedure, Context optimizer/LR, gates and label budget are
unchanged. This isolates the two Context corrections before adding few-shot
terms to the RL objective. No sample-efficiency reward terms are added in this
experiment; the original hardness reward remains intact.

Retrain coupled Context and RL: changing the training objective changes both
Context trajectories and the rewards/replay seen by the selector. All three
checkpoints record context_loss_mode. Evaluation rejects a mismatch; untagged
older checkpoints are legacy. Default output for the trial below is
protocol_runs/notebook_skip_zero_local_gamma0.95_context_aligned, preserving
all previous runs. Base CNN uses its original training procedure/seed.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --seed 42
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --per 20 --seed 42 --selector-mode learned --query-gate no-threshold
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --per 20 --seed 42 --selector-mode random --query-gate no-threshold
```

Compare pre-update accuracy, weighted F1, labels consumed and query timing
between modes using the SAME new checkpoint. A single-run improvement would
not establish convergence or sample efficiency. Original shallow best-weight
snapshots and validation-mode behavior remain known legacy limitations; these
are not silently altered in this loss experiment. No training/evaluation was
run by the assistant; source syntax and call sites were reviewed statically.
