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


## Original versus greedy learned online selector

No retraining is required. Both modes load the same original checkpoints and
replay, and preserve scaler, online loss, query gate, budget and online learner
updates. Only the source of online actions changes. Original uses the original
epsilon=1 path; learned uses argmax Q in eval mode (no dropout). DDQN replay
optimization remains active in both modes. Thus original is not a no-RL
ablation: its learned Q values are updated but do not choose actions.

```powershell
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --per 20 --seed 42 --selector-mode original
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --per 20 --seed 42 --selector-mode learned
```

Results are now separated in Results/original_seed42 and Results/learned_seed42.
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
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --per 20 --seed 42 --selector-mode original
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
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode original
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
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --per 20 --seed 42 --selector-mode original --query-gate no-threshold
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
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --per 20 --seed 42 --selector-mode original --query-gate no-threshold
```

Compare pre-update accuracy, weighted F1, labels consumed and query timing
between modes using the SAME new checkpoint. A single-run improvement would
not establish convergence or sample efficiency. Original shallow best-weight
snapshots and validation-mode behavior remain known legacy limitations; these
are not silently altered in this loss experiment. No training/evaluation was
run by the assistant; source syntax and call sites were reviewed statically.


## User-reported aligned-loss comparison before training-state corrections

For PER20, seed42, paper reward, local gamma0.95 and no confidence threshold,
the user ran both selector modes on the same aligned-loss checkpoint:

| Selector | Pre-update accuracy | Post-update accuracy | Post-update weighted F1 | Queries |
| --- | ---: | ---: | ---: | ---: |
| Learned | 0.580615 | 0.584343 | 0.492903 | 214 |
| Random | 0.648649 | 0.653308 | 0.607894 | 214 |

Frozen CNN accuracy was 0.383970 in both. Action counts were [124,949] for
learned and [528,545] for random (skip, apply). Action counts are not query
counts: disagreement and remaining-budget gates still apply.

Random selection remained ahead by 6.80 percentage points on pre-update
accuracy. The previous legacy-Context-loss random run reached 0.942218
pre-update accuracy; the aligned run therefore did not improve adaptation in
this pilot. Those runs used different coupled Context/selector checkpoints,
so the observed drop is not proof of a single loss term's causal effect.
No few-shot reward, budget-aware state or early-convergence objective was
implemented in these runs. The validation/snapshot defects below also remained
present in their training; new training is needed to assess the corrections.

## Correct validation mode and best-epoch checkpoints (training state v2)

This is an execution correction, not a new RL reward or a few-shot objective.
Validation now switches CNN and Context to eval mode inside a context manager,
then restores every submodule's prior mode. With no_grad already in place,
validation also leaves BatchNorm running statistics unchanged and disables
Dropout. Context training resumes normally after validation.

Best CNN, Context and selector state_dicts are deep copies, rather than tensor
aliases to models that continue training. The CNN always reloads its best
validation-loss epoch, including when it reaches the epoch limit. Context and
selector reload the same best validation-accuracy epoch before final clean-test
evaluation and saving. The selected epoch and metric are printed and saved.

Selector replay, target weights and counters are snapshotted at that same epoch,
so the checkpoint no longer combines best policy weights with later replay.
The existing online initialization remains intact: online target weights start
from loaded policy weights, the replay retains the last 500 saved experiences,
and online epsilon is set by selector mode. These are evaluation initialization
rules, not exact training resumption. Optimizer states are not saved.

All new training checkpoints record training_state_version=2. Default output
folders now append _validated_best; old folders/checkpoints/results are preserved.
Explicit --output can still evaluate older checkpoints with their matching
settings; manifests/diagnostics record their actual training-state version and
the runner prints that corrections require retraining. Mixed-version folders
are rejected. Old aliased best-epoch tensors cannot be recovered from final
checkpoints by evaluating them again.

The architecture, Context loss option, reward, transitions, gamma, gates, data,
scalers, online LR and budget are unchanged by this commit. The CNN checkpoint
can nevertheless change because its best-epoch restoration is now correct.
Comparisons to older runs therefore include a changed CNN/Context/selector
checkpoint; do not attribute their differences solely to RL or Context loss.

Continue the current aligned-loss setting with a NEW training run, then compare
learned and random selectors using that SAME checkpoint:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --seed 42
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --per 20 --seed 42 --selector-mode learned --query-gate no-threshold
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --reward-mode paper --transition-mode local --gamma 0.95 --context-loss-mode aligned --per 20 --seed 42 --selector-mode original --query-gate no-threshold
```

Default output:
protocol_runs/notebook_skip_zero_local_gamma0.95_context_aligned_validated_best.

Assess pre-update accuracy/F1, query timing and per-class coverage. A fix does
not guarantee improved performance. No model training/evaluation was run by the
assistant; source syntax, the diff and validation/snapshot paths were checked
statically.


## Selector naming and true random online ablation

Current CLI names:

| Mode | Online action source | DQN checkpoint/replay/optimizer |
| --- | --- | --- |
| original | Original notebook epsilon=1 path | Present, including online learning |
| learned | Greedy Q argmax | Present, including online learning |
| random | Independent random updater | Completely absent during evaluation |

Earlier revisions called the original notebook path --selector-mode random.
That historical name is now --selector-mode original. Old random_seed42 result
folders are preserved; new original-mode results use original_seed42. Repeating
an OLD command with random now runs the NEW no-RL ablation, not the notebook
selector. Model weights and original online numerical logic are unchanged by
this naming update.

The new implementation is amcal_burst_random.py, invoked directly or dispatched
by amcal_burst_legacy.py evaluate --selector-mode random. Dispatch occurs before
any DQN checkpoint validation/loading. Only CNN/Context checkpoint files are
required. No DQN is instantiated, no Q values or rewards are computed, no
replay is loaded/created, and no selector optimization occurs. Diagnostics
report dqn_loaded=false, dqn_updates=0, replay_size=0 and reward_computations=0.

For the user's pure random updater, --query-gate no-threshold (CLI default)
dispatches to uniform sampling: exactly min(214,N) distinct positions are
chosen without replacement from the stream using seed42. Positions depend
only on stream length, budget and seed, not features, predictions or labels.
The stream order is unchanged. A selected row is predicted before its label
is queried and Context is updated once. The legacy post-update prediction is
also recorded separately. Sampling random positions across the whole stream
is not an imposed early-block schedule. The budget is consumed by the last
selected position, then no further Context updates occur.

This pure random policy removes disagreement AND confidence filters so that
every selected row is actually updated. It changes the entire query policy;
its difference from the original gated method is not solely the removal of
DQN optimization. For a narrower matched-gate control, use --query-gate legacy:
50/50 random actions retain the original disagreement and 0.005 confidence-gap
filters, still without any DQN/replay/reward work. It may consume less than
214 labels if too few rows pass the gate; actual use is reported rather than
silently forcing extra queries. The direct runner calls its pure policy
--query-gate none and its matched policy --query-gate legacy.

Reuse the successful notebook_original CNN and Context checkpoint, the same
PER0-fitted scaler/encoder and class-count online CE weights, original Context
loss, online Adam lr0.25/weight_decay1e-4 and eval-mode Context behavior.
Every command reloads initial model weights; no online-adapted models are saved.
CNN tensors and buffers are compared before/after to verify it remains frozen.

Scope: ONLINE ablation only. The reused initial Context checkpoint was trained
with the original selector. This is not an end-to-end removal of RL from offline
training; that would require a separate random-selector training experiment.

Results are saved separately under the model folder:
Results/random_no_rl_none_seed42 for pure random updates, or
Results/random_no_rl_legacy_seed42 for the matched filter, with trace,
diagnostics, checkpoint/data hashes and an invocation manifest.

```powershell
git pull --ff-only origin experiment/amcal-protocol
# Original notebook reference (same saved successful models and threshold):
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --reward-mode original --transition-mode terminal --gamma 0.2 --context-loss-mode legacy --query-gate legacy --per 20 --seed 42 --selector-mode original
# Requested pure random updater: no RL or confidence/disagreement filters:
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --per 20 --seed 42 --selector-mode random
# Optional matched-filter control, also completely without online RL:
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --per 20 --seed 42 --selector-mode random --query-gate legacy
```

Reported skip-zero comparison: terminal skip-zero random-action run achieved
0.932898 post-update accuracy, while the later local/gamma0.95 run achieved
0.958062. Original reward runs achieved 0.952470/0.971109 at different points
and the latest original reference was 0.953402. These used different trained
checkpoints/RNG trajectories; the local-run improvement cannot be attributed
to skip reward zero alone. Compare pre-update metrics and same-checkpoint
selector controls, and do not interpret historical mode names as no-RL runs.

No training/evaluation was run by the assistant. Syntax, dispatch ordering,
loaded checkpoint paths, budget accounting and the absence of DQN calls in
the ablation runner were reviewed statically.


## Original-style console output and all-level evaluation

The no-RL random ablation now prints the same concise per-level results and
Combined Results Table columns as original: Test Accuracy, weighted Precision,
Recall, F1, Context Updates Used, DQN Updates Used and DQN Loss. DQN columns are
zero in the ablation. The frozen-CNN/pre-update/post-update diagnostic line is
retained. Detailed per-class JSON, trace and manifests remain in files instead
of flooding the console.

Omitting --per now evaluates all ten levels in BOTH modes:
0, 1, 3, 5, 7, 10, 12, 15, 17, 20.
--per 20 still evaluates only PER20. The standalone random runner has the same
default. Each level reloads the same initial CNN/Context checkpoints, creates
a fresh Context optimizer and resets the label budget; no adapted weights carry
over between levels. Original also reloads its DQN and replay per level, as before.
The random runner keeps its previous per-invocation seed initialization for each
level. Original keeps its existing random-draw history across levels, so an
all-level invocation is not numerically identical to an isolated PER20 run.

Both modes save a combined CSV, a transposed cumulative-accuracy CSV and the
same cumulative-accuracy PNG format. They use their separate result directories:
original_seed42 versus random_no_rl_none_seed42 (or random_no_rl_legacy_seed42).
Original model/learning logic, the random query policy, Context loss, scaler,
learning rate and budget are unchanged. Dispatch reuses the loaded legacy module
to avoid the duplicated pair of device announcements in random mode.

```powershell
git pull --ff-only origin experiment/amcal-protocol
# Original method on every Burst level, with its original threshold:
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --reward-mode original --transition-mode terminal --gamma 0.2 --context-loss-mode legacy --query-gate legacy --seed 42 --selector-mode original
# Pure random no-RL ablation on every Burst level:
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --seed 42 --selector-mode random
```

For the matched-filter no-RL control append --query-gate legacy to the random
command. It retains both disagreement and 0.005 confidence filters rather than
uniform fixed-budget sampling. No training/evaluation was run by the assistant;
syntax, the unchanged online loop, per-level reloads and reporting paths were
reviewed statically.


## Small online RL trial: remove both filters and gradually use learned Q

No architecture, offline training, reward, Context objective, state features,
transition semantics, gamma, replay logic/capacity, learning rate or budget is
changed in this trial. Reuse the successful notebook_original checkpoint.
Only online exploration is scheduled, and both prediction/confidence filters
are explicitly disabled for the intended random-updater comparison.

--query-gate none now removes BOTH CNN/Context disagreement and confidence-gap
checks in original/learned mode. A Context query requires action=1 and remaining
budget only. --query-gate no-threshold retains the disagreement filter, exactly
as before; --query-gate legacy retains both old filters. These names are distinct.

--selector-mode learned accepts epsilon-greedy exploration controls:
--epsilon-start, --epsilon-end, --epsilon-decay.
Defaults remain 0, 0, 1 (the existing greedy learned path).
The original notebook mode remains epsilon=1 with its original action path;
random remains a fully no-RL updater. Exploration parameters are rejected for
training or other selector modes, so they cannot silently alter those controls.

In the initial fixed trial, epsilon starts at 0.9 and is multiplied by 0.99
AFTER each successful online selector optimizer step, with a floor of 0.1.
Thus epsilon=max(0.1, 0.9*0.99**K), where K counts optimizer steps, not stream
rows, epochs or skipped samples. This is the probability of choosing a random
action, not a queried-label fraction. A random action is still uniformly skip
or apply; other decisions use argmax Q with policy dropout disabled. At zero
epsilon the prior deterministic action path consumes no extra random draw.

The original training replay is loaded; online DQN optimization, including the
original terminal replay and original rewards, remains unchanged. Natural online
skips still do not create new replay entries; no unqueried labels are used for
training reward. The unchanged hindsight skip path remains available for queried
negative-reward rows. No new sample-efficiency reward has been added.

Traces record action_epsilon before selection, action_source (policy,
exploration or original), labels_before, post-step epsilon and query timing.
Diagnostics/console distinguish actual policy versus exploration decisions
WHILE budget remains. Decisions after budget exhaustion do not count towards
that evidence of active selection. Scheduled runs have their own results and
manifest suffix so they do not overwrite the previous greedy or original runs.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --per 20 --seed 42 --selector-mode learned --query-gate none --epsilon-start 0.9 --epsilon-end 0.1 --epsilon-decay 0.99
# The existing pure-random comparator, with BOTH filters also absent:
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --per 20 --seed 42 --selector-mode random --query-gate none
```

User's pure-random PER20 reference: pre-update accuracy 0.905871,
post-update 0.911463, 214 queried labels. Use pre-update accuracy/F1, actual
label count, policy decision count and query timing for assessment. The learned
selector may use less than its ceiling; it is not forced to query all 214.
The random comparator samples exactly 214 positions uniformly without
replacement, so query timing also differs. A higher single-run score alone
does not isolate the contribution of learned Q from exploration/timing; repeat
controlled comparisons before claiming an RL benefit.

For all ten levels omit --per 20. Every level reloads the original model weights,
replay, budget and learned epsilon schedule. As before, the original/learned
runner's random-draw history carries across levels; standalone PER20 is not the
same RNG trajectory as PER20 last in an all-level run.

No training or evaluation was run by the assistant. Syntax, unchanged model/
reward/training methods, the no-filter query condition and schedule/trace paths
were reviewed statically.


## Test prediction disagreement alone, without confidence threshold

--query-gate disagreement is a new explicit, consistent option for original,
learned and random. It permits a Context query only when action=1, CNN and
Context predictions differ, and budget remains. It never checks the confidence
gap. Equal predictions mean no query and no Context update. No RL penalty is
introduced at this stage: rewards, selector optimization and all losses stay
unchanged. Any proposal to penalize an RL request on an agreement row is a
separate later experiment, after measuring this hard gate's effect.

For original/learned this is the explicit name for the disagreement-only rule
already available as no-threshold. The no-threshold flag's historical behavior
is preserved, including uniform random sampling in the no-RL mode. Use
disagreement for the present experiment to avoid that mode-dependent meaning.

In random/disagreement, the same independent 50/50 action generator used in
random/legacy is retained; only the confidence threshold is removed. Natural
skips and agreeing predictions consume no labels. A filtered stream might not
consume all 214 labels; actual usage is reported. Pure random/none still chooses
a uniform fixed-budget subset and is unchanged. Comparing pure sampling to a
filtered mode includes changes in sample eligibility and query timing.

Both commands below evaluate all ten levels from the same original initial
CNN/Context checkpoints, without modifying them or requiring retraining:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --seed 42 --selector-mode original --query-gate disagreement
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_original --seed 42 --selector-mode random --query-gate disagreement
```

Append --per 20 to restrict either command to the existing PER20 references.
Original results use original_seed42_disagreement_only; random results use
random_no_rl_disagreement_seed42. Diagnostics record confidence_gap_threshold
as null in both modes. Existing learned epsilon schedule can also be combined
with --query-gate disagreement, but changing that schedule simultaneously is
not required for this gate experiment.

Compare original/disagreement against original/legacy, and random/disagreement
against random/legacy to assess removal of the confidence threshold. The
condition-alone versus no-filter comparison can then guide a separate penalty
experiment. No training/evaluation was run by the assistant; syntax and condition/
dispatch paths were reviewed statically.

# Training disagreement penalty and configurable test budget

This is an optional, isolated training change. Use a NEW output folder; existing
`notebook_original` models remain available as the control. CNN, Context architecture,
pretraining, Context objective, original skip rewards, transitions and gamma stay unchanged.

With `--training-selector-mode disagreement`, observe CNN and Context predictions
before the batch update, without labels, dropout or BatchNorm updates. For action 1:

- Equal predictions: no Context update; store the requested action 1 in replay with
  reward `-agreement_penalty` (default **-0.2**) and unchanged local state.
- Unequal predictions: keep the original Context update and original competitive reward.
- Action 0: keep the original skip reward. ALL actions still enter training replay.

The DQN MSE/TD loss is unchanged; the new reward changes its target. The magnitude
0.2 is a configurable experimental choice, not a proven optimum. Agreement can include
both models being wrong, so this rule does not mean the example is necessarily easy.
This change adds no budget state, future-stream transition, cost-aware meta objective
or convergence guarantee. Training and evaluation report rejected requests/budgets;
use pre-update stream accuracy for the main comparison.

The penalty is applied during adversarial TRAINING only. Existing online reward/replay
logic is preserved. At evaluation, `--query-gate disagreement` retains unequal
predictions and removes only the confidence-gap threshold. Rejected online requests
consume no label budget. Budgets are maximums: gated runs may use fewer than requested.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_train_disagreement --seed 42 --training-selector-mode disagreement --agreement-penalty 0.2
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_train_disagreement --seed 42 --training-selector-mode disagreement --agreement-penalty 0.2 --selector-mode learned --query-gate disagreement --budget 214
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_train_disagreement --seed 42 --selector-mode random --query-gate disagreement --budget 214
```

Without `--per`, evaluate all ten levels independently. `--budget` is an integer
maximum per level (0..1073), default 214; for example 107 is roughly 10% of 1073.
It applies consistently to original, learned and no-RL random evaluation. Alternate
budgets have separate result folders. Only random `--query-gate none` guarantees the
exact requested number; the disagreement condition remains enforced in these commands.
The random control removes ONLINE RL only; its initial Context still comes from
the same adversarial training. Compare multiple seeds before claiming improvement.

