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


## Selection-cost reward experiment (agreement idea disabled)

Use `--reward-mode paper --training-selector-mode original --selection-cost LAMBDA`.
The ONLY algorithm change is subtracting LAMBDA from action-1 rewards, both offline
and in the existing queried-action online replay. Skip rewards remain exactly zero.
The TD/MSE loss, original Context loss/update eligibility, network/state, terminal
transitions, gamma, dataset and online disagreement gate are unchanged. The earlier
agreement penalty/training gate is disabled. Natural online skips/gate rejections
still do not create replay experiences; this experiment does not change that rule.

Default cost 0 preserves earlier runs/checkpoints. Positive cost requires fresh
training: evaluation verifies checkpoint cost, preventing mixed reward scales in
loaded replay. Random control does not compute rewards: omit the cost flag and
reuse the cost-trained CNN/Context. It ablates online RL, not offline RL.

A first diagnostic value is **2.0**, not an established optimum. Original action-1
rewards [3, 2, -2, 0] then become [1, 0, -4, -2]; skip is 0. This removes the positive
incentive for many requests without replacing the original hardness categories.
Both-wrong cases still earn +1, so this alone cannot guarantee sample efficiency.
Tune any further cost on validation, not the reported attack test performance.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_skip_zero_cost2_seed42 --seed 42 --reward-mode paper --training-selector-mode original --selection-cost 2
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_skip_zero_cost2_seed42 --seed 42 --reward-mode paper --training-selector-mode original --selection-cost 2 --selector-mode learned --query-gate disagreement --budget 214 --epsilon-start 0.05 --epsilon-end 0.05
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_skip_zero_cost2_seed42 --seed 42 --selector-mode random --query-gate disagreement --budget 214
```

All levels run by default and reload initial checkpoints independently. Record actual
queries, query timing, and pre-update accuracy versus both the cost-0 learned control
and the same-checkpoint random control. Fewer labels alone do not prove improvement.
Neither a budget-state input nor a convergence-gain/meta-learning reward is added.

## Offline epsilon schedule: isolated convergence diagnostic

Only the offline exploration schedule changes in this experiment. Historical defaults
remain start=1, end=0.01, decay=0.995 for existing commands. New TRAIN-only flags
`--train-epsilon-start 1 --train-epsilon-end 0.05 --train-epsilon-decay 0.9`
use epsilon `max(0.05, 0.9**completed_epochs)`. Epsilon is constant within an epoch;
the action function retains its original batch-level exploration draw. The floor is
reached after 29 completed epochs, so epoch 30 uses 0.05. Epoch 16 uses ~0.2059,
whereas historical epoch 16 uses ~0.9276. Earlier best snapshots may still have
larger epsilon: the console reports used/next epsilon and restored snapshot epsilon.

No changes to reward, cost, architecture, TD/MSE loss, terminal/local semantics,
replay composition/capacity, optimizer, Context training or best-snapshot selection.
Best snapshots are still chosen by Context validation accuracy, not selector quality.
A faster schedule is an experimental setting, not a guarantee of better Q learning.
Training schedule metadata is saved in checkpoints and manifests. Online epsilon
remains separately specified by `--epsilon-start/end/decay`; do not pass training
schedule flags to evaluate. Evaluation reads schedule metadata from checkpoints.

For an isolated comparison, rerun an EXISTING cost rather than changing the cost
as well. This example fixes cost **0.42**, seed **42** and online epsilon **0.25**:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --reward-mode paper --selection-cost 0.42 --train-epsilon-start 1 --train-epsilon-end 0.05 --train-epsilon-decay 0.9
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --reward-mode paper --selection-cost 0.42 --selector-mode learned --query-gate none --budget 214 --epsilon-start 0.25 --epsilon-end 0.25
```

Compare with the OLD cost-0.42 checkpoint evaluated using the identical online
flags, including `--query-gate none`. Earlier cost-0.42/epsilon-0.25 results used
`disagreement`; they are not an isolated baseline for the no-filter command above.
All ten levels reload initial weights independently. Inspect pre-update accuracy,
actual queries and query timing. This change does not validate convergence solely
by a reduced DQN training loss. No models were executed during implementation.

## Diagnostic linear release of ONLINE label budget

`--budget-schedule linear` changes only when the existing budget becomes available.
At 1-based step t of N known stream rows, the cumulative allowance is
`ceil(B*t/N)`, capped at B. For B=214,N=1073: step 1 allows 1 label, step 546
allows 109, and the final step allows 214. Unused released labels carry forward.
Requests at the current limit get no Context update, label query, reward/replay
insertion or selector optimizer step. They are not converted to skip rewards.
Existing action choices, epsilon schedule and overall exhaustion freeze remain.
This needs only N, not future features, labels or class boundaries. It preserves
row order and all weights, rewards, loss, query filters, replay/optimizer settings.
It imposes a diagnostic budget controller, NOT learned budget management or
proof of a superior RL policy. Full budget remains the default for older commands.

No retraining is needed. Use the SAME fast-epsilon/cost-0.42 models and online
settings as the supplied learned trace. First compare PER0 directly:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --reward-mode paper --selection-cost 0.42 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --per 0
```

Remove `--per 0` to evaluate all ten levels, each independently. Linear-release
results have a separate `_budget_linear` folder suffix. Trace includes cumulative
released budget and requests blocked by release; JSON/manifest records the rule.
The total budget is still a ceiling; RL may skip and use fewer than 214 labels.
There is no final forcing of updates and no retroactive replay of blocked requests.
Compare pre-update accuracy, class coverage and query timing with the full-budget
learned baseline and uniform no-RL random control on the same checkpoints. The
random runner with `none` and full schedule is uniformly spread. A later matched
control also accepts the linear flag (see below). Model training and evaluation were not run here.

## Strict reproducibility diagnostic (same checkpoint; no retraining)

`--deterministic --seed 42` enables strict deterministic algorithms in PyTorch,
cuDNN deterministic mode, disables cuDNN benchmarking, and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA context creation if no setting exists.
Unsupported nondeterministic operations raise an error rather than silently falling
back. These controls apply to legacy and random runners. This tests a plausible
source of variation; it does NOT establish CUDA as the cause of prior differences.
No changes to losses/rewards, models, action logic, replay, epsilon or budget rule.

In strict mode, evaluation resets Python/NumPy/Torch/CUDA seed BEFORE model
construction at EVERY level, making a level independent of which levels preceded it.
Previous default commands preserve their legacy across-level RNG sequence.
Learned action inference already disables policy dropout; its optimization still
uses original training dropout, now seeded reproducibly. Context/CNN stay in eval.

New result folders contain `_deterministic`. Per-level diagnostics record runtime
flags/versions, hashes of the three checkpoints and the input CSV, and a trace hash.
The console prints `Trace SHA256`; compare it across two fresh-process repeats.
Same runtime, GPU and input/checkpoint bytes are required; no cross-platform or
cross-version identity is promised. If CUDA raises an unsupported-operation error,
report it instead of switching strict mode off. No model was run during this edit.

Run the following command twice in separate Python invocations (no retraining):

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --per 0
```

Keep console output and compare trace hashes. Identical accuracy without identical
traces is weaker evidence. These files overwrite repeated identical configurations;
copy a trace if both need preserving. If differences remain, check recorded input/
checkpoint hashes and runtime fields before changing algorithm hyperparameters.

## Matched no-RL random control with linear budget release

The no-RL random control now supports `--budget-schedule linear`. It draws a
50/50 independent skip/select request at each step and executes a requested update
only below the SAME cumulative cap as learned: ceil(B*t/N), unused capacity carried
forward. With `--query-gate none`, no prediction/confidence filters are involved.
It does not instantiate/read DQN, compute Q/rewards, use replay, or optimize RL.
CNN/Context checkpoints, input order, online Context loss and optimizer are shared
with learned. This isolates the selector comparison under a matched external cap;
the cap itself is not learned. Coin flips are independent of labels/predictions.

Default `full + none` preserves uniform fixed-budget sampling without replacement.
Linear uses random requests, not preselected uniform positions or forced updates.
Both learned and linear random have a budget CEILING: a late skip can leave it
partly unused. Actual query counts must be reported, not assumed to be exactly 214.
The NumPy random request stream is separate from learned's Python/Torch RNG and
DQN optimization; matched cap does not mean identical random draws/actions.

Reuse existing weights, no training. Run this control for all ten levels:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --deterministic --selector-mode random --query-gate none --budget 214 --budget-schedule linear
```

Results: `Results/random_no_rl_none_seed42_deterministic_budget_linear`.
Trace includes release limit, labels before query and release-blocked requests.
Compare with existing learned deterministic/linear run; use pre-update stream
metrics for primary reporting. Full-schedule random result folders remain separate.

## Online real-skip replay diagnostic

`--online-skip-replay` is optional and applies only to learned evaluation with paper
rewards. Fresh training is NOT needed; existing initial replay/weights are reused.
A real action 0 is inserted with reward exactly 0, current state as local next state,
and the existing terminal/local done rule. No label is read for this reward, no
Context optimizer step occurs, and label usage does not increase. Action-1 requests
blocked by budget release, exhaustion or query gates are NOT converted into skips.
After total label-budget exhaustion, selector learning remains frozen as before.
A temporary linear-release cap does not prevent recording an actual action-0 skip.

Each inserted skip triggers the existing selector optimize call, target-sync counter
and online epsilon-decay-on-success rule. Thus DQN optimizer steps can exceed 214;
214 still limits queried labels/Context updates, not computation. Epsilon is held
at 0.05 in this experiment. Existing queried-action replay and hindsight entries,
MSE loss, reward categories/cost, architecture and Context objective are unchanged.
Online trace logs `skip_replay_added`; diagnostics/console count real skip entries.
Average Reward retains its earlier queried-action-only scope; skip-zero rewards
are visible separately in the trace. Default flag-off behavior is preserved.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay
```

All levels run independently. New result folders include `_skip_replay`, leaving
previous runs separate. Compare to learned flag-off under identical deterministic,
linear budget and epsilon settings, and the matched linear random control. This
changes both replay coverage AND when online DQN is optimized; it is not an isolated
fixed-computation ablation of replay composition. Record optimizer counts alongside
pre-update accuracy and actual queries. No model was executed during this edit.

## Skip replay with query-only optimizer cadence

Add `--online-skip-replay --skip-optimizer-mode queries` to record label-free zero
rewards for actual skips, but do not optimize DQN or decay epsilon on those skips.
The existing optimize call after each executed Context/query update is unchanged.
With the loaded replay of 500 and minibatch 128, 214 executed queries produce 214
optimizer steps. If fewer labels are used or replay is too small, actual optimizer
counts may be lower; the console reports them. Default `all` retains the prior skip
experiment. `queries` requires `--online-skip-replay` (learned/paper evaluation).

Skip entries remain available in mixed replay minibatches at subsequent queries.
Global exhaustion freeze, state/reward/cost, target-sync replay-insertion counter,
Context training, architecture and online query filters remain unchanged. No final
forced optimization is added; late entries may receive no further training. This
controls optimizer count versus prior runs but is not an identical-RNG ablation:
replay composition/sampling and target-sync timing can differ. No retraining needed.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_fast_eps_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay --skip-optimizer-mode queries
```

All levels run independently. New result folders include
`_skip_replay_query_optimizer`. Compare pre-update accuracy/label use with no-skip
learned and same-cap random, plus the prior all-optimizer skip-replay run. Training/
evaluation was not executed during this edit; only source structure was verified.

## Context-modulated uncertainty in selector state (paper-alignment experiment)

`--selector-state context` makes the scalar state uncertainty label-free
`1 - max_k softmax(CNN(Context(x)))_k`, both BEFORE offline action selection and
BEFORE online action selection. Features remain original x: state width is still
input_length+1 (101 on Burst). Paper Eq.13 and Algorithm1 use the modulated input;
Eq.13 writes 1-softmax without specifying a scalar reduction. This experiment keeps
the existing scalar 1-max convention; that reduction needs explicit manuscript
clarification rather than claiming the vector equation is implemented literally.

Previously the current state used CNN(x), while post-update next state used
CNN(Context(x)). Context-mode current and post-update states now use the same
source. Offline state observations use temporary eval/no-grad mode to avoid
extra BatchNorm updates and dropout-dependent uncertainty, then restore training
flags. Context updates/reward computation remain in their original modes. Online
Context-mode pre-update logits are computed once before action choice and reused
for scoring/gates/reward. CNN(raw x) still provides base predictions and hardness
reward comparisons. No label enters the state observation.

Default `base` preserves earlier training/evaluation semantics. Context-mode needs
fresh training: state input meanings differ even though tensor dimensions match.
All three saved artifacts tag the state source; evaluation checks it and refuses
old/mixed checkpoints/replay with a different definition. The no-RL random control
uses the same trained CNN/Context but never constructs a selector or consumes state;
its diagnostics report the offline state mode from Context checkpoint metadata.

Architecture, reward/selection cost, losses, gamma/terminal semantics, replay sizes,
epsilon schedules and online external budget controller are unchanged. This fixes
an implementation mismatch; it does not establish convergence or learned budget
management. Old checkpoints remain separate. Train strictly reproducibly for the
new run; old offline training was not strict, so for a fully isolated state ablation
also train a fresh base-mode control under the same strict settings.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_state_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --train-epsilon-start 1 --train-epsilon-end 0.05 --train-epsilon-decay 0.9
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_state_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay --skip-optimizer-mode queries
```

All levels reload initial weights independently. For the matched no-RL control:

```powershell
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_state_seed42 --seed 42 --deterministic --selector-mode random --query-gate none --budget 214 --budget-schedule linear
```

Trace logs state source and scalar uncertainty. Use pre-update accuracy/actual label
use and matched-control results. No training or evaluation was executed here.

## Online reward timing: post-update alignment diagnostic

Offline training computes competitive reward using Context predictions after the
batch's Context update. Legacy online reward used the BEFORE-update prediction
although its computation occurred after the optimizer step. New optional flag
`--online-reward-timing post` observes Context prediction AFTER the queried sample's
update for action-1 reward, keeping the frozen/raw CNN prediction as comparator.
The reward categories, cost and skip reward are unchanged. Default `pre` preserves
previous experiments; output folders include `_reward_post` when enabled.

No label is queried for skipped or budget/gate-blocked requests. Reward uses the
label already acquired for the actual Context update. Pre-update predictions are
still appended for primary scoring before that update. Legacy post-update accuracy
is retained separately; it is not substituted as the main scientific metric.
Trace records `reward_timing` and `reward_context_prediction`, JSON/manifest records
both offline and online timing. Skip reward stays label-free zero with the current
paper/real-skip-replay settings. Negative-query hindsight skip entries are preserved.

No retraining needed: existing checkpoints/replay were trained with post-update
reward timing. This aligns timing, not all numerical modes: offline Context reward
observations still use training mode/batch updates, while online Context stays eval
and updates single samples. Architecture, state, gamma/terminal semantics, costs,
replay sizing, epsilon and external budget cap remain unchanged.

Test the SAME context-state/train-epsilon-0.2 checkpoint:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_state_train_eps0p2_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay --skip-optimizer-mode queries --online-reward-timing post
```

All ten levels run independently. Compare with prior pre-timing learned and the
same-checkpoint matched linear random control. Different rewards alter later action
and query trajectories, so accuracy need not increase. No model was executed here.

## Add queried-update CE gain to competitive reward

New optional `--reward-gain-weight beta` adds `beta * (CE_before - CE_after)` to
the EXISTING action-1 competitive reward after its cost. Paper skip remains exactly
zero. No architecture/state/Context objective changes. Gain uses UNWEIGHTED CE per
example in both training and online, keeping class-count weighting conventions in
the Context objective separate. Predictions are detached/no-grad and observed in
eval mode for this gain, avoiding dropout/extra BatchNorm observation updates.
The gain is raw, NOT clipped; an update that increases CE receives a negative term.
Beta=1 is an experimental starting value, not a validated optimum.

Offline, pre-update logits already observed for Context-mode state provide CE_before;
a post-update eval observation provides CE_after. The gain is added only to actual
selected training examples, not skips. Online, it uses the pre-score logits and post
Context-update logits for the queried sample, with its already acquired label. No
extra labels or future samples are used. Score-before-query-update remains primary.
Trace/diagnostics log raw CE gain; training console logs mean selected CE gain.

This measures fitting improvement on the queried training example; it does not
prove improvement on later unseen samples, meta-learning or fast convergence. The
competitive hardness term remains intact, so this is additive reward shaping, not
replacement by pure cooperative utility. Default beta=0 preserves prior runs.
Positive beta requires paper reward, Context-mode state, original training selector,
and post-timing evaluation. Fresh training is required because old stored replay
has different reward targets. Checkpoint compatibility guards prevent mixing them.

Keep cost=0.42, offline epsilon floor=0.2, online epsilon=0.05 and all prior controls.
Train in a NEW directory:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_gain1_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --reward-gain-weight 1 --train-epsilon-start 1 --train-epsilon-end 0.2 --train-epsilon-decay 0.9
foreach ($level in @(0, 7, 20)) {
    python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_cost0p42_context_gain1_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --reward-gain-weight 1 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay --skip-optimizer-mode queries --online-reward-timing post --per $level
}
```

The loop starts independent invocations for pilot levels 0/7/20; retain each console
result. Each level has distinct trace/diagnostics files, but single-level summary
CSV/PNG is overwritten by the next invocation. For one aggregate all-level table,
run evaluate without `--per` after the pilot. Do not choose beta/cost based on final
test scores; use validation. Same-checkpoint matched random control omits the gain
flag because no RL/reward exists in that runner. No model was executed here.

## Select raw adversarial snapshots using a copy-only validation adaptation episode

Optional TRAIN-only `--checkpoint-selection adaptation` selects a checkpoint by
budgeted pre-update accuracy on the EXISTING validation CSV. Fewer queried labels
break exact accuracy ties; no scalar tradeoff coefficient is added. Default
`context` preserves the original clean Context validation-accuracy criterion.
The base CNN checkpoint is still selected by its ordinary validation CE.

Defaults: probe at epoch 1 and every 10 epochs, a fixed random subset of at most
200 validation rows (original row order retained), 20% label cap (40 at N=200), seed
42. No training/test rows, attack test files, new split or synthetic attacks are used.
Input tensors have the train-fitted preprocessing from the existing loader.
This tests adaptation on ordinary validation data, not robustness under a held-out
adversarial domain. Its suitability for that claim needs separate validation data.

For each candidate, deepcopy the frozen CNN and Context and instantiate a SEPARATE
DQN initialized from the candidate policy, with target initialized from policy like
online loading. Load only the last 500 replay entries, minibatch 128, Adam selector
1e-4, Context Adam 0.25. Use current Context state, post-update competitive+gain reward,
fixed epsilon 0.05, no filters, linear budget release, natural skip-zero replay and
query-only DQN optimizer cadence. Policy is eval for actions, train for optimization.
Observer scores before each query; only queried labels reach adaptation/reward.
All Python/NumPy/Torch/CUDA RNG states are restored after the probe, including on
exceptions. Original models/replay/counters/optimizers are not passed to updates.

The SAVED snapshot is the raw training candidate, not its validation-adapted copy.
History is saved at `Models/adaptation_validation_history.json`, including candidate
epoch, pre-update accuracy, labels, subset indices and validation-file hash. Context
and selector checkpoints record the selection criterion and selected probe metrics.
No checkpoint compatibility guard forces users to repeat training-selection flags
at inference. Stop patience is still measured in epochs since the last improvement
of the active selection metric; unscheduled epochs do not create new candidates.
Changing selection may change early-stop time/final raw snapshots; training update
formulas themselves remain unchanged. Budget release remains an external controller.

Fresh training is required to compare this checkpoint-selection experiment:

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_context_gain1_adaptation_validated_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --reward-gain-weight 1 --train-epsilon-start 1 --train-epsilon-end 0.2 --train-epsilon-decay 0.9 --checkpoint-selection adaptation --validation-probe-every 10 --validation-probe-samples 200
foreach ($level in @(0, 7, 20)) {
    python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate --output AMCAL/ANT_Burst_TEST/protocol_runs/notebook_context_gain1_adaptation_validated_seed42 --seed 42 --deterministic --reward-mode paper --selection-cost 0.42 --selector-state context --reward-gain-weight 1 --selector-mode learned --query-gate none --budget 214 --budget-schedule linear --epsilon-start 0.05 --epsilon-end 0.05 --online-skip-replay --skip-optimizer-mode queries --online-reward-timing post --per $level
}
```

The probe settings are logged in the training manifest; trial evaluation settings
match that probe except the larger real stream/budget. Choose hyperparameters on
validation, not final attack test scores. This is model selection, not training a
meta-objective or learned budget management. No training, forward pass or probe was
executed during implementation; source/isolation structure was checked only.

# Controlled online Context learning-rate comparison

`amcal_burst_legacy.py evaluate --online-lr VALUE` now sets the Context Adam
learning rate for original, learned, and no-RL random selection. The default
is still 0.25. This option does not change offline training, DQN learning rate,
architecture, rewards, gates, budget release, preprocessing, or checkpoint
selection. It does not require retraining. Values must be finite and positive;
nondefault values are rejected for `train`.

Use the existing `notebook_cost0p42_context_gain1_seed42` checkpoints to compare
0.25 against 0.01 with identical seeds, budgets, and schedules. Keep learned
evaluation's reward/state/gain settings matched to those checkpoints. The
random dispatch forwards the same learning rate without constructing a DQN.
Nondefault rates add `_lrVALUE` to result directories and learned manifests,
so these results do not overwrite the 0.25 baseline. Per-level diagnostics
record `online_lr`; learned evaluation manifests also record it.

Start with levels 0, 7, and 20 as an exploratory diagnostic, using pre-update
accuracy for both modes. A higher score is not guaranteed, and these attack
test levels should not be used to select final hyperparameters. Confirm any
eventual learning-rate choice on validation data and multiple seeds.


# Experimental nonlocal training reward and longer training

`--reward-generalization-weight BETA` (default 0) enables a separate experiment.
It replaces the same-selected-sample CE-gain bonus: local `--reward-gain-weight`
must be 0. Paper skip reward remains zero; original competitive action-1 reward
and selection cost remain. CNN/Context/DQN architecture and Context objective
are unchanged. This is a change to the training reward, not just a runtime fix.

After CNN training, reserve approximately 10% of each class's distinct feature
groups from the existing oversampled TRAIN tensors for reward observations.
All exact feature duplicates in those groups are excluded from Context
pretraining and Context adversarial optimizer batches. One representative per
reserved group supplies reference observations. The CNN has already seen the
training file, so this is held out from Context gradient updates, not an unseen
CNN test. Validation/test files, splits, and saved CSVs are untouched. Record
indices/counts/seed in `Models/generalization_partition.json`. Conflicting labels
within identical feature groups or insufficient groups fail explicitly.

For each batch containing selected examples, sample up to 16 reference groups
per class. Observe class-balanced, unweighted CE in eval/no-grad mode on the
SAME reference mini-batch immediately before and after the Context optimizer
step. For each executed action-1 in that batch:

`reward = competitive_reward - selection_cost + BETA * (reference_CE_before - reference_CE_after)`

Skips get zero. The bonus is shared batch credit: it measures the selected
group's effect and does NOT identify each individual's causal contribution.
It is a one-step nonlocal reward; it does not by itself establish formal
meta-learning or long-horizon Bellman learning. Reward reference labels are
used repeatedly during training, so normal validation remains independent
checkpoint selection. No attacked test labels are used for this reward.

`--context-epochs`, `--context-patience`, and `--context-min-epochs` configure
adversarial training only. Defaults stay 300/30/0. A pilot may use 600/80/100:
no early stop before epoch 100, patience 80 since the best validation score,
maximum 600, always restore the best synchronized validation snapshot. For the
new reward only, the Context cosine scheduler spans pretraining plus the
configured maximum, avoiding its historical 100-step oscillation. More epochs
do not guarantee better validation or convergence.

Fresh training in a new folder is required, with paper reward, context state,
original training selection, local gain weight 0, and normal (`context`)
checkpoint selection. Artifacts carry `reward_generalization_weight`; learned
evaluation rejects mismatches to prevent mixing objectives/replay.

Initially evaluate with `--freeze-online-selector`, `--selector-mode learned`,
and matching `--reward-generalization-weight`. Keep online skip replay OFF.
The policy uses learned Q values and any requested epsilon exploration, but
its Q/target/replay do not update. Context still adapts within the same budget.
This prevents overwriting the nonlocal objective with a different online
reward, since unqueried independent labels are unavailable. Any logged online
competitive reward is diagnostic only. This isolates offline learned
selection, not online RL adaptation. Random remains an online no-RL ablation
using the same CNN/Context checkpoints. Compare pre-update scores with matched
rates, budgets, schedules, and seeds. Do not choose final hyperparameters from
attack test performance or assume a target accuracy will be reached.


# Online RL with previously paid reference labels

For generalization-trained checkpoints, an additional diagnostic alternative
to freezing the online selector is now available:
`--online-generalization-reference paid-history --online-reward-timing post`.
Use learned selection and matching positive `--reward-generalization-weight`;
omit `--freeze-online-selector`. No retraining is needed for the existing
`notebook_generalization1_long_seed42` checkpoint.

Maintain up to 16 distinct already-queried feature rows per class in a fresh
per-level cache. After action/gates/budget permit a query, observe eval/no-grad
class-balanced CE on that prior cache before/after the current Context update.
The same prior-cache rows are used for both observations; exact duplicates of
the current feature row are excluded. Add `BETA * (CE_before - CE_after)` to
the competitive reward minus selection cost. With an empty cache the bonus
is absent. Only after reward observation is the current paid label admitted
to the cache. Skips/gate rejections/budget-blocked requests never admit labels;
paper skip reward remains zero. The global unique-label budget is unchanged.

These references have ALREADY been used for Context optimization. Therefore
the online bonus measures retention/rehearsal on previous queried examples,
NOT independent held-out generalization as in offline training. It is related
to the new objective but not an identical reward distribution. No future or
unqueried stream labels enter reference reward computations; this cannot
establish meta-learning or unseen-data generalization by itself.

Online replay/Q optimization resumes with the usual query updates. Optional
`--online-skip-replay --skip-optimizer-mode queries` records zero-reward real
skips without extra optimizer steps. Cached-label evaluation adds computation,
not label queries. CNN remains fixed; Context adapts only on executed queries.
History is bounded, class-balanced in its loss, and freshly empty each level.
Diagnostics/trace record `reference_ce_gain`, reference-row counts, source,
and objective scope. Result directories/manifests have
`_reference_paid_history`, keeping the frozen-selector experiment separate.

This new test changes online selector updating/reference reward while retaining
the same saved models, rates, gates, budget schedule, and seed. Compare against
the frozen selector and the SAME saved-model no-RL random control, using
pre-update scores. Improvements are not guaranteed. All previous defaults and
offline training, partition, reward formulas, architecture are unchanged.

