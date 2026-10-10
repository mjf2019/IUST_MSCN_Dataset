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
