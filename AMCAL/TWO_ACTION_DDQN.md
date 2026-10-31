# Two-action DDQN pilot

This experiment changes only query selection on the strong two-action paper-reward
checkpoint. It does not start from the older PairOnly initial Context.

- Prediction always follows Context -> Frozen CNN before feedback is acquired.
- Action 0 skips feedback; action 1 buys a label and performs the unchanged batch5
  Context update with source replay weight 0.1 and Context learning rate 0.00025.
- The healthy Context, frozen classifier, preprocessing, compact state, TRAIN
  normalizer and Actor hidden encoder come from the parent two-action experiment.
  IAT uses its original CNN; PS uses its Light CNN.
- DDQN uses two return outputs, initially zero. Actor logits are not reused as
  discounted returns. The encoder is trainable during DDQN TRAIN.
- Query reward is the original paper table: +3 when both predictions are wrong,
  +2 when only Frozen is wrong, -2 when only Context is wrong, and 0 when both
  are correct. Skip reward is zero and requires no current label.
- DDQN uses actual successive stream states, discount 0.95, Adam learning rate
  0.0001, replay capacity 4096, batches of 32, one update every four transitions,
  and target synchronization every 15 optimizer updates. Illegal queries are
  masked in action selection and successor-action selection. Terminal transitions
  have no bootstrap. TRAIN exploration decays from 0.30 to 0.05 across epochs.
- The next state is recorded when the next input arrives, after any previous
  Context update. No future label is used in the TD target. Labels of skipped
  rows are used only by the offline metric logger.
- TRAIN and VAL use levels 0/1/3. TEST uses 5/7/10/12/15/17/20. Every episode
  resets Context, Adam, feedback memory and the floor(0.2*N) cap. DDQN/Actor
  weights stay frozen throughout VAL/TEST; only Context adapts on paid labels.
- Best epoch is selected on VAL mean online accuracy minus positive original-domain
  forgetting; lower consumption breaks exact ties. TEST does not select a model.
- Controls use exactly the same sampled rows, stream order, initial Context,
  update and reference audit: parent Actor, uncertainty with its saved TRAIN
  threshold, random querying, prefix querying and no querying. They are deployment
  controls, not independently retrained no-RL ablations.

This is a closed-loop sequential RL experiment with a competitive reward. It does
not establish a formal shared-objective min-max optimization or promise an advantage
over uncertainty. In particular, the original reward is not a direct measure of
future generalization gain or forgetting; the matched controls diagnose its value.

## Run separately

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_ddqn.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_ddqn.ps1 -Dataset PS
```

Defaults are 300 rows per domain and three epochs. Existing completed parent runs
`competitive_selector_seed42_p300_r2_v1_paper_reward` and the matching healthy
`minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_natural` must be present locally,
including their model files. No CNN retraining or package installation is performed.
The wrapper executes the new unit tests before invoking training/evaluation.

Use `-ActorRun`, `-Teacher` and `-Output` for explicit paths. Outputs default to
`protocol_runs/two_action_ddqn_seed42_p300_e3_v1`. Repeating the same command resumes
from the last completed epoch, or reuses verified completed training. A partial
epoch is recomputed. Protocol mismatches are preserved and rejected.

```powershell
.\AMCAL\push_amcal_two_action_ddqn_results.ps1 -Dataset Both
```

Results include per-epoch TRAIN traces, independent VAL summaries, selected-epoch
metadata, and seven-level TEST traces/metrics for all six query policies. Console
summaries show AC/PR/RC/F1, consumed labels, final target accuracy and original-domain
forgetting. The push script stages only JSON/CSV from the new experiment folders;
model binaries and unrelated staged files are excluded/rejected.

Tests cover DDQN target selection, budget masks, terminal targets, state transfer,
checkpoint restoration, training resume, label causality, frozen evaluation and
agreement of the matched parent-Actor environment with the existing implementation.
Code and tests were written without executing tests, training or evaluation.
