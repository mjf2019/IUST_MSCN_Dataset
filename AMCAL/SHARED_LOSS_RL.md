# PairOnly shared-loss/DDQN pilot

This is an opt-in control, not a replacement for published PairOnly results.
It starts from the original IAT CNN or final Light PS CNN and the initial Context
in the existing PairOnly batch5 run. No CNN or initial Context is pretrained again.
No new neural-network layer is introduced. Existing entry points retain their
default behavior, serialized configuration and configuration fingerprint.

## The two changes

1. The query payoff is the **actual pre-update objective of the same batch5 step**,
   including its protected replay and modulation terms, minus the request cost.
   Skip has payoff zero. Frozen CE remains a detached reference inside the
   original Context objective. No correctness reward, original-paper discrete
   reward, gain/harm surrogate or future-label probe is mixed into this payoff.
2. Ordinary Double-DQN fits both existing `future` and `update_value` heads from
   **one real transition per row**. The old synthetic skip transitions and paired
   advantage loss are disabled in this mode. Both no-query routes share the
   skip-return critic; query return is skip return plus the existing update head.
   Prediction routing still uses the existing supervised correctness head.

The target is `reward + 0.95*(1-terminal)*target_Q(policy_argmax(next_state))`.
The query is masked after budget exhaustion. Target synchronization remains every
15 TD updates; TD Adam LR remains 0.0001. Incompatible inherited return offsets
are cleared by zeroing only the output layers of the two value heads. Routing
weights remain inherited and frozen during offline fitting; acquired feedback
can train routing during VAL/TEST, following the original deployment behavior.

There are still three external actions: 0 predicts Frozen without feedback,
1 predicts Context without feedback, 2 predicts Context and then acquires its
label and updates Context. The current true label never selects the action or
its pre-feedback prediction. Skip transitions have a known zero reward and need
no label. The finite-stream runner uses the existing one-input, unlabelled TD
lookahead; it is not a live packet-arrival latency implementation.

## What min-max means here

For the fixed-state, fixed-batch local surrogate in round k, the competitive
part uses the same payoff in opposite directions:

`min_Context max_query-policy E[sum_t gamma^t q_t (L_update(Context; B_t)-cost)]`

subject to the hard 20% label cap. `L_update` is the original batch objective,
not Context-minus-Frozen CE. Context takes a clipped Adam descent step on that
loss when queried; the DDQN approximates policy improvement for the discounted
payoff using real successors. Prediction routing is a separate cooperative,
supervised task and is not trained to make predictions worse.

This is an alternating local shared-loss game with an approximate RL solver.
It is **not** proof of a global saddle point, convergence of a partially observed
nonstationary game, or exact differentiation of future adaptation trajectories.
The critic sees summaries of history, not the entire Adam/memory state. Hardness
and future learning benefit are different objectives: selecting higher loss does
not guarantee better target accuracy, fewer labels or less forgetting. Protected
replay is retained but does not guarantee zero forgetting. The implementation
must be judged by validation and independent matched controls, not by its name.

## Protocol

* TRAIN/VAL levels: 0, 1, 3. TEST: 5, 7, 10, 12, 15, 17, 20, only with Evaluate/All.
* Default: 300 rows per domain and 3 epochs (9 short TRAIN streams). All original
  training anchors remain available apart from independent source audit rows;
  PilotRows caps adaptation/evaluation streams, not source pretraining or anchors.
* Preserve the original stream permutation. All matched methods get the same
  rows/order. The pilot does not assert preservation of within-class chronology.
* TRAIN epsilon 0.3, VAL/TEST epsilon zero; no query bonus or forced spending for
  SharedLossRL. Cap is `floor(0.2*N)` separately for every domain.
* Context initial weights, Adam and paid memory reset per level, including TRAIN.
  The global DDQN replay/optimizer persist across TRAIN streams; the initial
  Context checkpoint is never replaced by a temporary adapted Context.
* Context LR 0.00025, alpha 0.8, paid batch at most five, replay CE weight 0.1;
  source modulation weight and memory capacities are inherited unchanged.
* Reserve independent source audits before pilot sampling. Their IDs/features
  are removed from all target intensities; their features are absent from TRAIN
  anchors. Audit labels are observer-only. VAL selects checkpoints by mean online
  AC minus positive routed-system source drop. TEST does not select checkpoints.
* Controls: original PairOnly with bonus=1, Random and Uncertainty replacing both
  decisions, Context-only front-loaded budget and Context-only without feedback.
  All use the same initial CNN/Context. Uncertainty uses maximum route confidence,
  q75 calibrated on unlabelled source VAL features. These are deployment controls,
  not independently retrained no-RL ablations. Parent PairOnly uses its inherited
  utility/correctness/paired objectives and saved query cost, rather than the new
  shared payoff. All comparisons are this pilot's matched rows, not old tables.

Online AC/PR/RC/F1, labels and original-domain forgetting are printed and saved.
Final target AC is retrospective on the same stream and is descriptive. Routed
source forgetting and fixed-Context-path forgetting are separately recorded.
TRAIN traces include shared loss, query cost, TD reward and terminal flags;
training.json records query counts, discounted returns, TD loss and update counts.
Keep the full traces when evaluating whether the policy simply exhausts its cap
on the prefix. Performance superiority is not assumed.

## Windows execution

```powershell
.\AMCAL\run_amcal_shared_loss_rl.ps1 -Dataset IAT -Action All -PilotRows 300 -Epochs 3
.\AMCAL\run_amcal_shared_loss_rl.ps1 -Dataset PS -Action All -PilotRows 300 -Epochs 3
.\AMCAL\push_amcal_shared_loss_results.ps1 -Dataset Both
```

The wrapper prefers the repository venv, runs small preflight software checks and
installs nothing. The default input is the existing `paironly_batch5_trainseed42`
run for IAT or `light_ps_paironly_batch5_trainseed42` for PS, each ending in
`paironly_batch5/Models`. Use `-Source` to specify another compatible PairOnly run.
Source architectures and file hashes are verified; old models/results are not
overwritten. New outputs are under `protocol_runs/shared_loss_rl_*`.

Repeat the same command to resume from the last completed epoch. An interrupted
epoch repeats from its saved boundary; a failure before the first saved epoch
preserves partial output and requires a fresh `-Output`. Models and optimizer
states stay local; the push script stages only this experiment's JSON/CSV files.
Software, source data and model hashes are recorded. This pilot has software
verification only until the user runs it; its results must not be substituted
into the manuscript's existing tables without protocol reconciliation.
