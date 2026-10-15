# Optional three-action routing experiment

This is a separate extension, not the unchanged paper reproduction. The legacy
runner and its two-action checkpoints are unchanged.

Actions:

* 0: predict with Frozen; do not buy a label.
* 1: predict with Context followed by Frozen; do not buy a label.
* 2: predict with Context before feedback, buy a label, then update Context.

With q=+1 for a correct prediction and -1 otherwise:

    r0 = q(Frozen(x), y)
    r1 = q(Frozen(Context_before(x)), y)
    gain = (CE_before - CE_after)/(CE_before + CE_after + 1e-8)
    r2 = q(Frozen(Context_before(x)), y) + gain_weight*gain - selection_cost

CE is the unweighted CE on the queried example. The Context optimizer keeps the
source checkpoint's original/aligned objective. The gain is an immediate proxy;
it does not establish improvement on future samples or attack detection.

The state contains raw standardized features, both models' probability vectors,
and remaining total budget fraction. It never contains a true label or an attack
indicator. Both model outputs are computed for routing; selecting the Frozen
route means its prediction is returned, not that Context computation is avoided.

Training starts from an existing CNN/Context run and initializes a fresh
three-output DQN. Context resets to the source snapshot at each training episode;
the router persists. Source training labels provide rewards for actions 0 and 1.
Only performed action-2 updates provide action-2 rewards. Counterfactual action
0/1 transitions preserve Context and budget. DDQN transitions use the next stream
sample. The best joint Context/router snapshot is selected using the separate
source validation set, evaluated without adaptation. No attack test files are
used for training or checkpoint selection. This is source-domain training, not
meta-training on adversarial episodes; out-of-domain routing remains empirical.

Evaluation reloads all three models and fresh optimizers for each level. It
reports only predictions BEFORE feedback. Only purchased labels enter Context
updates or online DQN replay; a paid label also permits scoring the two alternative
routes. Unqueried test labels are observer-only except for the explicitly oracle
class-quota controller. The cap is floor(fraction*class_count); rejected action 2
is masked before selection when the class quota is exhausted, while actions 0/1
remain available. No forced filling, rehearsal or retention guard is used.

Evaluation uses training-fitted preprocessing persisted with the new checkpoint.
Thus this extension should not be compared to legacy target-fitted-scaler runs
as if the routing decision were the only changed factor. Attack-only results do
not establish routing performance on mixed clean/attacked traffic; that requires
a separate disjoint mixed-stream evaluation.

Artifacts: `Models/three_action_best.pth`, `training_history.json`, and a separate
`Results/three_action_*` folder with per-sample actions, routes, paid labels,
before/after CE, rewards, combined pre-feedback accuracy and F1, and manifests.
Cost and gain weight are checkpoint settings; changing either requires retraining.

Run `python AMCAL/ANT_Burst_TEST/test_three_action_reward.py` first. These tests
cover the reward formula and exhausted-budget action mask without PyTorch.
They do not substitute for a real model/gradient integration test.
