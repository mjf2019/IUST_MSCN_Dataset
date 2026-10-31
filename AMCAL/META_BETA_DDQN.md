# Learned reward weight: experimental two-action DDQN

This separate experiment preserves all original source files, model checkpoints
and results. No tests, imports, training or inference were executed by the
authoring assistant. No packages are installed.

The reward is still paper_reward + beta * measured same-domain TRAIN gain.
Beta is now a positive trainable scalar: initialized by TRAIN reward/gain RMS
and updated through one differentiable virtual SGD critic step. The outer
objective maximizes signed measured gain times virtual query probability on
independent TRAIN trajectories/probes. It retains gain magnitude, not just sign.
Only beta receives this outer update; it is not fitted by reducing its own TD
residual. The virtual SGD step locally approximates the actual Adam update.

DDQN is retained with gamma .95, LR .0001, batch32, target interval15 and
replay4096. Replay stores raw paper/gain components and reconstructs rewards
at the current detached beta. Beta uses Adam .01 every15 TD updates, with
log range [-8,8] as a numerical guard. Hypergradients and guard hits are logged.

Existing beta26 IAT / beta55 PS query policies warm-start Q and provide fixed
TEST references. They do not set the new scalar. The original IAT CNN and
Light PS CNN, initial Context, fixed Context -> CNN path and Context batch5 /
LR .00025 / replay .1 update are preserved.

Inner TRAIN keeps the existing300 rows per level0/1/3 and24-row gain probes.
Additional TRAIN meta streams have100 rows per level and separate24-row probes.
Their IDs/features are disjoint from inner streams/probes, other meta data and
the reserved source reference. Held-out meta rows/probes are removed from inner
replay. Each adaptive trajectory has a20% feedback cap. Probe labels, extra
trajectories and historical pretraining are additional TRAIN supervision;
total offline supervision is not claimed to be20%.

Default: three additional Q-training epochs, selecting an epoch by VAL online
AC minus positive source forgetting, with fewer labels breaking ties. VAL does
not update Q or beta. TEST never selects or fits a model. Only temporary Contexts
adapt; their final weights are discarded. TEST resets Context/Adam/memory/cap per
level/budget; Q and beta remain frozen.

Comparison: LearnedBeta versus unchanged SourceFixed and uncertainty querying,
same data/order/Context/replay/budget. The uncertainty case is a policy replacement
on the same pretrained Context, not an independently retrained AL method.
Warm start, extra TRAIN samples and further Q training mean beta's causal effect
needs a matched-training fixed-beta ablation. Exploratory paid meta targets are a
local surrogate, not unbiased estimates of all counterfactual sample gains.

There is no new explicit label-cost or forgetting term. Better accuracy, lower
label use and less forgetting are hypotheses. The local competitive Context update
and closed-loop DDQN remain; no exact common min-max saddle equivalence is claimed.

## Run on the user's machine

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_meta_beta_ddqn.ps1 -Dataset IAT
    .\AMCAL\run_amcal_meta_beta_ddqn.ps1 -Dataset PS

Each command runs the accompanying CPU tests, trains a separate query checkpoint
and compares three policies on all seven TEST levels,300 rows each,20% budget.
Defaults save under protocol_runs/meta_beta_ddqn_seed42_p300_e3_v1.

Optional later budget evaluation (no retraining):

    .\AMCAL\run_amcal_meta_beta_ddqn.ps1 -Dataset IAT -Action Evaluate -Budgets 0.2,0.4,0.6,0.8,1.0
    .\AMCAL\run_amcal_meta_beta_ddqn.ps1 -Dataset PS -Action Evaluate -Budgets 0.2,0.4,0.6,0.8,1.0

    .\AMCAL\push_amcal_meta_beta_results.ps1 -Dataset Both

Only JSON/CSV diagnostics/results are staged; model/trainer binaries stay local.
Resume is from completed training epochs and completed TEST levels.

