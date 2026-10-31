# Independent target TEST: learned-beta DDQN versus window uncertainty

Evaluation only on IAT and PS, all eligible official TEST rows at levels
5/7/10/12/15/17/20. Only two methods are run: LearnedBetaDDQN and
UncertaintyWindow (window5, highest entropy, no percentile threshold).
Existing learned-beta checkpoints and initial CNN/Context are reused. There
is no training, model selection or new beta fitting. Historical TRAIN sizes
and initial source weights are unchanged. Older result directories are preserved.

## Split and budget

Before either method adapts, target rows across all seven levels are jointly
grouped by source flow ID OR exact processed feature equality (transitive union).
The first fixed fold of a five-fold stratified group partition is held out.
The same flow or feature group cannot be held out at one level and used for
adaptation at another. Stratification uses labels only to construct the split,
not to rank/query samples or optimize a model. A seed-fixed first fold is used;
folds are not chosen based on performance.

Approximately20% per level is the independent same-level target TEST set;
the remaining approximately80% is the adaptation stream. Groups are never
broken to force exactly20% of rows. The manifest records actual sizes, flow
IDs, feature/data hashes and class counts for every level. No target rows
are randomly truncated or discarded by this new split.

The cap is floor(.2 * adaptation_stream_rows), not20% of all target rows.
This is nominally16% of all rows. Both denominators and actual cap use are
printed and saved. Window uncertainty distributes this cap over arrived
windows using the original cumulative floor quota, including the last window.

Held-out IDs and processed features are also removed from the source replay
pool. Dynamic paid replay contains adaptation samples only. Both methods
use the same purged source pool, initial Context, source reference and stream
order. Replay cannot silently reintroduce the held-out target samples.
If the resulting source replay is empty or loses a class, the run stops
instead of falling back to a leaking pool.
Existing historical model weights are not retrospectively refitted; this
guarantees isolation from the current adaptation, not absence of all possible
historical feature duplicates in pretraining.

## Metrics

* AC/PR/RC/F1 online: pre-feedback predictions on the adaptation stream.
* target_initial_AC/PR/RC/F1: the initial model on the independent holdout.
* target_final_AC/PR/RC/F1: the final live adapted model on that holdout.
* target_holdout_gain: final minus initial on exactly the same independent rows.
* source forgetting: source-reference initial minus final metrics.
* labels_percent: labels divided by adaptation stream size.
* labels_percent_of_all_target_rows: labels divided by stream plus holdout size.

Legacy same-stream final metrics are retained only under explicitly named
adaptation_stream_seen_before/after diagnostic fields, not target_final columns.
The holdout is never queried, replayed or used as a training reward, threshold,
checkpoint selection metric or action input. It is read for observation only.

Context/Adam/memory/cap reset for each method and level. The saved CNN, query
policy and learned beta remain fixed. DDQN feedback is immediate; the window
method has delayed feedback. This protocol does not remove that scheduling
difference or turn the shared pretrained Context into independently retrained
active-learning architectures.

## User-run commands

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_target_holdout20.ps1 -Dataset IAT
    .\AMCAL\run_amcal_target_holdout20.ps1 -Dataset PS

Each command runs small CPU contract tests, then14 real TEST cases.
No tests, imports, training or inference were executed by the authoring assistant.

    .\AMCAL\push_amcal_target_holdout20_results.ps1 -Dataset Both

Only JSON/CSV manifests, metrics and traces are staged. Output directory:
protocol_runs/target_holdout20_learned_vs_uncertainty_seed42_w5_full_v1.
Completed cases are reused only after protocol, execution and file-hash checks.

