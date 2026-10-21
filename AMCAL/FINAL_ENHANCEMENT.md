# AMCAL final enhancement experiments

Branch: `AMCAL_final_enhancement`. Entry point: `AMCAL/amcal_final_enhancement.py`.
The published `amcal_burst_structured_bank_protocol.py` and its Models/Results
are retained as the reference implementation. New training requires a fresh output
directory. New checkpoints are named `final_enhancement.pth` and are not compatible
with the original evaluator.

The selector still maximizes prediction/transfer return, the chosen Context
minimizes its adaptation objective, and Context changes feed subsequent selector
states. The frozen CNN is never optimized online.

## Profiles

| Profile | Online Context | Reference / routing | Inference |
|---|---|---|---|
| `accuracy` | Original pretrained Context architecture | Protected memory, ranking, paired update advantage, version-weighted replay | Full bank |
| `compact` (default) | Independent small residual Contexts | Same enhancements | Full bank |
| `shared` | Frozen shared feature trunk and three independent residual adapters | Same enhancements | One trunk evaluation per bank forward |
| `cascade` | Shared trunk and independent adapters | Same enhancements plus a pretrained pre-router | Selective routes and full-bank fallback |

Use the original entry point for the original baseline. First compare `accuracy`
with that baseline, then compare `compact`, `shared` and `cascade`. This separates
learning changes from architectural and conditional-computation changes.

## First run (PowerShell, repository root)

```powershell
git fetch origin
git switch --track origin/AMCAL_final_enhancement

python AMCAL/amcal_final_enhancement.py train `
  --source-run AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42 `
  --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT `
  --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_compact_seed42 `
  --profile compact --warm-start-selector --seed 42 --device cuda

python AMCAL/amcal_final_enhancement.py evaluate `
  --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT `
  --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_compact_seed42 `
  --seed 42 --device cuda
```

If the local branch already exists, use `git switch AMCAL_final_enhancement`.
The PowerShell launcher below handles the same paths, one online seed (42) and the
persistent audit:

```powershell
./AMCAL/run_amcal_final_enhancement.ps1 -Action Train -Profile compact
./AMCAL/run_amcal_final_enhancement.ps1 -Action Evaluate -Profile compact
./AMCAL/run_amcal_final_enhancement.ps1 -Action Audit -Profile compact
```

Use `-Profile accuracy`, `shared` or `cascade` for separate runs. `-OutputRun`
can override the generated path. Repeated training seeds require separate paths;
the launcher's default evaluation seed is 42 and reuses one trained checkpoint.
Multiple evaluation seeds remain optional via `-Seeds @(42,48,54,62,72)`.

## What changed

- Correctness routing retains BCE and adds pairwise ranking only when one route
  is correct and another is wrong. All-correct/all-wrong rows introduce no forced
  ranking target. Online routing uses a smaller LR and a checkpoint prior.
- Paired TD compares an executed query with prediction through the SAME Context:
  `Q_query_j - Q_predict_j`. Its target is transfer utility plus the difference
  between updated and skipped successor values. Prediction rewards cancel.
  Unexecuted Context update outcomes are not fabricated.
- Replay recency is supplemented by Context version distance. Context resets
  advance the hidden-bank generation; target synchronization remains every 15
  successful TD optimizer steps. TD gradients cannot train correctness weights.
- A fixed total memory allowance of at most 256 rows is partitioned into 64
  source TRAIN anchors, 64 class-stratified historical rows and 128 recent FIFO
  rows. FIFO evictions enter the historical per-class reservoir. References are
  unique, balanced within partitions and exclude feature-identical current rows.
- Protected reference gain emphasizes recent feedback; damage measures source
  and historical rows. Excess normalized damage receives an additional penalty.
  This threshold is a loss-transfer threshold, not an accuracy-drop percentage.
- Up to eight protected reference rows contribute CE to the selected Context
  gradient; no additional labels are purchased. Source validation and test/audit
  labels are never memory/replay inputs. Offline protected-memory episodes
  reproduce online target-feedback availability rather than reading all target
  TRAIN labels into the reference immediately. Dense TRAIN route supervision
  remains explicit.
- Compact Contexts use two 32-wide Transformer layers, FFN width 128 and a
  depthwise/tokenwise residual adapter. The large flatten/dense decoder and final
  input LayerNorm are removed. The update is bounded around the identity input.
- Compact/shared initializations are refined using TRAIN CE and the unchanged
  source Context as an output teacher. Validation CE selects the warm start,
  including the identity initialization. Inspect `context_warm_start.json`;
  distillation does not guarantee recovery of the old initial quality.
- Shared trunks are frozen after offline refinement. Every slot has independent
  adapter weights and Adam state. Optional `--bank-executor vmap` vectorizes
  independent banks with a persistent parameter stack refreshed after updates.
- Correctness scores are returned in one forward, reliability stays on-device,
  and trace scores use a batched transfer.
- Cascade routing uses frozen embeddings/probabilities, confidence, margin,
  source-calibrated novelty, paid-feedback learning, periodic/random full audits,
  and a learned compute-benefit head. Its target compares the pre-feedback cheap
  candidate with the full learned router, subtracting a per-extra-Context cost.
  This is a relative compute proxy, not measured wall-clock reward. TRAIN or
  already-purchased labels alone supervise it. Full-bank query rows retain the
  original four counterfactual correctness targets from one paid label.

## Controls and options

```text
--disable ranking paired prior memory replay damage   # choose one or more
--rank-weight .2                --paired-weight .5
--route-lr-online .0003         --version-half-life 8
--memory-ce-weight .1           --memory-replay-size 8
--source-capacity 64            --historical-capacity 64
--recent-capacity 128           --reference-size 32
--damage-limit .02             --damage-weight 1
--transformer-layers 2          --token-width 32
--feedforward-width 128         --residual-bound 2
--cascade-confidence .8         --cascade-margin .15
--cascade-audit-fraction .05    --cascade-max-gap 32
--compute-cost .02
--routed-utility                # optional fixed-pre-update routing-weighted probe
```

Specify enhancement settings during TRAIN; evaluation restores and validates the
saved settings. Retrain into a fresh directory for component ablations. The
`memory` ablation restores the old reference-only protocol and disables protected
gradient replay. `replay` alone removes memory gradients but keeps protected
reward references. Do not combine `--routed-utility` with `--disable memory`.

For matched feedback and destinations, evaluate both modes with
`--query-schedule uniform --update-policy fixed`. The two modes then receive the
same sample IDs and destination updates; routing can differ. The default utility
does not change Context gradients as a function of routing. Explicit routed
utility changes rewards but protected replay sampling remains isolated.

Validation checkpoint selection uses mean online validation accuracy minus
`retention_selection_weight * max(0, source_drop_pp - max_reference_drop)/100`.
Defaults are one and two percentage points. These source-validation observers
select offline checkpoints; their labels never enter the online learner.

## Evaluation and diagnostics

Independent evaluation defaults to levels 5, 7, 10, 12, 15, 17 and 20. All use
source-disjoint TEST identities and hard per-stream label caps. `--continual`
instead retains adaptive state and reserves a fixed source-ID audit subset at
every intensity; audit rows never enter adaptation. It reports the final audit
matrix, forgetting and backward transfer. Acquiring the same identity in another
domain is charged again.

Saved outputs include combined accuracy/F1/label tables, full traces, source and
same-level endpoints, per-Context damage, full-bank execution fraction, Context
and trunk forward counts, synchronized timing, and response/row P50/P95/P99.
Same-level unqueried endpoints remain policy-dependent diagnostics; the fixed
continual audit provides the independent holdout.

For cascade rows, unknown route outputs are saved as `null`. A complete
all-route/allowed-action Oracle is `null` when any row skipped a route. A separate
`oracle_on_full_rows` is conditional on full execution and must not be compared
as if it covered the whole stream. No diagnostic performs hidden extra full-bank
forwards within a sparse row to manufacture a complete Oracle.

Endpoint observers run outside online timing and do not mutate the learner.
The runner retains the finite-stream frozen cache and one-input-lookahead TD
protocol. Response percentiles include prediction/action-selection synchronization
and exclude frozen-cache construction and the current row's update. Cache is
charged once to total online time. Successor prefetch can shift prediction work
to the preceding row's update stage. These are not live arrival/annotation-delay
service latency measurements. For identical-trajectory timing repeats, use e.g.
`--evaluation-tag timing-repeat-2`; run sequentially on an idle machine.

Exact execution hashes prevent overwriting prior evaluation runs. Config,
checkpoint and source/domain hashes are recorded. GPU accuracy/speed improvements
must be measured on the real artifacts before changing the paper's conclusions.

The same entry point can use supplied feature-space `Domains` and `Bank` folders.
For example, use `--source-run .../FGSM/Bank --burst-dir .../FGSM/Domains`,
TRAIN `--levels 0 2`, and evaluate `--levels 1 3 4 5`. These folders must contain
the original conversion's source-row maps and source-fitted preprocessing. The
loader enforces independent source IDs; it does not enable the all-supplied-row
diagnostic implicitly.

## Tests

```bash
python -m unittest discover -s AMCAL/tests -p test_final_enhancement.py -v
python -m unittest discover -s AMCAL/tests -p test_structured_bank_protocol.py -v
```

CPU tests cover gradient separation, paired credit, version weighting, memory
capacity/class retention/duplicate exclusion, selected-slot independence, actual
compact Context execution, vectorized refresh, hard caps, unqueried-label
invariance, sparse missing-route semantics, observer isolation and a real
CNN/Context train-save-load-evaluate/persistent-audit smoke run on synthetic
source-disjoint CSVs. These are correctness checks, not benchmark results.

For F=100 the real module parameter counts are 7,360,813 for the original Context
and 28,933 for one compact Context. A shared three-slot bank has 29,455 unique
parameters, of which 783 adapter parameters are trainable online. These counts
do not establish a latency ratio or an accuracy improvement.

## Matched sequential forgetting test (accuracy, seed 42)

The reference implementation is `AMCAL/amcal_burst_sequential_retention.py` on
`experiment/amcal-protocol` at commit
`44b907e0da1238d7d75e1fdfc154f4e2fbd8b274`. The current `develop` branch
does not contain this test. Its committed results are in
`AMCAL/ANT_Burst_IAT_TEST/protocol_runs/sequential_burst_audit_seed42`.

Reuse the trained accuracy checkpoint; do not run Train or All again:

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_final_enhancement.ps1 -Action Audit -Profile accuracy -Seeds @(42)
```

The fixed order is 5, 7, 10, 12, 15, 17, 20. Twenty percent of common TEST
source IDs are reserved using seed+8121, exactly as in the reference test.
These IDs are excluded from adaptation at every intensity. For the original
data this gives 214 audit rows and 859 adaptation rows per domain, with a
171-label cap per domain. Contexts, optimizers, selector, replay, reliability
and memory continue across stages; the annotation cap is renewed per domain.
TRAIN source anchors remain available to the enhanced method as designed.
Audit and clean VAL labels are observer-only and never enter learner memory.

Protocol version 2 records stage 0 and all seven post-adaptation stages,
including clean VAL accuracy (AuditLevel=0). FinalAudit excludes clean VAL.
Forgetting is the mean previous-domain best score after learning that domain
minus its final score; BWT is final minus the score immediately after learning
that domain. Both exclude the last domain and use percentage points.
CleanDrop is initial minus final clean VAL accuracy. FinalAudit and
MeanOnlineAccuracy are distinct: one measures the final model on fixed audit
rows, the other measures predictions made before feedback along the stream.

Reference learned results: FinalAudit=66.7557%, Forgetting=2.0249 pp,
BWT=+1.7134 pp, CleanInitial=71.0158%, CleanFinal=66.7288%,
CleanDrop=4.2870 pp, total labels=431. Reference random results:
FinalAudit=62.4166%, Forgetting=3.6604 pp, BWT=-2.6480 pp,
CleanDrop=14.4455 pp, total labels=1197. Compare label usage as well as accuracy;
equal caps do not guarantee equal paid-label counts.

Read `sequential_summary.csv`, `sequential_audit_matrix.csv`,
`combined_results.csv`, and `execution.json` under the new evaluation directory
in `final_enhancement_accuracy_trainseed42/Results`. This audit does not replace
the existing independent-level evaluation and does not overwrite it.

Validation: 40 CPU tests passed after this change, including the real CLI smoke
test checking stage 0, clean-drop and forgetting/BWT arithmetic, the exact split,
and exclusion of audit IDs from both adaptation stages. The full dataset/GPU
audit must be run on the machine holding the trained checkpoint and data.

## Current-bank router refresh (evaluation adaptation, opt-in)

`--router-refresh` addresses stale correctness labels: replay entries describe
Context versions that may no longer produce the same predictions. After each
paid Context update, the enhanced runner reuses the already sampled reward
reference batch (up to 32 existing TRAIN anchors / previously paid rows),
recomputes all four routes using the current bank, and takes one direct BCE plus
pairwise-ranking step on the correctness head. Old targets are not substituted
for these refreshed targets and refreshed states are not added to stale replay.
The regular pre-feedback routing and TD/paired-update learning remain active.

Reference labels were already available before the current feedback. The current
input is excluded by the reference sampler. Reliability features use a snapshot
from before observing the current label; Context output targets use the bank
after its update. Tensors are detached, and the additional optimizer step owns
only correctness parameters. It cannot change Contexts, CNN, TD heads, the label
schedule, or reference sampling RNG. There is no additional label storage beyond
the existing 256-row memory. Only learned mode adapts the router; random mode
does not. The extra forward/backward work is timed and can increase runtime.

First compare against the fixed-destination evaluation `evaluation_a952e57562db`
(seed 42, final audit 78.3712%, online 73.4741%, 1197 labels). This holds the bank
updates and paid-label schedule fixed while testing router adaptation:

```powershell
python AMCAL/amcal_final_enhancement.py evaluate `
  --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_trainseed42 `
  --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT `
  --continual --query-schedule uniform --update-policy fixed `
  --router-refresh --seed 42 --device cuda
```

This reuses the existing checkpoint without retraining or overwriting previous
evaluations. The flag is evaluation-only; execution metadata records version,
target timing, steps and zero extra label capacity. Per-level metrics report
`fresh_route_updates` and `fresh_route_rows`, separately from ordinary routing
updates. If routing improves under fixed destinations, follow with learned
destinations using the same flag to measure its interaction with the full loop.

Validation: 41 CPU tests pass, including refreshed-target agreement with the
current bank, routing-only parameter changes, exactly unchanged Context/CNN
trajectories under fixed feedback, no learning from unqueried labels, zero-budget
behavior, no refresh in random mode, and the real CLI uniform-query audit with
serialization. Accuracy gains require the full dataset experiment; none are
claimed from the synthetic tests.

## Informative-route supervision (online ablation)

The seed-42 current-bank refresh trial reduced final audit accuracy from
78.3712% to 77.8371% with fixed update destinations. Leave `--router-refresh`
off for this independent experiment.

`--router-informative-weight 3` multiplies the existing routing loss weight by
three only when at least one route is correct and another is incorrect. These
are the rows where choosing a route can change correctness. Replay recency,
bank-version weights, sample importance and normalized loss remain in effect.
Unanimously correct/incorrect rows still receive BCE calibration supervision;
their ranking loss is zero. The existing pairwise ranking term is retained,
not introduced as a new component. Batch size, optimizer-step count, replay
capacity, Context architecture and annotation budget do not change.

This is an opt-in online-adaptation experiment, not checkpoint retraining. The
default multiplier is 1 (the prior behavior); execution metadata records a
non-default multiplier and produces a separate result directory. Only the
correctness head receives this weighted loss. Context minimization and the
DDQN/paired update-return objectives remain unchanged. Training the selector on
new labels still requires paid feedback. The flag cannot be combined with
`--router-refresh`, to keep the comparison interpretable.

First run the full learned-destination loop with uniform paid feedback, keeping
the min-max update decisions active. Compare with `evaluation_728b41beb74a`:
final audit 77.9039%, online 73.6238%, forgetting 2.0249 pp, 1197 labels.
Run learned mode only; repeating the unaffected random baseline is unnecessary:

```powershell
git pull origin AMCAL_final_enhancement
python AMCAL/amcal_final_enhancement.py evaluate `
  --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_trainseed42 `
  --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT `
  --continual --query-schedule uniform --update-policy learned `
  --router-informative-weight 3 --modes learned --seed 42 --device cuda
```

Validation includes exact reproduction of the original routing loss at weight
1, analytically checked relative gradient emphasis, retained supervision for
unanimous rows, independent correctness versus TD/paired parameter updates,
unqueried-label invariance, and a real train-save-load CLI audit checking saved
execution metadata, disabled refresh, hard caps and a frozen CNN. Scientific
accuracy improvement is not established until the full dataset trial finishes.

## Joint four-route comparison (new selector architecture)

The baseline uniform-query trace at seed 42 contains 890 rows with exactly one
correct route; the selector chooses it on only 30.1124% of these rows. Of 904
errors with some correct available route, 622 occur in this group. This motivates
comparing full route states jointly rather than only increasing loss weights.

`--joint-router` adds a residual cross-route scorer to the shared correctness
network: route-state projection to 64 features, four-head self-attention across
the four route tokens, and a shared per-route residual output. Each token can
inspect all competitors' full states. There is no positional embedding or
majority-vote rule, so exchanging Context tokens exchanges their scores. The
existing frozen-route flag still distinguishes the frozen classifier. The final
residual layer starts at exactly zero, reproducing the old policy's logits and
seven Q values when its weights are loaded. At route width 166 this adds 35,649
parameters to the 136,963-parameter selector; Context capacity does not change.

Only routing supervision updates the shared scorer and comparison module.
Context minimization remains unchanged, and DDQN/paired transfer-return updates
still train the future/update-value heads with detached routing rewards. Thus
the learned Context-update destination remains part of the max decision. The
experiment below fixes feedback timing to the existing uniform-query control,
without fixing the learned prediction route or update destination.

Train in a fresh output directory using `--warm-start-enhancement` to reuse the
existing accuracy checkpoint, preprocessing, CNN and initial Context. It checks
all source-model and TRAIN-domain hashes and configuration compatibility. Initial
Context pretraining and correctness warmup are skipped. TRAIN episodes still
adapt Contexts and train the selector; temporary VAL episodes select a checkpoint
and are discarded. TEST is not used for fitting or selection. Epoch zero is
included as a validation candidate; at most six additional epochs are requested.

The one-command Windows launcher trains then evaluates with seed 42:
`./AMCAL/run_amcal_joint_router.ps1 -Action All`. To resume evaluation after
training finishes, use `-Action Evaluate`. It stops on Python failures. The
equivalent explicit commands are below. PowerShell is unavailable in the CPU
test environment, so the launcher received static review only.

```powershell
git pull origin AMCAL_final_enhancement
$base = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs'
$joint = "$base/final_enhancement_accuracy_joint_trainseed42"
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'

python AMCAL/amcal_final_enhancement.py train `
  --source-run "$base/final_enhancement_accuracy_trainseed42" `
  --output $joint --burst-dir $burst --profile accuracy `
  --joint-router --warm-start-enhancement --epochs 6 --patience 3 `
  --seed 42 --device cuda

# Run after successful training. The saved checkpoint enables the joint router.
python AMCAL/amcal_final_enhancement.py evaluate `
  --output $joint --burst-dir $burst --continual `
  --query-schedule uniform --update-policy learned --modes learned `
  --seed 42 --device cuda
```

Do not enable refresh or informative weighting for this trial. The original
checkpoint and its results remain untouched. Compare final audit 77.9039%,
online 73.6238%, forgetting 2.0249 pp and 1197 labels from the original learned
uniform-query loop, and inspect the one-correct-route subset in traces. Any
improvement includes additional selector training; a matched continuation run
without `--joint-router` is needed before attributing it specifically to the
architecture. Strict loading rejects partial comparison heads and missing old
policy weights; saved joint checkpoints are loaded automatically at evaluation.

Validation: 47 CPU tests passed. New checks cover exact neutral initialization,
cross-route sensitivity, Context-token permutation equivariance, old/new
checkpoint compatibility and rejection of incomplete states, learning of the
new head, separation from TD/paired gradients, unqueried-label invariance, and
real CLI warm continuation / epoch-zero selection / save-load evaluation. The
synthetic integration test confirms identical copied Context hashes, unchanged
parent checkpoint, frozen CNN, hard caps and disabled refresh. The full GPU/data
experiment has not been run in this environment.

## Aligned persistent training and checkpoint selection (baseline architecture)

The joint-router seed-42 trial did not improve the baseline: final audit 77.5033%
versus 77.9039%, online 73.6903% versus 73.6238%. This next trial reuses the
original accuracy selector, with no joint head, refresh or informative weighting.
It changes the training/selection protocol rather than Context or selector size.

`train --aligned-sequence --warm-start-enhancement --query-schedule uniform`
uses the following protocol:

- One training epoch is a complete ordered pass over TRAIN levels 0, 1, 3.
  Contexts, optimizers, reliability, memory, selector and TD replay persist within
  that pass. A new pass resets the bank/reference history and clears old replay;
  the learned selector weights are retained. Initial Context pretraining and
  correctness warmup are skipped, reusing the source checkpoint and hashes.
- Supervision is paid-only, even in TRAIN rollouts. The online optimizer settings,
  replay capacity, LR, epsilon, 20% per-domain cap and uniform query schedule match
  evaluation. Pretrained source TRAIN anchors remain available as before. Min
  Context updates and max learned destination decisions remain active.
- VAL has a separate fixed 20% source-identity audit split (seed+8121). Audit IDs
  are excluded from adaptation at EVERY validation level, including level 0.
  Clean retention uses only these audit IDs. A temporary independent bank/agent
  adapts on the remaining VAL rows in sequence and is discarded afterwards.
- Checkpoint selection uses the mean FINAL audit accuracy across VAL domains,
  minus the existing excess clean-drop penalty. Epoch zero remains a candidate.
  Intermediate audit matrices, final accuracy, online accuracy, forgetting, BWT,
  label counts and adaptation/audit source IDs are recorded. VAL observation
  never updates the TRAIN agent or bank. TEST is never used for selection.

TRAIN and VAL still use intensities 0, 1, 3; the held-out TEST sequence remains
5, 7, 10, 12, 15, 17, 20. This matches the adaptation and selection mechanics,
not the held-out domain distribution. Uniform queries intentionally consume the
full 20% cap; selective query decisions remain disabled in this controlled trial.

Run from the repository root in PowerShell:

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_aligned_sequence.ps1 -Action All
```

The launcher performs at most two complete TRAIN passes (six domain stages),
selects on independent VAL audits, then evaluates learned mode with seed 42.
The fresh output is `final_enhancement_accuracy_aligned_trainseed42` under the
same `protocol_runs` directory. To run evaluation after training separately, use
`-Action Evaluate`. Source models/checkpoints/results are not overwritten.
The baseline comparison is `evaluation_728b41beb74a`: final audit 77.9039%, online
73.6238%, forgetting 2.0249 pp and 1197 labels. As with any continuation trial,
further training and the changed protocol both contribute to the comparison.

Validation: 50 CPU tests pass, including paid-only TRAIN label invariance,
exact caps and one bank reset per multi-domain pass, exact source-identity audit
exclusion, no effect of audit labels on learned weights or Contexts, correct
final-audit/forgetting/BWT arithmetic, and a real baseline warm-start, aligned
training, epoch-zero selection, serialization and persistent evaluation run.
The CLI integration checks that saved selection uses the final audit matrix and
that training protocol metadata survives evaluation. PowerShell is unavailable
here; the launcher received static review. Scientific performance requires the
full dataset/GPU run on the machine holding the checkpoints.

## Fixed TRAIN calibration of relative utility

`run_amcal_utility_normalization.ps1` tests relative utility scaling on the existing
accuracy checkpoint, with seed 42, uniform feedback and learned Context destinations.
`-Action All` first calibrates on TRAIN levels 0, 1, 3 and then evaluates the persistent
held-out sequence 5, 7, 10, 12, 15, 17, 20. It does not retrain or overwrite the checkpoint.
Calibration makes one temporary baseline-policy rollout with online adaptation on TRAIN;
that adaptation is discarded. It reads neither validation nor test splits. Its runtime
cost and paid TRAIN feedback are additional calibration work, recorded in the JSON.
The held-out label budget and Context-min / selector-max feedback loop stay the same.

For each row let `U_j` be the three raw update-value predictions and `R_j = 2*p_j - 1`
the three Context correctness rewards. Calibration takes the 75th percentile of the
rowwise centered RMS of each component over TRAIN. The fixed multiplier is
`scale = max(Q75(RMS(R)), 1e-4) / max(Q75(RMS(U)), 1e-4)`.
The actor uses `mean(U) + d / max(1, max(abs(d))/clip)`, where
`d = weight * scale * (U - mean(U))`, with default `weight=1`, `clip=1`.
Thus only differences between Context utilities are scaled; their common mean is
preserved. Clipping shrinks large relative contributions, and does not stretch each
individual row to unit range. Similar typical TRAIN magnitudes do not imply equal
predictive reliability or guarantee improved accuracy.

Only actor action scores change. Recorded rewards, raw critic predictions, TD losses
and paired update targets retain their original units. Double-DQN chooses its successor
action with normalized actor scores and evaluates that action using the raw target
critic. There are no new learned parameters or state-dict keys. `scores` in traces
retains raw Q/correctness semantics; `raw_update_values` and `selection_scores` record
both components for normalized runs. Execution metadata includes the complete fixed
calibration, checkpoint SHA, TRAIN source provenance and weight/clip. Evaluation never
refits the scale. A mismatched checkpoint or existing calibration output is rejected.
Without `--utility-calibration`, existing behavior is preserved.

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_utility_normalization.ps1 -Action All -Seed 42
```

If calibration completed but evaluation was interrupted, resume with `-Action Evaluate`.
Do not rerun `All` over an existing calibration. Results use a configuration-specific
`Results/evaluation_*` directory under the original accuracy run. Keep the calibration
JSON with the results. CPU checks cover TRAIN-only calibration, unchanged checkpoint,
raw bootstrap units, destination changes, clipping and unqueried-label invariance;
PowerShell is reviewed statically. GPU accuracy and runtime require the experiment.

## Counterfactual TRAIN supervision of update utility

Relative scaling alone did not consistently improve the seed-42 persistent test:
weight 1 had final audit 78.1709%, online 72.8754%, forgetting 2.4922 pp;
weight 0.1 had final audit 77.1028%, online 73.4076%, forgetting 4.3614 pp.
The baseline remains 77.9039%, 73.6238%, 2.0249 pp, respectively. Each used
1,197 test feedback labels. These are exploratory single-seed results.

The next isolated experiment warm-starts the original accuracy checkpoint and
fits **only its existing `update_value` head** during TRAIN. Correctness, common
future value, CNN, initial Context checkpoint and target critic weights remain
fixed during that fitting. There are no new inference parameters and no utility
normalization. Normal online learning resumes for temporary validation learners
and for the final test learner.

On a paid TRAIN row, prediction and action selection precede feedback. The row's
single acquired label then supports three alternatives. Each alternative copies
a different Context together with its complete Adam state, makes exactly the
same Context-min update as the real learner, and measures the original utility
on the same pre-feedback protected reference. All alternatives reuse the same
protected replay draw. The current input is excluded from that reward reference.
Copies never modify the real bank, Adam state, reliability history, memory,
versions or RNG. After simulation, the real learner updates only its chosen
Context and inserts the acquired label once into memory. There are three virtual
Context steps plus one real Context step per paid TRAIN row, without acquiring
additional labels. Simulation cost is additional offline compute.

For Context j, the target is `utility_j + gamma * (V_target(next_j) - V_target(skip))`.
The successor uses the alternative's updated Context and reliability, with the
same next input features and no next label. Double-DQN action selection uses the
current raw policy; target evaluation uses the inherited frozen critic throughout
fitting. Terminal transitions have no bootstrap. This is a one-step paired return
estimate with inherited continuation values, not an oracle for full future return.
The update head minimizes raw SmoothL1 plus 0.5 times centered SmoothL1 across the
three Contexts. Both terms retain reward units; no per-row normalization is used.
A 512-row TRAIN-only counterfactual replay holds states and targets/successors,
uses the existing version/age weighting, and is cleared between passes. Adam
learning rate is 1e-4, weight decay 1e-4, batch at most 32, starting at 8 paid rows.

The launcher runs three paid-only persistent TRAIN passes over 0, 1, 3, using the
existing aligned-sequence protocol. Within each pass Contexts, their optimizers,
reference memory and reliability continue between levels; they reset between
passes while update-head weights and its optimizer continue. The selector's
other heads are frozen only during TRAIN. Checkpoint selection uses the existing
source-disjoint persistent VAL audit with its clean-retention penalty and includes
the inherited checkpoint as epoch zero. If epoch zero wins, report that the new
training was not selected. No TEST labels select a checkpoint.

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_counterfactual_utility.ps1 -Action All -Seed 42
```

Output: `AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42`.
The source run is read-only and the destination must be fresh for training.
`-Action Evaluate` resumes after successful training. Evaluation runs the usual
learned min-max closed loop with uniform feedback and the unchanged 20% budget,
without counterfactual simulations or actor scaling. The selected checkpoint and
execution JSON retain `training_protocol.utility_supervision`; per-epoch stage
logs include simulation counts, utility-head update counts, fitting loss and the
maximum difference between simulated and real utility for the selected Context.
TEST inference/online adaptation gains require measured results; none are assumed.
PowerShell is reviewed statically; CPU tests verify Adam-copy fidelity, live-state
and RNG isolation, unqueried-label invariance, frozen heads, terminal handling,
TRAIN-only execution, and real CLI serialization/checkpoint provenance.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results" "$run/enhancement_training.json"
git commit -m "Add counterfactual utility training and evaluation seed 42"
git push origin AMCAL_final_enhancement
```

## Original balanced buffer: inter-level training with RL queries, both tests

This experiment uses the finalized original coupled v1 balanced-reference bank,
`structured_bank3_balancedref_lr0p0005_seed42`, with the existing persistent
continuation training. It does not use enhancement memory, counterfactual utility,
actor normalization or the joint router. Use policy queries consistently for
TRAIN, temporary VAL adaptation, independent TEST and sequential TEST. All seven
actions are available to RL/epsilon until the 20% per-domain cap is exhausted;
there is no forced spending or uniform query schedule. Exploration remains 0.05.

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_balanced_sequence.ps1 -Action All -QuerySchedule policy -EvaluationSet Both -Seed 42
```

This trains once, with two passes over TRAIN 0, 1, 3, then evaluates the same
selected checkpoint twice using only learned mode and seed 42:

- Sequential forgetting: 5 -> 7 -> 10 -> 12 -> 15 -> 17 -> 20 retains the Contexts,
  selector, optimizers, replay, reliability and original balanced paid-reference
  memory. The fixed TEST audit identities are excluded from every adaptation
  level. Outputs are in `Results/Sequential`, including the full audit matrix,
  final audit mean, forgetting, BWT, clean-domain drop and actual labels used.
- Original independent per-level test: the same seven held-out levels use all
  disjoint TEST rows. Each level restarts the selector from the same selected
  checkpoint and resets Contexts, optimizers, replay, reliability and reference.
  It calls the original structured-bank evaluator, with the matching balanced
  reference flag, LR 0.0005, cap 20%, epsilon 0.05 and training scaler. Outputs
  are the original `Results/combined_structured_*` CSV reports and
  `Results/structured_learned_*` traces/metrics. Each sequence-trained independent
  evaluation records checkpoint SHA, selected epoch and training protocol in its
  execution metadata and result fingerprint.

The two tests have different adaptation row counts because the sequential audit
reserves TEST identities. Compare each to its corresponding original protocol,
including actual labels acquired, rather than comparing their online percentages
as identical conditions. Within each TRAIN pass the bank/reference survives level
transitions; between passes they reset while the RL weights continue. Selection
uses persistent VAL audit only and includes the inherited selector as epoch zero.
If epoch zero wins, the inherited selector was retained; do not attribute the TEST
result to successful new training.

Default output is
`AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_sequence_policy_seed42`.
The existing launcher defaults remain unchanged (`uniform`, sequential only).
`-Action Evaluate -EvaluationSet Independent -QuerySchedule policy -Seed 42`
can resume the independent part after the sequential part has finished. A fresh
output is required for training and for the sequential audit. CPU integration
covers actual policy-query training and both evaluation entry points, disjoint
TRAIN/VAL/TEST use, budget caps, and fresh selector/replay at each independent
level. PowerShell is reviewed statically; GPU experiments run on the user host.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_sequence_policy_seed42"
git add -f -- "$run/Results" "$run/structured_sequence_training.json"
git commit -m "Add balanced inter-level RL training and both tests seed 42"
git push origin AMCAL_final_enhancement
```

## Evaluate the counterfactual-trained model with RL-controlled feedback

The seed-42 counterfactual training selected pass 3 on VAL (70.40498% final audit).
Its uniform-feedback sequential TEST gave final audit 78.63818%, online 73.47414%,
forgetting 1.71340 pp, BWT -0.38941 pp and 1,197 labels. The inherited accuracy
baseline gave 77.90387%, 73.62382%, 2.02492 pp, +0.93458 pp and 1,197 labels.
These are single-seed results with mixed tradeoffs, not a uniform improvement.

`run_amcal_counterfactual_rl_tests.ps1` evaluates the existing selected checkpoint
without training, actor normalization or TEST counterfactual simulation. It
switches feedback to `--query-schedule policy --update-policy learned`: the
original coupled seven-action policy chooses query timing, prediction route and
Context destination, with fixed epsilon 0.05 and a hard 20% cap. A query predicts
through its chosen Context before seeing the label, then updates that same Context.
The cap is a maximum, not a required spend. This deliberately tests a checkpoint
trained with uniform feedback under RL feedback selection; it does not claim the
checkpoint has been retrained for that schedule.

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_counterfactual_rl_tests.ps1 -Seed 42
```

The default performs exactly two evaluations on held-out levels 5, 7, 10, 12, 15,
17, 20, each starting from the same saved checkpoint:

- Sequential forgetting: `--continual`, retains adaptive state across levels and
  excludes the fixed TEST audit identities from every adaptation stream.
- Independent per-level: no `--continual`, reloads selector weights and resets
  Contexts/optimizers/replay/reference/reliability at each level, using all TEST
  partition rows. It is independent of the preceding sequential run.

Both store their own `Results/evaluation_*` folder in
`final_enhancement_accuracy_counterfactual_trainseed42`; `execution.json` records
`query_schedule: policy`, the checkpoint SHA and `continual: true` or `false`.
Only the sequential folder contains `sequential_summary.csv`. The checkpoint and
previous uniform-feedback results are never overwritten. Compare each test to
its matching protocol and include actual feedback counts: independent and
sequential adaptation streams have different sizes. With epsilon exploration,
"RL-controlled" does not mean deterministic greedy decisions.

For an interrupted suite, resume only the unfinished part with
`-EvaluationSet Independent` or `-EvaluationSet Sequential`. Exact repeated
configurations are rejected by the existing evaluator. No training action is
provided by this launcher. CPU integration verifies both real evaluation paths,
shared initial checkpoint, empty replay on independent starts, retention between
sequential levels, hard caps, no TEST simulation, and unchanged checkpoint bytes.
PowerShell is reviewed statically; GPU performance is measured on the user host.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add counterfactual model RL-feedback tests seed 42"
git push origin AMCAL_final_enhancement
```
