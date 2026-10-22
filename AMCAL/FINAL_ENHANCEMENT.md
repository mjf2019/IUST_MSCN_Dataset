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

## Two-Context evaluation ablation (five actions)

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_two_contexts.ps1 -Seed 42
```

This evaluates the existing counterfactual-trained accuracy checkpoint with **two
actual Context modules**, three prediction routes (frozen CNN + Context 1/2), and
five coupled actions (skip/predict 0/1/2, query/update Context 1/2). There is no
third module, dummy route, padded state, third-slot update or third-route oracle.
Current-state probability aggregates, reliability, pooled selector features,
legal masks, paired advantages, successor values and observer routing all use the
active route count. The scorer and utility heads are shared across routes, so
all inherited learned tensors load strictly without slicing or reinitialization.
Default three-Context operation retains its previous shapes and result metadata.

This is a bank-size removal ablation using weights trained with three Contexts;
it does not train a new two-Context architecture or establish its best attainable
accuracy. The frozen CNN, shared selector checkpoint, preprocessing, memory,
reward, Context learning rate, epsilon 0.05, source partitions and 20% cap are the
same as in the corresponding three-Context run. Learning still occurs online on
paid labels, with separate Adam state for each of the two Contexts. The initial
Context weights are identical across slots, as in the original bank. The cascade
and joint-router variants are excluded from this initial ablation; utility
normalization and counterfactual TRAIN fitting with two Contexts are not enabled.

The default `policy` schedule matches the latest three-Context RL-feedback test.
It runs both sequential forgetting and independent per-level TEST, seed 42, on
5, 7, 10, 12, 15, 17, 20. No training is run. The sequential protocol retains all
adaptive state between levels and excludes fixed audit identities; the independent
protocol restarts from the same inherited checkpoint at every level. Actual label
spending may differ from the three-Context policy despite equal maximum caps.
Compare accuracy, forgetting, oracle, actual labels and measured runtime together.

For a comparison to the earlier uniform-feedback result (78.63818% sequential
final audit), run separately with `-QuerySchedule uniform`. That changes feedback
scheduling and should be compared to the matching three-Context uniform run.
Do not compare two-Context independent accuracy directly to sequential final audit.
`-EvaluationSet Independent` or `Sequential` resumes only the unfinished test.

Results are separate `Results/evaluation_*` directories under the existing
`final_enhancement_accuracy_counterfactual_trainseed42` run. Execution metadata
records `context_count: 2`, `action_count: 5`, the shared checkpoint SHA and whether
state is continual. No prior results or checkpoint files are overwritten. Traces
have five Q values followed by three route probabilities (`scores` length 8),
three route predictions and actions 0..4; slot queries have length 2. This differs
from the three-Context trace schema (seven Q values, four probabilities).

CPU tests cover checkpoint tensor compatibility, five-action masks/bootstrap,
active-route state/observer shapes, two-slot online TD and paired learning,
unqueried-label invariance, real CLI execution of both tests and checkpoint
immutability. GPU accuracy and latency remain experimental; PowerShell is reviewed
statically.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add two-Context RL-feedback tests seed 42"
git push origin AMCAL_final_enhancement
```

## Single-Context evaluation ablation (three actions)

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_one_context.ps1 -Seed 42
```

This is the next bank-size removal test: one real Context, two prediction routes
(frozen CNN and Context), and three coupled actions (predict CNN, predict Context,
query/update Context). The default uses **uniform feedback**, seed 42, the same
20% cap, and both persistent forgetting and independent per-level evaluation on
5, 7, 10, 12, 15, 17, 20. It reuses the selected counterfactual-trained checkpoint
without offline retraining. Shared selector/utility tensors load unchanged;
state aggregates, reliability, TD masks and paired advantages use only the two
active prediction routes. No inactive Context is created or scored. With one
Context there is no competition between alternative update destinations; the
selector still decides the prediction route, and `policy` feedback can additionally
choose whether to query. Context-min adaptation and closed-loop feedback remain.

Use the matching uniform-feedback sequential results as references: three Contexts
78.63818% final audit, 73.47414% online, 1.71340 pp forgetting; two Contexts 76.16822%,
73.20805%, 3.27103 pp. Both used 1,197 labels. Their independent metrics must be
compared separately. This removal ablation does not measure the best performance
of a model trained specifically with one Context.

`-QuerySchedule policy` is an optional separate evaluation against policy-feedback
runs; it is not the default. `-EvaluationSet Independent` or `Sequential` resumes
one unfinished test. The source checkpoint and previous results are unchanged.
Output directories under the same run's `Results/evaluation_*` record
`context_count: 1`, `action_count: 3`, `query_schedule`, `continual` and checkpoint
SHA. Traces have two route predictions, three Q values + two probabilities in
`scores` (length 5), actions 0..2, and one slot-query count. No training command,
counterfactual simulation or normalization is enabled by the launcher.

CPU tests verify shared-checkpoint compatibility, one-Context masks and online
TD/paired learning, unqueried-label invariance, and both real CLI evaluation
protocols with unchanged checkpoint bytes and exact uniform caps. PowerShell
is reviewed statically; GPU accuracy and speed require the experiment.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add one-Context uniform-feedback tests seed 42"
git push origin AMCAL_final_enhancement
```

## Four- and five-Context evaluation ablation

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_contexts_4_5.ps1 -Seed 42
```

This performs exactly **four evaluations**, without offline retraining: four
Contexts with sequential forgetting, four Contexts with independent per-level
TEST, then the same two protocols with five Contexts. All use seed 42, uniform
feedback, the same selected counterfactual-trained checkpoint, frozen CNN,
preprocessing, 20% per-domain cap, epsilon 0.05 and original held-out levels
5, 7, 10, 12, 15, 17, 20. Sequential runs keep adaptive state between levels and
exclude the fixed TEST audit identities; independent runs reset at each level.
The same uniform label schedule is used at each matching stream size/count.

The bank contains four or five actual Context modules, each starting from the
same saved initial Context weights and maintaining separate online parameters
and Adam state. Extra Contexts are not new pretrained experts. The inherited
selector/correctness/utility weights are shared across routes and load strictly
without parameter duplication, slicing or new random head initialization. State
aggregates, reliability, masks, paired advantages and observers use every active
route, with no dummy padding. Four Contexts give five prediction routes and
nine coupled actions; five give six prediction routes and eleven actions.
The test measures runtime bank expansion of a model trained with three Contexts,
not the best performance after dedicated four/five-Context training. More Contexts
do not guarantee more accuracy or diversity.

Compare sequential results with the matching uniform-feedback references:

| Contexts | Final audit % | Online % | Forgetting pp | Clean drop pp | Labels |
|---|---:|---:|---:|---:|---:|
| 1 | 76.30174 | 73.90654 | 4.67290 | 11.64958 | 1197 |
| 2 | 76.16822 | 73.20805 | 3.27103 | 6.52377 | 1197 |
| 3 | 78.63818 | 73.47414 | 1.71340 | 5.21901 | 1197 |

Keep independent per-level accuracy separate from final sequential audit. Also
compare route oracle and measured runtime at the same protocol; these are
single-seed exploratory comparisons. The optional `-QuerySchedule policy` creates
separate RL-feedback evaluations with potentially different actual label counts.
Uniform is the default for a same-budget capacity comparison.

To resume a particular unfinished case, use `-ContextCounts @(4)` or `@(5)` and
`-EvaluationSet Sequential` or `Independent`. Completed exact configurations are
rejected by the existing evaluator. No training action is available. Models and
previous results are preserved; each case gets its own `Results/evaluation_*`
folder under the counterfactual run. Metadata records context/action counts,
checkpoint SHA, feedback schedule and continuation flag; expanded banks use a
separate expansion-protocol descriptor while older result metadata is unchanged.

Trace Q/probability lengths are 14 (9+5) for four Contexts and 17 (11+6) for five;
actions are 0..8 or 0..10, predictions have five or six entries, and slot-query
counts have length four or five. No counterfactual TEST simulation, utility
normalization, cascade or joint-router ablation is mixed into this experiment.
CPU tests cover both expanded shapes, masks, permutation symmetry, actual
TD/paired online learning and unqueried-label invariance, plus all four real
CLI cases with checkpoint immutability and exact caps. PowerShell is reviewed
statically. Scientific GPU accuracy and speed require the experiment.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add four- and five-Context uniform-feedback tests seed 42"
git push origin AMCAL_final_enhancement
```

## Single-Context retention: protected-replay CE weight 0.1 -> 0.3

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_one_context_retention.ps1 -Seed 42
```

This runs exactly two evaluations of the same single-Context checkpoint,
sequential forgetting and independent per-level TEST, with uniform feedback and
seed 42. No offline training is performed. Only the coefficient of the existing
protected-replay cross-entropy in the online Context loss changes, from the
checkpoint value 0.1 to 0.3. The current-row objective, residual loss, reward
retention/gain weights, routing/utility architecture, CNN, initial Context and
selector weights, memory capacities and sampling, replay size 8, LR 0.0005,
epsilon 0.05 and 20% cap are inherited unchanged. The single Context's actual
online parameters and downstream learned trajectory may change, as intended.
No extra labels or replay rows are acquired.

The existing Context update is `current_objective + memory_ce_weight *
mean(protected_replay_CE) + residual_penalty`. The new flag
`--online-memory-ce-weight 0.3` overrides only `memory_ce_weight` after loading the
checkpoint, without editing its config or bytes. It is initially restricted to
single-Context uniform-feedback evaluation with learned routing, protected
references and positive replay size, independently of other routing/utility
ablations. Other evaluation defaults and all TRAIN paths are unchanged.

Reference single-Context uniform sequential results with weight 0.1 are:
final audit 76.30174%, online 73.90654%, forgetting 4.67290 pp, clean-domain drop
11.64958 pp and 1,197 labels. Independent accuracy was 69.68446% with 1,498
labels. The objective is reduced forgetting/clean-domain drop while retaining
online and final accuracy; no benefit is assumed before the experiment.

Results use fresh configuration-specific `Results/evaluation_*` directories under
`final_enhancement_accuracy_counterfactual_trainseed42`. `execution.json` includes
the effective config (`memory_ce_weight: 0.3`), the original checkpoint SHA, and
`online_adaptation_override.memory_ce_weight` with `checkpoint: 0.1` and
`online: 0.3`. The override participates in the run fingerprint, preventing
confusion with the previous 0.1 results. Only the sequential run has the fixed
TEST audit and `sequential_summary.csv`; compare the independent run separately.
Use `-EvaluationSet Independent` or `Sequential` to resume the unfinished part.
The optional `-MemoryWeight` parameter accepts a different finite nonnegative
coefficient, but the default experiment is one coefficient (0.3), not a sweep.

CPU tests verify that the coefficient changes real Context updates with identical
acquired source IDs and memory capacity, reject invalid/mixed settings, and run
both CLI protocols with exact caps, effective/original config provenance and
unchanged checkpoint/CNN. PowerShell is reviewed statically. GPU accuracy and
retention effects must be measured on the full experiment.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add one-Context replay weight 0.3 retention tests seed 42"
git push origin AMCAL_final_enhancement
```

## Single-Context protected-reference damage guard

Increasing the online replay CE weight from 0.1 to 0.3 did not solve the seed-42
forgetting problem: final sequential audit fell from 76.30174% to 74.16555%,
forgetting rose from 4.67290 to 8.02181 pp, and online accuracy stayed almost flat
(73.90654% vs 73.87327%). Clean-domain drop improved from 11.64958 to 9.04007 pp.
The next isolated experiment retains the original replay weight **0.1** and
controls the actual Context step instead of increasing that coefficient.

```powershell
git pull origin AMCAL_final_enhancement
./AMCAL/run_amcal_one_context_damage_guard.ps1 -Seed 42
```

This runs two evaluations, sequential forgetting and independent per-level TEST,
using the same single-Context checkpoint, uniform feedback, seed 42, epsilon 0.05,
LR 0.0005, replay size 8, fixed memory capacity and 20% label cap. No offline
training is run. After a paid label is acquired, the ordinary Context-loss
clipped gradient is computed once on an isolated copy of the selected Context
and its complete Adam state. Candidate Adam steps use LR multipliers 1, 0.5,
0.25, each from the same pre-step weights, moments and step counter. The largest
acceptable candidate is committed. The configured base LR is restored and Adam
advances exactly once on acceptance. If all candidates fail, the live Context,
buffers, gradients, Adam state, Context version and update-history counters remain
unchanged. If no protected reference is available, the Context step is rejected.

Acceptance checks only source/historical rows (`segments != 2`) from the SAME
already-sampled pre-feedback reward reference. No extra reference or replay draw
is made. Let `d=(CE_before-CE_after)/(CE_before+CE_after+1e-8)`. Harm is the mean
across represented classes of `max(-d,0)` on those protected rows. Accept when
harm is at most the checkpoint's existing `damage_limit` (0.02). Recent-reference
rows do not determine this guard; the ordinary reward still uses the full
reference and the actual committed outcome. The threshold is a per-step normalized
loss constraint on a sampled past reference, not a guarantee of TEST accuracy,
zero forgetting, or a cumulative bound over the entire sequence.

A rejected step still charges the acquired label, observes paid-label reliability,
adds the label once to memory and trains the selector normally. Its actual
transfer gain/harm are zero and its utility is minus the unchanged selection
cost. It does not fabricate a Context update or advance Adam for a rejected trial.
The prediction was already scored before this feedback. No VAL, TEST audit or
unqueried labels enter the guard, gradients, references or acceptance rule.
This preserves the paid-feedback closed loop while constraining Context-min
adaptation; there is no new learned selector head or reward formula.

`--context-damage-guard` is restricted to single-Context protected evaluation with learned updates
(uniform or policy feedback) without routing/normalization ablations or the replay-weight override.
All unflagged paths remain unchanged. The new execution metadata records the
threshold, LR multipliers, reference scope, optimizer transaction and unchanged
checkpoint SHA. Fresh `Results/evaluation_*` directories contain per-domain
`damage_guard` metrics (accepted/rejected/backtracked steps, total trials, maximum
accepted harm, accepted step scale and missing-reference counts). Traces attach
an acceptance decision only to paid rows. Existing timing includes trial probes
and copy/commit overhead; runtime improvement is not assumed.

Use `-EvaluationSet Independent` or `Sequential` to resume one unfinished test.
Compare to the original weight-0.1 single-Context results at identical label counts,
including online accuracy, final audit, forgetting, clean-domain drop and guard
acceptance rates. CPU checks cover full-step equivalence with populated Adam
moments, correct backtracking, exact live-state/RNG preservation on rejection,
protected-vs-recent reference scope, missing-reference rejection, accepted-only
version/Adam counters, label charging and unqueried-label invariance, plus both
real CLI protocols and checkpoint immutability. PowerShell is reviewed statically;
scientific effects require the user-host GPU run.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add one-Context damage-guard retention tests seed 42"
git push origin AMCAL_final_enhancement
```


### Single-Context RL feedback with and without damage guard

```powershell
git pull --ff-only origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_one_context_rl_damage_guard.ps1 -Seed 42
}
```

This launcher runs four evaluations: RL baseline and RL damage guard, each with
sequential forgetting and independent per-level tests. It reuses the same
counterfactual-trained checkpoint; no offline training occurs. Both variants use
`--context-count 1 --query-schedule policy --update-policy learned --modes learned`,
epsilon 0.05, a 20% label cap, Context LR 0.0005 and the saved replay weight 0.1.
The guard has exactly the previous acceptance rule, reference scope and Adam
transaction semantics; only the uniform-only restriction is relaxed. RL chooses
whether to acquire a label as well as the prediction route; with one Context
there is only one possible Context-update destination. Paid labels still count
even when their Context step is rejected. The actual number and identities of
queried labels can differ between variants because their closed loops diverge.
Report actual labels together with accuracy, forgetting and runtime; equal caps
do not imply equal consumed budgets. No unqueried or audit labels supervise RL
or the guard. Each independent level starts afresh; sequential levels preserve
Context, selector, optimizer and memory state under the existing protocol.

Use `-Variant Baseline` or `Guard` and `-EvaluationSet Sequential` or `Independent`
to run only an unfinished case. Existing evaluation directories are preserved.
The execution JSON records `query_schedule: policy` and, for guarded runs only,
`context_damage_guard`; both record the same checkpoint SHA.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add one-Context RL baseline and damage-guard tests seed 42"
git push origin AMCAL_final_enhancement
```


### Dedicated single-Context RL sequence training

```powershell
git pull --ff-only origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_one_context_rl_training.ps1 -Action All -Seed 42
}
```

This is a separate TRAIN run, not another evaluation of the transferred
three-Context selector. It warm-starts the shared selector and initial Context
from `final_enhancement_accuracy_counterfactual_trainseed42`, then specializes
the correctness, future and update-value heads using one Context, policy-chosen
queries and paid-only supervision. The Context continues to minimize current
and protected replay loss on acquired labels inside each rollout. The selector
maximizes the existing correctness/transfer return with paired Double-DQN
updates; the reward, selection cost, memory layout and replay weight 0.1 are
unchanged. No damage guard, counterfactual simulations or utility normalization
is applied. The frozen CNN and the inherited initial Context artifact are
unchanged. Only acquired labels and source TRAIN anchors supervise the learner.
Unqueried labels are used solely for observer metrics.

TRAIN follows levels 0, 1, 3. Within each epoch, Context, Context Adam state,
selector state, reliability history and protected memory persist across levels.
At the next epoch, Context/Context Adam/history/memory reset; learned selector
weights and its optimizer state persist. This follows the existing aligned
sequence protocol. The effective TRAIN exploration is 0.05, matching validation
and TEST. The label cap is 20% per domain, not a forced consumption target.

Each candidate selector is validated by a fresh temporary one-Context learner
with policy queries. Fixed source IDs are held out for observer-only VAL audits
at every intensity and are excluded from VAL adaptation and novelty calibration.
The existing selection score is final persistent VAL audit accuracy minus
excess clean-drop penalty. The inherited epoch-zero candidate is also eligible;
if training does not improve that score, it remains selected. This possibility
is reported explicitly in `enhancement_training.json`. There is no TEST-based
checkpoint or epoch selection.

The launcher defaults to 12 epochs, patience 5 and seed 42. It writes a fresh
run at `final_enhancement_one_context_rl_trainseed42`, preserving the source
checkpoint and all previous results. `-Action All` performs one training run
(including VAL selection), then exactly two held-out TEST evaluations: independent
per-level and sequential forgetting, both learned RL mode and without damage
guard. `-Action Train`, `Evaluate` (independent) or `Audit` (sequential) can run an
individual stage. An existing output directory is never overwritten by training;
use the evaluation actions to resume after training, or choose a new `-OutputRun`.

Specialized checkpoints record `training_context_count: 1` and the policy-feedback
training protocol; evaluation rejects a mismatched Context count. Existing
three-Context checkpoints retain their count-ablation behavior. Validation now
constructs its initial reliability history from the active bank size; the old
default of four routes was only valid for three Contexts.

Compare this run against the one-Context transferred RL baseline
(`evaluation_5a3414a8f372` sequential and `evaluation_525c2e425df9` independent).
Report selected epoch, actual labels per level, online and final audit accuracy,
forgetting, BWT, clean drop and runtime. Equal caps do not imply equal label
consumption. CPU tests cover TRAIN unqueried-label invariance, one-Context
training and validation, persistent cross-level state, actual TD learning,
VAL-audit exclusion, both real CLI TEST protocols and artifact immutability.
PowerShell is reviewed statically; accuracy effects require the user-host run.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_one_context_rl_trainseed42"
git add -f -- "$run/enhancement_training.json" "$run/Results"
git commit -m "Add dedicated one-Context RL training and evaluation results seed 42"
git push origin AMCAL_final_enhancement
```

Large checkpoint files do not need to be pushed for result analysis.


### Uniform-feedback diagnostic of the dedicated one-Context RL checkpoint

```powershell
git pull --ff-only origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_one_context_rl_uniform_diagnostic.ps1 -Seed 42
}
```

This runs exactly two evaluations of the existing selected checkpoint in
`final_enhancement_one_context_rl_trainseed42`: independent per-level and
sequential forgetting. It does not train or modify the checkpoint. Relative to
the dedicated RL evaluations, only the query schedule changes from `policy`
to `uniform`. The learned prediction router remains active. Context count 1,
seed 42, epsilon 0.05, budget cap 20%, Context LR 0.0005, saved memory/reward
settings and audit protocol remain unchanged; damage guard is off.

The purpose is to assess how much the new model's low feedback rate contributes
to poor accuracy. Uniform feedback changes both label count and selected label
identities, and therefore the subsequent closed-loop adaptation trajectory;
this is a diagnostic, not a comparison at equal consumed labels. Compare first
with specialized policy-feedback runs `evaluation_d6d83a0d2044` (independent)
and `evaluation_7370c9a6cbb2` (sequential). Also compare against the original
transferred one-Context uniform runs `evaluation_5aaf0ac447aa` and
`evaluation_c5d2890f1f23` at matched uniform budgets (1498 and 1197 labels).
A recovery would implicate feedback scheduling; a remaining gap would justify
further investigation of the changed selector and adaptation interaction.

The source checkpoint SHA should remain
`e1131d1b07d764c5ccb590c5d84f88ca5113b75e5f5be7ebaeffb21d9b118e91`.
New execution JSONs record `query_schedule: uniform`; existing Results
directories and policy-feedback results are preserved. Use
`-EvaluationSet Independent` or `Sequential` to resume one unfinished case.
The launcher is reviewed statically; no numerical result is assumed.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_one_context_rl_trainseed42"
git add -f -- "$run/Results"
git commit -m "Add uniform-feedback diagnostic of dedicated one-Context RL checkpoint seed 42"
git push origin AMCAL_final_enhancement
```


### Single-Context query-value-only RL training

```powershell
git pull --ff-only origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_one_context_rl_query_value_only.ps1 -Action All -Seed 42
}
```

This isolated run restarts from the original counterfactual-trained checkpoint
(`final_enhancement_accuracy_counterfactual_trainseed42`, SHA
`c7cc1e00f9eeda6a68c1583cb41ad1a52f30cb6226dddef453092c857a88b66d`),
not from the all-head single-Context specialization. It adds
`--query-value-only-training` to the dedicated one-Context policy-feedback
sequence protocol. During TRAIN, all correctness and future-head parameters
are frozen at their inherited values. Only `update_value` requires gradients
and belongs to the TD Adam optimizer. Routing supervision is skipped during
TRAIN, so route update counters are zero. The existing TD and paired query
objectives, target-network synchronization, reward and selection cost are
unchanged. Every epoch checks exact equality of all non-update-value selector
weights with the parent; the training log records `frozen_selector_unchanged`.

Context-min adaptation still updates the single Context on acquired labels
using the original protected-memory loss. Adaptive Context, Adam, history and
memory persist across TRAIN levels 0, 1, 3 and reset at the next epoch; selector
weights and its optimizer persist. Only acquired labels and source TRAIN anchors
supervise learning. No damage guard, utility normalization or three-alternative
TRAIN simulation is enabled. The initial Context artifact and frozen CNN are
unchanged. This is a parameter-isolation experiment, not a new reward formula
or a claim that feedback scheduling or forgetting has already improved.

The frozen-head restriction applies only to the TRAIN learner. Each temporary
VAL learner and each TEST learner uses ordinary paid-feedback online adaptation
of all selector heads, matching the previous evaluation protocol. Best-epoch
selection uses the existing disjoint persistent VAL audit and clean-drop penalty,
including epoch zero. The selected checkpoint therefore preserves the parent's
correctness/future weights at evaluation start; those heads may subsequently
adapt online on paid labels. Audit and unqueried labels never train the learner.

The default output is
`final_enhancement_one_context_rl_query_value_only_trainseed42`. The launcher
uses seed 42, epsilon 0.05, a 20% per-domain label cap, Context LR 0.0005,
12 maximum epochs and patience 5. `-Action All` performs one training run with
VAL selection, then exactly two held-out TEST evaluations: independent and
sequential forgetting, both with RL query decisions. Use `-Action Train`,
`Evaluate` (independent), or `Audit` (sequential) to run individual stages.
Existing runs are never overwritten. Compare first against the original
transferred RL runs (`evaluation_525c2e425df9` independent and
`evaluation_5a3414a8f372` sequential), reporting actual labels alongside accuracy,
forgetting, BWT, clean drop and runtime. No benefit should be attributed to the
new training without accounting for changed label consumption.

CPU tests verify exclusive optimizer ownership, exact frozen-head equality
after real TD updates and checkpoint selection, update-value weight changes,
TRAIN unqueried-label invariance, unfrozen temporary VAL learners, both real
CLI TEST protocols and source/checkpoint immutability. The launcher is reviewed
statically; numerical accuracy effects require the user-host GPU experiment.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_one_context_rl_query_value_only_trainseed42"
git add -f -- "$run/enhancement_training.json" "$run/Results"
git commit -m "Add one-Context query-value-only RL training results seed 42"
git push origin AMCAL_final_enhancement
```


### Parallel query-credit experiments (three isolated runs)

Pull once before launching jobs; then run the three commands in three separate
PowerShell tabs from the repository root and the same activated environment:

```powershell
git pull --ff-only origin AMCAL_final_enhancement
```

```powershell
.\AMCAL\run_amcal_query_credit.ps1 -Variant PairOnly -Action All -Seed 42
```

```powershell
.\AMCAL\run_amcal_query_credit.ps1 -Variant Connected -Action All -Seed 42
```

```powershell
.\AMCAL\run_amcal_query_credit.ps1 -Variant LongHorizon -Action All -Seed 42
```

| Variant | TRAIN query objective | Persistent TD boundaries | Gamma |
|---|---|---|---|
| PairOnly | Paired advantage only | End each level | 0.95 |
| Connected | Paired advantage only | Bridge to next level | 0.95 |
| LongHorizon | Paired advantage only | Bridge to next level | 0.99 |

All three start independently from the original counterfactual-trained checkpoint
`c7cc1e00f9eeda6a68c1583cb41ad1a52f30cb6226dddef453092c857a88b66d`.
They train only update_value, with correctness/future heads fixed, one Context,
policy queries, seed 42, epsilon 0.05 and the existing 20% per-domain label cap.
Context adaptation, protected memory, reward, selection cost and paired loss
weight 0.5 remain unchanged. The first experiment removes only the absolute
TD gradient from the previous query-value-only TRAIN loss; its numerical value
is retained as a diagnostic. The paired target is
`utility + gamma * (V(after query) - V(skip))`. Existing warmup threshold
(32 TD replay entries), replay sampling order, gradient clipping and target
synchronization are retained. Ordinary online learning during temporary VAL
and TEST still updates all selector heads with its original losses.

The Connected variant preserves a nonterminal successor at each intermediate
persistent level boundary. That successor uses the first unlabelled adaptation
input of the next level under its actual deterministic permutation. Its Context
outputs reflect the hypothetical updated or skipped Context respectively;
reliability history is shifted into the next level's coordinates on a copy,
the next level's label cap is recharged, and position is zero. No next input
label, audit row, extra query or Context update enters this transition. Only
the last level is terminal. Boundary outputs are discarded after target
construction, and the ordinary next-level episode recomputes its prediction.
The same boundary protocol is applied to temporary persistent VAL learners
and sequential TEST; independent domains remain terminal. Connected checkpoints
require learned policy feedback. The feature lookahead extends the runner's
existing one-input TD lookahead across a known sequence boundary; it assumes
the next adaptation input is available, not an unknown future label.

Gamma 0.99 lengthens discounting relative to 0.95; this is a hypothesis test,
not a guarantee of long-term retention. Boundaries and gamma are saved in the
checkpoint training protocol and inherited automatically at evaluation. Existing
checkpoints and unflagged execution retain their previous behavior.

Each command runs one training job (up to 12 epochs, patience 5, including the
inherited epoch-zero candidate and VAL-only selection), then two TEST jobs: independent
and sequential forgetting. There are three training jobs and six TEST jobs
across the three tabs. Output directories are distinct:
`final_enhancement_query_credit_paironly_trainseed42`,
`final_enhancement_query_credit_connected_trainseed42`, and
`final_enhancement_query_credit_longhorizon_trainseed42`. Source artifacts are
read-only. If a stage is interrupted after training, use `-Action Evaluate` or
`-Action Audit` with the same variant to resume. Training never overwrites runs.
GPU contention means simultaneous-run timing cannot establish speed superiority;
compare accuracy/retention/actual labels first and time candidates separately
if needed. Parallel jobs do not guarantee shorter total wall time on one GPU.

Per-domain TRAIN logs now include `paired_target_diagnostics`: sampled replay
row count, optimizer step count, mean immediate utility, discounted future
difference, total target and predicted advantage, positive utility/target counts,
terminal count and the excluded absolute TD loss. Replayed rows are counted
repeatedly; these are training-distribution diagnostics, not unique-label statistics.
Connected trace rows additionally mark TD terminal/boundary flags. Existing
label counts still count each acquired label once.

Compare PairOnly against the prior query-value-only run, Connected against
PairOnly, and LongHorizon against Connected. Both training rollouts and online
trajectories can diverge; report actual queried labels alongside accuracy and
forgetting. CPU tests check loss isolation from arbitrary absolute TD rewards,
terminal discount masking, agreement within floating-point tolerance between a bridged successor and the
actual next-level state, absence of successor-label dependence, frozen-head
identity, TRAIN causal supervision and real CLI train/VAL/both TEST protocols.
PowerShell is reviewed statically; scientific effects require user-host runs.

After all three finish, push only reports and results (not large checkpoints):

```powershell
foreach ($variant in @('paironly','connected','longhorizon')) {
    $run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_${variant}_trainseed42"
    git add -f -- "$run/enhancement_training.json" "$run/Results"
    if ($LASTEXITCODE -ne 0) { throw "Cannot stage results for $variant" }
}
git commit -m "Add parallel query-credit experiments seed 42"
git push origin AMCAL_final_enhancement
```



### Main-paper baseline: reuse the existing binary implementation

The original-paper reference is **`mjf2019/AMCAL-paper` main/elsarticle**, not
`revision/reviewer-response`. The earlier seven-action launcher was incorrectly
labelled original-paper AMCAL. This section and the launcher replace that
identification. Seven-action experiments remain separate and are not evidence
for the original binary paper method.

```powershell
git pull --ff-only origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_paper_original.ps1 -Action All -Seed 42
}
```

This uses the already approved implementation
`AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py`, without editing its algorithm,
classes, reward, optimizer, transition, scoring, or checkpoint-selection code.
`amcal_paper_legacy_levels013.py` is a data/dispatch adapter, not a second AMCAL
implementation. It calls the existing CLI and temporarily replaces only the
training loaders passed to its existing `train_adversarial_system`; it restores
the original function afterwards and records/verifies the source-file SHA256.
It retains that implementation's existing architecture and selector state
representation rather than redesigning either from the prose.

The frozen base CNN is freshly pretrained on level-0 TRAIN data only. The
existing Context pretraining and binary-DDQN/Context training routine receives
the combined TRAIN subsets of levels 0, 1, 3 and the corresponding combined VAL
subsets. Its algorithm, warmup, training exploration, Context persistence and
early stopping are inherited. This **three-domain training dataset is the
requested experimental extension**: the main paper originally trains on level 0.
No persistent episodic-training redesign is introduced. The source run
`structured_bank3_balancedref_lr0p0005_seed42` supplies only the saved source-row
partition provenance; none of its CNN, Context or selector weights is reused.

The adapter validates source maps, feature schemas, labels and disjoint source
TRAIN/VAL/TEST identities across every domain. Prepared level-0 training rows
supply the original legacy scaler/encoder. It creates separate combined
Context TRAIN and VAL CSVs under `PreparedData`. The original training routine
prints a test observer; the data adapter supplies VAL as that placeholder so
held-out target TEST performance cannot enter training. Separate held-out
level files contain only the saved TEST IDs. Original dataset files are
never modified.

The existing runner is selected with `--reward-mode paper --transition-mode local
--gamma 0.95 --context-loss-mode aligned --training-selector-mode original`.
Paper reward is 3/2/-2/0, with zero reward for skipping, zero added selection
cost and no external prediction/confidence query gate. It has one shared
Context and two selector actions. Training labels are unconstrained by an
online label cap. Defaults retain the previous implementation's schedules:
200 maximum CNN epochs, 300 adversarial epochs, patience 30 and its internal
Context pretraining. The launcher exposes `-BaseEpochs`, `-ContextEpochs`,
and `-Patience`. No accuracy outcome is assumed.

`-Action All` trains once, then evaluates the original independent-domain
protocol on held-out levels 5, 7, 10, 12, 15, 17, 20 with seed 42. Each level
reloads the saved model. The cap is floor(20% of the source TEST count), epsilon
is fixed at 0.05, Context LR is 0.0005 and order is shuffled with seed 42.
The existing same-level/original-reference forgetting diagnostics are enabled.
This does **not** run the later persistent-sequence forgetting protocol; adding
that would require a separate adapter. `-Action Evaluate` runs evaluation only
after training. Existing seven-action results are preserved.

The old legacy runner scores after feedback. Its existing traces also cache
predictions before feedback. The data adapter writes
`Results/legacy_levels_summary.csv` with both accuracies explicitly separated,
plus base accuracy and actual labels, without changing predictions or learning.
Use its pre-feedback accuracy for comparisons with the recent enhancement runs;
do not mix the legacy post-feedback score with pre-feedback accuracy.
Metadata outputs are `legacy_data_protocol.json`, `legacy_execution.json`,
`legacy_training_summary.json`, and the original `reproduction_*.json` reports.
The default output directory is now `paper_main_legacy_levels013_seed42`;
it cannot overwrite the earlier `paper_original_levels013_seed42` run.

CPU verification covers source-ID partition isolation, invalid-map rejection,
separation of pre/post-feedback reports, actual existing legacy training and
an independent TEST level, budget enforcement and an unchanged legacy source
hash. PowerShell is reviewed statically. User-host training is required for
scientific results.

```powershell
$run="AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paper_main_legacy_levels013_seed42"
git add -f -- "$run/legacy_data_protocol.json" "$run/legacy_execution.json" "$run/legacy_training_summary.json" "$run/reproduction_*.json" "$run/Results"
git commit -m "Add existing binary-paper AMCAL results trained on levels 0 1 3 seed 42"
git push origin AMCAL_final_enhancement
```

Prepared CSVs and large model checkpoints are not needed for result analysis.

## PairOnly random-selector ablation (seed 42)

Run the existing paired-only checkpoint without retraining:

```powershell
.\AMCAL\run_amcal_paironly_random.ps1 -Seed 42
```

This runs exactly two evaluations: independent levels and a persistent sequential
forgetting test. Default checkpoint: `final_enhancement_query_credit_paironly_trainseed42`.
A single Context is retained. Each step samples uniformly from legal actions:
CNN prediction/skip, Context prediction/skip, or Context prediction/query.
Once the 20% hard cap is exhausted only the two prediction actions remain legal.
RL action/observer networks are bypassed and RL optimizer updates are disabled;
Context adaptation, protected memory and its replay continue on paid labels.
The target checkpoint uses local (not routed) reward-reference utility, so its
utility measurement also does not need any RL head. The learned differential
objective is therefore inactive at inference: this is an ablation of the
selector on the PairOnly-trained checkpoint, not a new training objective.

Independent levels restore the initial Context and memory separately; sequential
levels carry Context, optimizer and protected memory through the sequence.
The same checkpoint, seed, split, cap, Context LR and audit protocol are used as
the preceding PairOnly learned-selector evaluations. Random selection may spend
more labels than RL: compare actual counts alongside accuracy and forgetting.
The evaluation fingerprint includes `random_control.version=2` to distinguish
this control from older random runs that still executed selector networks.
Results go into separate `Results/evaluation_*` directories with `random_` files;
the learned-selector results and model checkpoint are preserved.

For a single evaluation use `-EvaluationSet Independent` or `Sequential`.
Run evaluations sequentially for comparable timing. CPU tests check the absence
of selector calls/updates, unchanged RL parameters, independence from arbitrary
selector weights, Context learning, the hard budget and both real CLI protocols.
PowerShell is reviewed statically; full-data results require the user-host run.

```powershell
$run='AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
git add -f -- "$run/Results"
git commit -m "Add PairOnly random-selector ablation seed 42"
git push origin AMCAL_final_enhancement
```

## Matched random queries with frozen learned prediction routing

```powershell
.\AMCAL\run_amcal_paironly_random_query.ps1 -Seed 42
```

No retraining: evaluate the same PairOnly checkpoint with one Context, seed 42,
20% cap and Context LR 0.0005. Runs independent and persistent sequential tests.
The `random_query` evaluation mode draws from the same legal-action RNG as the
preceding full-random experiment, using only the draw's query/skip decision.
Until budget exhaustion query probability is 1/3; afterwards it is zero.
Actual query IDs and Context adaptation match full random for this local-utility
configuration. Prediction is independently the argmax of the checkpoint's frozen
correctness head across CNN and Context, without epsilon exploration. Neither
RL weights nor optimizer states learn; the future/update-value heads are bypassed.
Paid Context adaptation, protected memory and replay remain enabled.

A queried row can predict with CNN while updating Context. Therefore `action`
encodes feedback destination on queried rows, while `route` is the actual
prediction route. `prediction_route_scores` records the frozen routing logits.
Audit/clean observers also use the frozen learned router. Separate result IDs,
`random_query_` files and `random_query_control.version=1` preserve existing runs.
Only evaluation with one Context, policy query schedule and learned destination
is supported; cascade and routed utility are rejected.

Compare this primarily against full random: matched labels and Context states
isolate the benefit of prediction routing. Comparing with ordinary learned mode
also changes query decisions and online selector learning, so it does not isolate
query selection alone. Audit initial accuracy can differ from full random because
prediction routing differs even before adaptation.

CPU tests check identical acquired source IDs and exact Context parameters across
two persistent stages versus full random, correct independent prediction during
queries, unchanged RL weights, zero RL updates, zero budget, rejection of TRAIN,
and both real CLI evaluation protocols. PowerShell is reviewed statically.

```powershell
$run='AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
git add -f -- "$run/Results"
git commit -m "Add PairOnly matched random queries with frozen learned routing seed 42"
git push origin AMCAL_final_enhancement
```

## PairOnly paid online routing and observer-only routing ceilings

```powershell
.\AMCAL\run_amcal_paironly_router_oracle.ps1 -Test Both -Seed 42
```

No training. Default `Both` runs exactly four evaluations on the existing PairOnly
checkpoint: random queries with online correctness routing (independent and
sequential), then ordinary full RL with routing-oracle reports (independent and
sequential). Use `-Test Router` or `-Test Oracle` to run one pair, and
`-EvaluationSet Independent` or `Sequential` to run one protocol. Output directory,
Context count, cap, Context LR and seed match preceding PairOnly controls.

`random_query_online` retains the exact full-random query/skip RNG and uses
correctness-head argmax for prediction without exploration. Only acquired labels
supervise correctness of the cached pre-feedback outputs of both routes, using
the existing routing loss, replay and routing optimizer. Future/update-value
parameters and target weights stay fixed; no TD training or TD optimizer step is
performed. Prediction and query remain independent, so CNN may predict on a row
whose feedback updates Context. The Context and protected memory still adapt.
This diagnostic asks whether adapting prediction routing fixes the frozen-router
collapse while keeping acquired labels matched; it is not a new PairOnly training.

`--routing-oracle-audit` adds reports only; it never changes a prediction, query,
reward, acquired label, Context update or selector update. The existing per-level
`Oracle` is the fraction of pre-feedback rows for which CNN or Context is correct.
The new `routing_oracle_summary.csv` reports mean stream oracle, actual accuracy,
their gap and label counts. In persistent evaluation it additionally reports
`FinalRoutingOracleAudit`, `FinalAudit`, their gap and clean-domain oracle. The
sequential audit matrix gains `OracleAccuracy`, `Route0Accuracy` (CNN) and
`Route1Accuracy` (Context), using the same observer outputs without extra model
forwards. Full-bank evaluation is required; cascade is rejected.

These are perfect-routing ceilings conditional on the evaluated adaptation/query
trajectory, not a maximum over other label choices, model weights, training runs
or architecture capacities. If both routes are wrong, this oracle is also wrong.
Real labels are used only by the reporting observer for unpaid rows. Do not treat
oracle accuracy as a deployable predictor or its time as a service benchmark.
Oracle metadata creates fresh result fingerprints; existing runs are preserved.
Both diagnostic and full-RL runs receive oracle reports, allowing their routing
gaps to be compared without introducing an oracle-controlled learning trajectory.

CPU tests verify paid-only routing updates, exact matched queries and Context
states, frozen non-routing/target weights, invariance to unpaid labels, zero-label
behavior, persistent continuation and both real CLI protocols. Oracle tests
recompute the stream ceiling from saved predictions, bound audit scores by their
oracle, and verify byte-equivalent parsed traces with versus without reporting.
PowerShell is reviewed statically; full-data outcomes require the user-host run.

```powershell
$run='AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
git add -f -- "$run/Results"
git commit -m "Add PairOnly online routing and routing oracle audits seed 42"
git push origin AMCAL_final_enhancement
```

## PairOnly full RL with balanced protected routing replay

```powershell
.\AMCAL\run_amcal_paironly_protected_router.ps1 -Seed 42
```

Two evaluations only (independent levels and persistent forgetting), no retraining,
same PairOnly checkpoint, seed 42, one Context, policy queries, 20% cap and Context
LR 0.0005. `-EvaluationSet Independent` or `Sequential` selects one protocol.
Both evaluations include observer-only routing oracle reports. The full RL action
selector and ordinary TD learning remain active. Queries are not forced or random.

`--router-protected-replay` changes only the correctness-head routing loss. Each
paid step already samples a balanced reward reference. Reuse its source and
historical rows (exclude recent rows and the current input), then recompute their
route correctness with the bank AFTER its ordinary paid Context update. Build
fresh reference states using the pre-feedback reliability snapshot. No extra
sampling changes memory RNG or the Context replay trajectory. No extra labels,
reference capacity, Context optimizer steps or routing optimizer steps are added.
Protected routing evaluation does add batched CNN/Context forwards.

Within that existing routing optimizer step use:

`0.5 * paid-route-replay loss + 0.5 * fresh protected routing loss + original prior`.

Both routing losses retain BCE and the existing ranking term. Protected weights
give equal mass to available source/historical segments and equal class mass
inside each segment. Missing segments/classes do not receive synthetic rows.
If there is no eligible protected reference, retain the original routing loss.
Recomputed protected targets are used directly, not stored as stale route replay.
TD heads keep their original optimizer and losses; the extra protected routing
loss itself changes only routing parameters. This differs from `--router-refresh`,
which performs an additional separate routing step on the whole reference batch.
The two ablations cannot be combined.

Because routing affects RL action values and later states, learned queries and
actual label spending may change versus the baseline. Compare accuracy,
forgetting, clean-domain drop, oracle gaps, actual labels and cost together.
The principal baseline is the same PairOnly full-RL checkpoint, not the
random-query diagnostic. This is an evaluation-only ablation; it does not alter
saved training or overwrite earlier results.

`execution.json` records `router_protected_replay.version=1`. Each metrics JSON
adds `protected_routing_replay` with optimizer-step counts and source/historical
row exposures (repeated rows count repeatedly). Existing oracle and independent /
sequential reports remain available under fresh evaluation fingerprints.

CPU checks cover a single joint optimizer step, routing-only gradients from the
new loss, unchanged target/TD weights in isolation, exact matched Context weights
under fixed queries, invariance to unpaid TEST labels, historical/source exposure,
zero budget, TRAIN rejection and real CLI evaluation in both protocols. Existing
routing/oracle controls are regression tested. PowerShell is reviewed statically;
full-data accuracy/retention and latency require the user-host run.

```powershell
$run='AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
git add -f -- "$run/Results"
git commit -m "Add PairOnly protected routing replay results seed 42"
git push origin AMCAL_final_enhancement
```

## Protected routing replay at 20% (single follow-up)

```powershell
.\AMCAL\run_amcal_paironly_protected_router.ps1 -ProtectedWeight 0.2 -Seed 42
```

Run exactly two evaluations (independent and sequential), no retraining. Keep the
same PairOnly checkpoint, RL query selection and all previous settings, replacing
only the routing-loss mixture with `0.8 paid replay + 0.2 fresh protected replay`
before adding the original routing prior. No changes to memory sampling, Context
updates, TD learning, label cap, epsilon, seed or oracle-reporting protocol.
Acquired labels may differ because the updated correctness head participates in
subsequent full-RL actions; the cap remains unchanged.

The old launcher's default `-ProtectedWeight 0.5` is preserved. Nondefault values
are passed as `--router-protected-weight` and recorded in execution metadata as
`protected_fraction`, producing distinct evaluation IDs without overwriting the
50% reports. Per-level replay diagnostics record `protected_weight`. The Python
argument must be finite and in (0,1], and a nondefault weight requires protected
routing replay to be enabled. PowerShell formats the value with invariant culture.

The 50% trial had independent accuracy 70.76288% with 1126 labels, sequential
online 62.46466%, final audit 68.22430%, forgetting 3.27103 pp, clean drop 2.70270
pp and 224 labels. The original PairOnly full-RL baseline remains independent
70.08388% with 805 labels; online 68.30201%, final audit 70.62750%, forgetting
5.52960 pp, clean drop 6.24418 pp and 305 labels. Compare the new run to both;
reduced forgetting alone is not an accuracy improvement.

CPU checks verify the numeric 80/20 routing loss, paid-only supervision and
matched Context behavior at weight 0.2, one routing step per acquired label,
separate 20%/50% fingerprints and actual independent/sequential CLI outputs.
The unchanged controls are regression tested. Full-data outcomes require the
user-host run; PowerShell is reviewed statically.

```powershell
$run='AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
git add -f -- "$run/Results"
git commit -m "Add PairOnly protected routing replay weight 0.2 seed 42"
git push origin AMCAL_final_enhancement
```

## User-configurable protected and normal routing weights

```powershell
.\AMCAL\run_amcal_paironly_protected_router.ps1 -ProtectedWeight 0.3 -NormalWeight 0.7 -Seed 42
```

Both are relative nonnegative weights, normalized before mixing losses. Thus
`-ProtectedWeight 2 -NormalWeight 8` and `0.2 / 0.8` are the same 20% protected
experiment, with the same result fingerprint. The total must be positive. If
`-NormalWeight` is omitted, it defaults to `1 - ProtectedWeight`; the protected
weight must then lie in [0,1]. The old no-argument default remains 50% / 50%.
The original routing prior is added after the normalized mixture, so relative
weight magnitude does not change its scale or the effective routing learning rate.

Zero protected weight disables protected batch construction and extra protected
forwards, giving the exact ordinary full-RL trajectory while recording a separate
zero-weight control. Zero normal weight trains routing from protected examples
only; paid Context updates and ordinary TD still run. Existing sampling, Context
adaptation, RL queries, epsilon, caps, seed and oracle reports are unchanged.

Run untested weights sequentially with one seed:

```powershell
foreach ($p in @(0.05, 0.1, 0.3, 0.4, 0.7)) {
    .\AMCAL\run_amcal_paironly_protected_router.ps1 -ProtectedWeight $p -NormalWeight (1-$p) -Seed 42
}
```

Each weight runs exactly two evaluations by default. Add
`-EvaluationSet Independent` or `Sequential` to test one protocol first. Already
completed equivalent settings are rejected instead of overwritten; weights 0.2
and 0.5 have already been run. Do not include them again in a batch unless using
a separate output copy. Both weights are serialized to CLI with invariant culture;
execution metadata records the effective protected fraction (0.5 is the existing
legacy metadata default) and the mixture text, so different ratios remain distinct.
Metrics also record the normalized protected weight. Results stay under the same
PairOnly `Results/evaluation_*` directories and use the same push command.

Seed-42 20% results: independent accuracy 71.05578%, labels 1116; sequential online
63.86163%, final audit 71.62884%, forgetting 2.49221 pp, clean drop 5.31221 pp,
labels 375. These are exploratory TEST comparisons; use VAL for a final weight
choice before reporting a final held-out estimate.

CPU tests cover relative normalization, invalid/zero totals, endpoints, unchanged
80/20 loss, equivalent ratio fingerprints, exact baseline traces/weights at zero
protection and real CLI protocols. PowerShell is reviewed statically.

## Resume completed weight evaluations safely

The protected-router PowerShell launcher now passes `--skip-existing-completed`.
Repeating the same weights/seed skips a complete independent evaluation and then
continues to the sequential evaluation (or skips it too if complete). The flag
does not change result fingerprints or learner behavior. Other launchers retain
their existing strict duplicate handling unless they explicitly pass this flag.

A preexisting directory is not enough: reuse requires matching execution metadata,
valid per-level metrics, complete trace row counts, combined results and, when
requested, all sequential and oracle reports. The completion metadata is written
last by the evaluator. Missing, malformed or truncated artifacts are rejected and
preserved. No result is overwritten or deleted.

For a partial attempt, keep seed 42 and use a fresh tag:

```powershell
.\AMCAL\run_amcal_paironly_protected_router.ps1 -ProtectedWeight 0.05 -NormalWeight 0.95 -Seed 42 -EvaluationTag retry1
```

Use `-EvaluationSet Sequential` if only sequential evaluation needs a fresh attempt,
or `Independent` for the independent case. Repeating the same tagged command can
also skip its complete outputs. Do not run duplicate identical settings concurrently.
CPU checks verify complete repeats do not call the episode or rewrite files;
missing oracle reports, mismatched metadata and truncated traces are not reused.
PowerShell is reviewed statically.
