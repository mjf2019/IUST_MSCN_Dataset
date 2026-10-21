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
