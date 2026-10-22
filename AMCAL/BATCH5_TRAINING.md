# PairOnly batch5 training and PS light frozen classifier

Branch: experiment/paironly-paid-replay. These are new isolated runs; no old
models or evaluation directories are overwritten. Context LR .0005, alpha .8,
gamma .95, epsilon .05, seed 42 and 20% per-domain feedback cap remain fixed.

## IAT

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset IAT -Action All -Seed 42
```

Uses the existing IAT counterfactual checkpoint in
final_enhancement_accuracy_counterfactual_trainseed42. Repeats the final one-Context
PairOnly specialization with batch5 updates during TRAIN and temporary VAL
adaptation, then independent TEST and persistent sequential audit. Training
levels remain 0/1/3; test levels 5/7/10/12/15/17/20. Only update_value is trained
by the paired objective in the specialization; correctness/future inherited heads
remain frozen during that specialization. Ordinary paid-only online RL remains
active during validation and test. The inherited epoch-zero candidate remains
eligible for VAL selection; inspect selected_epoch instead of assuming fitting won.

Output: AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paironly_batch5_trainseed42.
Final checkpoint/results: paironly_batch5/Models and paironly_batch5/Results.
This is matched-update retraining, not the earlier evaluation-only batch5 control.

## PS

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset PS -Action All -Seed 42
```

Reuses only the original PS source partition/preprocessing, not the old CNN,
Context or selector weights. The fixed source recipe:

1. New light CNN: Conv widths 16/32/64, GroupNorm, global average pooling,
   a 128-wide selector embedding, dropout and label smoothing .05. AdamW with
   weight decay .01, maximum 60 epochs, early stopping patience 5 by VAL loss.
   Uses only source-disjoint TRAIN rows at levels 0/1/3; no TEST scoring.
2. Fresh original Context architecture is supervised against the frozen new
   CNN on those TRAIN domains; VAL-only selection, maximum 20 epochs.
3. Fresh structured selector; accuracy enhancement; counterfactual utility;
   finally one-Context PairOnly batch5 specialization. Earlier warm-start stages
   retain their original update rule; batch5 applies to final PairOnly TRAIN,
   its temporary VAL learners, and both final TEST protocols.
4. Independent TEST and persistent sequential forgetting audit. The frozen
   CNN remains immutable during all online adaptation.

For width 100, the light model has 16,674 parameters with 2 classes and 18,222
with 14, versus 1,211,586 and 1,213,134 in the original architecture. This reduces
capacity, but does not establish better generalization, accuracy or forgetting.
Source CNN training uses batch 32; this is separate from Context feedback batch5.

Output: AMCAL/ANT_Burst_PS_TEST/protocol_runs/light_ps_paironly_batch5_trainseed42.
Subdirectories: source, structured, accuracy, counterfactual, paironly_batch5.
The final adaptive metrics are under paironly_batch5/Results/evaluation_*.
Results/frozen_generalization.csv compares old/new frozen CNN on clean TRAIN,
VAL and TEST and shifted TEST domains after all training is finished. These
observer-only scores never select a checkpoint or enter learner memory.
source/source_training.json records train/VAL loss and accuracy curves, parameter
counts and the explicit source recipe. The table permits checking whether the
old-model overfitting hypothesis is supported, without assuming architecture
size is the only cause. Comparison with old adaptive runs changes both the source
model and final training update rule; it is not a one-factor ablation.

## Batch and metrics

One newly acquired label plus up to four distinct old paid recent/historical
rows, sampled without replacement, form the primary batch. Short/empty buffers
use available unique rows. The current input is excluded from old replay. No
unqueried labels or audit labels are included. One Adam step per acquired label;
existing protected replay remains an additional loss (not counted in the five
primary rows). Reference transfer reward and paired successor see the complete
batch update. No extra label is charged. An independent RNG is retained.

Batch5 is recorded in training_protocol.context_update. Evaluation inherits it
when no explicit mode is supplied and rejects conflicting update modes. Existing
checkpoints and their default update rule remain unchanged. Full-model backbone
loading recognizes only explicit light_ps_cnn_v1 or original legacy architecture;
there is no silent partial weight loading.

Inspect online accuracy/macro-F1 and label usage from combined_results.csv.
The sequential audit additionally produces sequential_summary.csv and
sequential_audit_matrix.csv: final audit accuracy, forgetting, BWT and clean drop.
Use the same seed/order/audit identities for comparisons. Do not substitute
same-level post-update diagnostics for the fixed independent audit.

## Resume

-Action Train trains only; -Action Evaluate runs independent evaluation only;
-Action Audit runs persistent evaluation only. Keep training knobs unchanged on
resume. Verified complete stages are skipped by input/model/log hashes. Partial
or mismatched stages are preserved and rejected: choose a new -OutputRoot.
For interrupted final evaluation, use -EvaluationTag retry1. No training is
started by Evaluate or Audit. Run IAT/PS sequentially on one idle GPU if timing
will be compared. -SourceRun can override the existing source path.

## Validation

Run CPU integration tests before large jobs:

```powershell
python -m unittest discover -s AMCAL/tests -p test_batch5_pipeline.py -v
python -m unittest discover -s AMCAL/tests -p test_paid_replay.py -v
```

The integration suite executes real CNN/Context/selector training, saving,
reloading, independent and sequential evaluation on tiny synthetic aligned CSVs.
It checks no TEST access during TRAIN, source immutability, light model loading,
paired-only specialization, batch5 TRAIN/TEST matching/inheritance, budget caps,
TEST identity isolation, frozen CNN invariance and verified completion/resume.
Seven CPU tests passed in this environment (five integration/regression tests plus two paid-replay tests). A pre-existing sampler fixture was corrected to allow two historical rows of the same class; the production memory policy was unchanged. These correctness checks are not scientific benchmark results. PowerShell is
reviewed statically; full IAT/PS GPU training/evaluation runs on the user machine.

Batch3 and batch7 are also supported by the same entry point:

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset IAT -BatchSize 3 -Action All -Seed 42
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset IAT -BatchSize 7 -Action All -Seed 42
```

Each uses one current paid row and up to BatchSize-1 distinct old paid rows,
one optimizer step, and the existing protected replay loss. The batch mode is
applied during final PairOnly TRAIN and inherited/matched during independent
and continual TEST. Fewer retained paid rows means a smaller initial batch.
Default BatchSize=5 and its output paths/recipe remain compatible. Sizes 3/7
write separate paironly_batch3_trainseed42 / paironly_batch7_trainseed42 roots.
Only syntax checks were run for this addition, per user instructions.

Explicit user-requested IAT Light CNN comparison, matching the PS architecture
and TRAIN 0/1/3 recipe. Source CNN/Context and selector stages train from scratch;
only source preprocessing/partition artifacts are copied from the existing run.
Final PairOnly uses batch5 and TRAIN Context LR .0005. Only TEST uses Bonus=1,
protected replay .1, Context LR .00025, with independent and continual audits.

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset IAT -LightCNN -BatchSize 5 -Action All -Seed 42 -Bonus 1 -ReplayWeight 0.1 -OnlineContextLR 0.00025 -EvaluationTag light_iat_bonus1_lr0p00025
```

Output root: light_iat_paironly_batch5_trainseed42. Real TRAIN levels 0/1/3;
TEST 5/7/10/12/15/17/20. No synthetic drift. IAT defaults remain original CNN
unless -LightCNN is explicitly passed. PS architecture remains Light CNN.
No execution tests run; Python syntax only.

PS original frozen CNN comparison (explicit user request):

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset PS -OriginalCNN -BatchSize 5 -Action All -Seed 42 -Bonus 1 -ReplayWeight 0.1 -OnlineContextLR 0.00025 -EvaluationTag original_ps_final_budget20
```

Fresh original CNN and Context TRAIN/VAL levels 0/1/3, same development recipe
and optimizer as the light PS comparison. Freeze CNN after pretraining. Final
one-Context PairOnly batch5; online bonus1, replay .1, LR .00025, budget .2.
Output root original_ps_paironly_batch5_trainseed42. TEST levels
5/7/10/12/15/17/20, independent plus continual. Existing light runs untouched.
Only source preprocessing/partitions inherited, no model/selector weights.
No execution tests run; Python syntax checked.

IAT clean-level-0-only complete training (explicit user request):

```powershell
.\AMCAL\run_amcal_batch5_pipeline.ps1 -Dataset IAT -CleanOnly -BatchSize 5 -Action All -Seed 42 -Bonus 1 -ReplayWeight 0.1 -OnlineContextLR 0.00025 -EvaluationTag clean0_final_budget20
```

Fresh original CNN, Context, structured selector, accuracy warmup, counterfactual
utility and final one-Context PairOnly all use only clean level 0 TRAIN/VAL.
No official 1/3 or synthetic drift examples are used for fitting or selection.
Existing preprocessing supplies only the schema, label encoder and disjoint source
IDs. StandardScaler is refitted on clean TRAIN; previous model/selector weights
are never loaded for training. Source CNN/Context and all selector stages are
newly trained; CNN is frozen thereafter. TRAIN Context LR .0005, TEST .00025.
Other settings match the main batch5 recipe. TEST now includes 1 and 3:
1,3,5,7,10,12,15,17,20. Independent and continual tests retain held-out identities.
Output root: clean0_original_iat_paironly_batch5_trainseed42. Original runs stay
unchanged. Input attack hashes in the run plan are integrity checks, not training
examples. No execution tests run; Python syntax only.
