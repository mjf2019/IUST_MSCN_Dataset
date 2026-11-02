# AF checkpoint calibration: online scores and retention

The clean initialization remains unchanged: 25 clean TRAIN rows per class and
exactly floor(20% of clean TRAIN) disjoint calibration rows. Its corrected author
AF fit creates a reusable level0 checkpoint. Clean VAL is observer-only.

## Causal target evaluation

Every target intensity reloads the same clean checkpoint. A seed-only permutation
matches the baseline evaluator's ordering rule; there is no future-label class
balancing. Rows identical to clean fitting features are excluded and their count
is recorded. The budget is floor(20% of this eligible stream), reset per intensity.

1. Predict the first budget-sized prefix using the clean checkpoint and its clean
   kNN support. Freeze these predictions BEFORE exposing its labels to adaptation.
2. Once this prefix's feedback has arrived, perform ONE author AF batch calibration
   from the clean checkpoint, with fresh optimizers, followed by author feature
   reset/supervised tuning. Only prefix labels enter the fit. Keep the full source25
   support and the acquired target support. Do not use later stream labels.
3. Predict the remaining suffix with the adapted extractor and target-only kNN.
4. Compute online AC and macro PR/RC/F1 over ALL eligible stream rows, INCLUDING
   the calibration prefix. Never replace prefix predictions with adapted ones.

Batched prediction is causally equivalent to individual prediction within each
phase because parameters/support do not change within either phase. This retains
AF batch calibration: it is NOT a gradient update after every individual feedback.
The acquisition schedule is a prefix, not uncertainty, random sampling of future
positions, or an oracle class-balanced selection.

online_trace.csv records row ID, position, immutable pre-feedback prediction,
label request, labels acquired before prediction, model phase and the update
boundary. Console output identifies online scores and prefix calibration accuracy.
Actual class counts of acquired support are recorded even if a class is absent;
there is no resampling to improve results or access future labels.

## Paired observers

Original-domain forgetting compares the initial/final predictor on the same clean
VAL reference, excluding fitting IDs and exact fitting feature duplicates across
all cases. Same-domain forgetting uses the same unqueried target observer rows
before/after, excluding prefix supports and exact fitting feature duplicates.
Observers do not feed training, query decisions, or checkpoint selection.

For AC/macro PR/RC/F1, forgetting = initial minus final and gain = final minus
initial, in signed percentage points. Negative forgetting means improvement.
same_domain_final_AC is a final observer score; online_AC is the full pre-feedback
stream score. They are different metrics and remain separate.

## Preservation and resume

Defaults write to AF_checkpoint20_online_seed42_v2. Existing offline-v1 results
stay unchanged. If the previous AF_checkpoint20_seed42_v1 clean checkpoint exists,
its hash, fitting inputs, seed, preprocessing, support IDs and training settings
must match before reuse. No clean retraining is needed in that case. Otherwise
train just the clean AF initialization, then the per-target continuations.
AMCAL/CNN/Context/DDQN are never trained or modified.

Online traces cannot be reconstructed from the old offline target predictions:
the causal prefix and target continuations need a new evaluation. A completed v2
clean checkpoint and completed target cases are hash-verified and skipped on rerun;
partial target fits restart from clean, never from another target case.

This is a checkpoint-initialized batch-calibration variant of the corrected author
AF port, not a claim of bit-identical reproduction of the original offline paper.

## Run and push

All means SDNCampus, ISCX-TOR, IUST and IoT, with all 12 attack/intensity cases.

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_af_clean_reference.ps1 -Dataset All
.\AMCAL\push_af_clean_reference_results.ps1 -Dataset All
```

For Burst use -Dataset IAT or PS, -SourceRoot pointing to the existing structured
run and -BurstDir pointing to official AdvBurst_FTSC-IAT/PS. All seven target
intensities 5/7/10/12/15/17/20 are evaluated. The same existing commands apply.
The push script includes only this v2 audit's JSON/CSV files, not model binaries.

Optional tests, executed by the user:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_af_clean_reference.py -v
```

Tests cover prefix-before-update event order, inclusion of calibration predictions
in the online denominator, exact budget, initial checkpoint loading, clean data
separation and signed forgetting. The assistant did not run Python, tests,
training, imports, inference, or package installation.
