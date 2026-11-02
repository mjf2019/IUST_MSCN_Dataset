# AF checkpoint-based retention

Train one clean AF checkpoint per dataset/seed: 25 clean TRAIN rows per class,
then corrected author AF adaptation/tuning with exactly floor(20% of clean TRAIN)
calibration rows. Source25 and calibration IDs/features are disjoint. Record the
denominator and every selected ID. Clean VAL is observer-only.

For EVERY target intensity, reload this same clean checkpoint, perform the author
two-optimizer adaptation and supervised tuning with 20% target calibration labels,
then discard the resulting model before the next intensity. Optimizers are fresh
per intensity. The optional fit(initial_state=...) path initializes all AF network
weights/buffers from clean; historical calls remain fresh fits. Author feature-layer
reset before supervised tuning is retained. This is a new checkpoint-initialized
AF experiment, not numerically identical to the author's fresh-joint-fit protocol.

The complete AF predictor includes the extractor and target-only cosine kNN:
before adaptation use clean calibration supports/k; after adaptation use acquired
target supports/k. Both are scored on identical held-out target rows and on a
common original-domain VAL reference. Target calibration rows and exact fitting
feature duplicates are excluded from evaluation. No target/VAL evaluation labels
enter fitting, support selection hyperparameter tuning, or model selection.

Original forgetting = original-domain score before minus after adaptation.
Same-domain forgetting = held-out target score before minus after adaptation.
Gain is the negative of the corresponding forgetting. All AC/macro PR/RC/F1
differences are signed percentage points; improvement is not clipped to zero.

This code trains one clean AF and one continuation per target case. Existing
independently fitted AF target results cannot provide these paired measurements
and remain unchanged. It trains no AMCAL/CNN/Context/DDQN models. Offline target
calibration results are not pre-feedback online metrics under a stream budget.
Initial clean calibration is additional offline supervision, recorded separately.

## Run and push

All means SDNCampus, ISCX-TOR, IUST and IoT (12 cases each).

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_af_clean_reference.ps1 -Dataset All
.\AMCAL\push_af_clean_reference_results.ps1 -Dataset All
```

Run/push separately by replacing All with the dataset name. Defaults use existing
clean0_AMCAL_attacks_seed42 data/preprocessing. Checkpoints and support IDs stay
local; push only JSON/CSV provenance, predictions and metrics from this experiment.

For Burst, use -Dataset IAT or PS, explicit -SourceRoot pointing to a structured run
with Models/protocol_preprocessing.joblib and -BurstDir pointing to official
AdvBurst_FTSC-IAT/PS. All seven unseen intensities 5/7/10/12/15/17/20 are evaluated.
No prior target AF run is required. Its calibration is exactly 20% of eligible
target TEST rows, with the remaining rows held out; it does not reuse historical
five-shot target support. Training of the initial AF reference uses level0 only.

Optional user-run tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_af_clean_reference.py -v
```

The assistant did not execute Python, imports, training, tests or installations.
Console output identifies the dataset/intensity, clean/target baseline, adapted
scores, labels and both signed forgetting values. Completed clean and per-case
checkpoints are hash-verified and skipped on reruns; partial fits restart only that
case from the clean checkpoint. Any recipe change requires a fresh output directory.
