# Clean-only training on supplied network attack CSVs

Run the new isolated adapter; prior supplied-dense scripts train on attacks too
and must not be used for this experiment.

```powershell
.\AMCAL\run_amcal_clean_supplied.ps1 -Dataset All -Action All -Seed 42 -Device cuda -AssumeHistoricalPoolOrder
```

Dataset keys: SDNCampus, ISCX-TOR, IUST, IoT. Roots are SDNCampus_TEST,
ISCX_TOR_TEST, IUST_MSCN_TEST, UNSW_IoT_TEST. Clean files are the matching
*_original.csv already present in Dataset. No attacks are generated or downloaded.

Training and validation for CNN, Context, RL and baseline initialization use
only the clean original CSV. AMCAL uses original CNN architecture on all four
datasets, one Context/three final actions, PairOnly batch5, bonus1,
online LR .00025 and protected replay CE .1. Existing training-stage defaults
and stage recipe are reused. All weights start fresh; no pretrained IAT/PS
model is imported. AMCAL source and training-stage checkpoints resume only when
their recipe/input/output hashes match; partial stages require a fresh output.

CW/DeepFool TEST strengths are .1,1,5. FGSM/PGD TEST strengths are .01,.1,.5.
Each attack has ordinal engine IDs 1,2,3 in that order, NOT the old 1..5 map.
Actual strengths are recorded in the manifest and combined comparison CSV.
TrainingDomains contains only clean level0; EvaluationDomains contains attack
CSVs. Every attack/strength starts again from the same clean-trained weights,
empty online feedback, fresh optimizers/selector state and floor(.2*N) label cap.
No continual mode or cross-level transfer. Cap is a maximum, not forced usage.
Prediction is recorded before the current label is acquired. Clean VAL is used
for source-retention observation; no TEST label selects checkpoints/parameters.

Historical supplied attacks cover the stratified seed42 20% source pool.
Use explicit *_rows.csv maps when available. Without them, the explicit
-AssumeHistoricalPoolOrder flag is necessary: matching labels does NOT establish
provenance. The old seed42 80/20 pool order is an UNVERIFIED assumption inherited
from these supplied datasets. The entire attack pool is held out from clean
training; remaining clean rows are split 75/25 into TRAIN/VAL. Exact-feature
duplicate groups crossing splits are purged TRAIN > VAL > TEST. The same TEST
identities/order and clean TRAIN-fitted scaling are used for all methods.
Identity/IP/MAC/port/time/generation columns are excluded using the existing
exclusion rules. Nonfinite/schema/mapping/class-coverage failures stop execution.

AF caveat: original AF is offline target-calibrated few-shot learning and cannot
simultaneously be original AF and a causal 20%-budget online baseline. This task
uses the existing tabular AF historical online adapter, explicitly named
AF-clean-source-online-extension. Since attack data are forbidden before test,
initial AF fitting uses supervised CE on clean TRAIN only (30 epochs, Adam1e-5,
same network); target domain alignment starts only on acquired online feedback.
No synthetic target, clean-as-shift workaround, or unqueried target feature is
used for training. The source support/kNN and paid-feedback DANN update are
the existing implementation. Never present this as original-paper AF.
DFE reuses the existing online adapter unchanged: clean-source representation
training; paid target labels update its template library while backbone stays
fixed. It must likewise be labelled as an adapted online experiment.

Outputs: each dataset's protocol_runs/clean0_paironly_batch5_attacks_seed42/.
The top-level Results/clean_attack_comparison.csv contains 36 rows per dataset.
AF/DFE do not require completed AMCAL RL training. Use -Methods AF,DFE separately
after source initialization, or -Methods AMCAL for AMCAL only. Original source
initialization is shared. Repeat -Action Evaluate to evaluate saved models.
Never push Models or exported Domains; push JSON manifests and result CSV/JSON.
No numerical experiments were executed; Python syntax/static checks only.
