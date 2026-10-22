# PairOnly batch5 with synthetic drift from clean Burst level 0

```powershell
.\AMCAL\run_amcal_synthetic_paironly.ps1 -Dataset IAT -BatchSize 5 -Action All -Seed 42
.\AMCAL\run_amcal_synthetic_paironly.ps1 -Dataset PS -BatchSize 5 -Action All -Seed 42
```

This is a separate fast pilot; old runs and checkpoints are preserved.
All stages develop on clean 0 and synthetic drift derived only from its TRAIN/VAL
source IDs. Official attack levels 1/3/5 are never used in training or selection.
TEST levels are 3,5,7,10,12,15,17,20 with independent and continual retention audits.
Real attack files are read only after all checkpoint selection completes.

Only source schema, labels and disjoint source IDs are inherited from the old
preprocessing artifact. StandardScaler is refitted on clean-0 TRAIN. No CNN,
Context or selector weights are inherited from earlier attack-trained models.
IAT CNN is freshly initialized original CNN; PS uses the light CNN architecture.
CNN and initial Context fit clean level 0 only (20/5 maximum epochs, VAL loss
patience 3). Existing structured -> accuracy -> counterfactual -> one-Context
PairOnly recipe is rebuilt on 0/synthetic (6 epochs per main stage, utility 2).
The final method has one Context and three actions. Batch5 is used in its final
PairOnly fitting and TEST; source and earlier warmup stages keep their original
update recipe. Context LR .0005, 20% online label cap, epsilon .05.

Synthetic transformation is label-independent and fixed before training:
feature-wise gain in [-.25,.25], shift in [-.3,.3] clean-TRAIN scaler units,
Gaussian noise sigma .1 and four severity blocks determined by source row ID.
Features nonnegative in clean TRAIN are clipped at zero after inverse scaling.
This is a feature-space robustness pilot, not a claim of physically realizable
network attacks. No parameter is chosen from real attack test results.
Synthetic file level 1 is a storage placeholder, not official Burst level 1.
Only TRAIN/VAL rows are generated for it; original TEST identities stay held out.
Source clean file is unchanged; no unrelated network identity features are added.

Output: ANT_Burst_{IAT|PS}_TEST/protocol_runs/
paironly_synthetic0_batch5_seed42/paironly_batch5/Results.
Input/config/output hashes protect resume; partial stages are preserved and
require a new OutputRoot. Report this as a fresh-source synthetic-training pilot;
its comparison with old runs also changes the source initialization/training
schedule, so improvement cannot be attributed solely to drift augmentation.
Only syntax checks were performed, per user request; run tests on the user host.
