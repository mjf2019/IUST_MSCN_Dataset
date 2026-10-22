# Source-only NPBO and DReLAB PairOnly pilots

Current protocol: adversarial examples are TEST-only. All initial CNN/Context
training, selector warmup, paid PairOnly fitting, and model selection use only
benign and direct attacks. The legacy target_train/target_val/audit_val pool names
now refer to source-domain benign/direct data. Training visits source once per
pass, not twice through aliases. Runtime assertions reject adversarial TRAIN/VAL.

Original group assignments are made before filtering adversarial TRAIN/VAL rows.
Those adversarial rows are discarded, never moved to TEST. Thus using Fraction=1
reads the entire scenario but does not evaluate derivatives of training groups.
DReLAB retains immutable-feature groups, with fixed seed role-stratified group
60/20/20 splitting. NPBO uses exact raw-feature groups; original/obfuscated
identity linkage is NOT verified from these CSVs. Exact processed-feature overlap
checks still apply; session/derivative independence is not claimed for NPBO.

DReLAB keeps the whole-group processed overlap purge (TRAIN > VAL > TEST,
observers > stream). No rows move between partitions; preprocessing stays fitted
on source TRAIN only. VAL and TEST pools are independently balanced 50:50
benign/attack, after purging, without replacement and with a seeded permutation.
Each DReLAB evaluation pool requires at least 20 rows per class; fail otherwise.
VAL attacks are direct; only TEST target stream and audit attacks are adversarial.
Source TEST audit uses benign/direct rows disjoint from adaptation groups.
These are balanced case-control results, not natural-prevalence accuracy.

Commands from the repository root:

```powershell
.\AMCAL\run_amcal_external_paironly.ps1 -Dataset NPBO -Action All -Seed 42
.\AMCAL\run_amcal_external_paironly.ps1 -Dataset DReLAB -Action All -Seed 42 -Fraction 1 -MaxTrainRows 0 -MaxStreamRows 0 -MaxAuditRows 0
```

NPBO reads its full CSV. DReLAB currently covers CTU/Neris RF/DDQN only; Fraction=1
reads all three scenario CSVs, not other attack families. Prepared fraction CSVs
and group maps are reused, with original hashes checked. The effective splitting
policy is recorded in the external protocol; old prepared manifests may describe
the earlier splitting policy, which is superseded by this run's protocol.

New defaults prevent mixing legacy adversarial-trained results:
- ASNM_NPBO_TEST/protocol_runs/paironly_sourceonly_seed42
- DReLAB_TEST/protocol_runs/paironly_sourceonly_CTUNeris_f1_seed42

Source CNN/Context use inverse-frequency weighted CE, no smoothing, globally
normalized TRAIN weights. Preprocessing: signed log, source TRAIN medians,
source TRAIN ANOVA at most 64 nonconstant features and StandardScaler.
Label, ID, IP, MAC, raw port, absolute time, host identity proxies and generation
metadata remain excluded. Context LR .0005; 20% TEST label cap. One Context,
three actions, original single-row Context update with protected replay; no batch5.
Selector warmup defaults to 3 epochs, PairOnly specialization to 6.

Results include online accuracy/Macro-F1 vs frozen, fixed target audit
accuracy/F1/attack recall, source before/after accuracy/F1/recall, and label use.
Source drop measures retention of benign/direct behavior; one target domain
cannot establish multi-domain BWT. Inspect role counts and purged counts.
Previous pilot outputs remain separate and are not unseen-attack evaluations.
No execution/numerical tests were run for this change, per user request;
only Python syntax was checked. Actual dataset split feasibility must be verified
on the user's machine.
