# NPBO and fast DReLAB PairOnly pilots

From the repository root on experiment/paironly-paid-replay:

```powershell
.\AMCAL\run_amcal_external_paironly.ps1 -Dataset NPBO -Action All -Seed 42
.\AMCAL\run_amcal_external_paironly.ps1 -Dataset DReLAB -Action All -Seed 42 -Fraction 0.1 -MaxTrainRows 2000 -MaxStreamRows 1000 -MaxAuditRows 5000
```

NPBO uses all raw data (11,445 records) without fractional sampling/caps by
default. Fixed held-out source/target/audit partitions still reserve records from
training. DReLAB starts with CTU/Neris, RF/DDQN attacks only. No attacks are
regenerated. Both use simple one-Context three-action PairOnly, Context LR .0005,
20% label cap, original single-acquired-row update and protected replay. No batch5.

For speed, source-only preprocessing selects at most 64 nonconstant source-TRAIN
features, with signed-log transform, TRAIN medians and TRAIN scaling, matching the
TUN pilot. Label/ID/IP/MAC/raw ports/absolute times and generation metadata stay
excluded. The selector warmup is 3 epochs and paired-only specialization 6 epochs
by default (override WarmEpochs/PairEpochs). Source CNN/Context use existing pilot
maximum epochs and VAL-only early stopping. No TEST score selects a model.

DReLAB reads the large benign CSV in chunks. It assigns immutable-feature groups
to TRAIN/VAL/TEST via a fixed hash rule, independently of sampling; then uses
seeded within-file 10% Bernoulli sampling. Actual percentages/counts are recorded,
not assumed exactly 10%. A group includes every flow with the same fields that
DReLAB's published attack leaves unchanged. This is a deliberately conservative
equivalence group, NOT a verified one-to-one original-flow ID. Attack immutable
keys must exist in original Neris data. Host-type fields may be used for grouping
only, never model inputs. No CSV index is assumed to prove derivative identity.

The sample's TRAIN pool is proportionally capped at 2,000 rows by the example
command, before source/target role selection; target VAL/TEST adaptation streams
are capped at 1,000 and each audit at 5,000. Caps sample without replacement and
preserve role proportions up to rounding. These are pilot results, not full-data
results. Zero caps mean unlimited. Ratios may still differ between partitions
because entire conservative groups are kept together. Source and target audits
are group-disjoint from adaptation; final selected-feature duplicates are checked
again. If conservative grouping leaves any required class/role absent, the run
stops instead of silently weakening independence. Inspect the input/split reports
before deciding a different protocol; no scientific independence is claimed from
an unchecked source index.

Prepared DReLAB CSV/group maps and original SHA256 hashes are retained under
DReLAB_TEST/Dataset_prepared_pilots. Official CSVs are unchanged. Integrity and
sampling configs prevent accidentally reusing a run with different inputs/caps.
Use a fresh -Output to change caps/epoch settings. Partial training is preserved.

Outputs:
- ASNM_NPBO_TEST/protocol_runs/paironly_simple_seed42
- DReLAB_TEST/protocol_runs/paironly_simple_CTUNeris_f0.1_seed42

Both have Models, Prepared, training.json, and Results/summary.csv/summary.json,
stream_metrics.json and trace.csv. Report online accuracy/F1 vs frozen, target
holdout accuracy/F1/attack recall, label use and source accuracy drop. Source
macro-F1 and attack recall before/after are also reported because source classes
are imbalanced. These are single-target-domain retention pilots, not multi-domain
BWT experiments. Full-data runs and additional seeds can follow after the pilot.

No numerical/execution tests were run for these new entry points, per user
request. Only Python syntax was checked. They reuse the previously exercised
pilot agent/stream; actual dataset behavior must be checked on the user machine.

DReLAB processed-feature overlap fix: TRAIN preprocessing stays fixed. Entire
held-out immutable groups whose processed features overlap TRAIN are purged;
TEST also purges overlap with retained VAL. Within VAL/TEST, observer pools take
priority over stream groups. No row is reassigned; removed counts are printed
and saved in split_manifest.json. Missing classes still stop the run. This changes
the retained evaluation population; report the counts with pilot results.
For an earlier failed run, use a new --Output ending in _purged_seed42. Existing
sampled CSV/groups are reused. No numerical execution tests were run for this fix.
