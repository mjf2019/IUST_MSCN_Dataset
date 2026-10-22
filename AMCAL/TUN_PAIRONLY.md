# Simple PairOnly on ASNM-TUN

```powershell
.\AMCAL\run_amcal_tun_paironly.ps1 -Action All -Seed 42
```

Automatically locates ASNM_TUN_TEST/Dataset_official_v1/**/ASNM-TUN.csv.
Output: AMCAL/ASNM_TUN_TEST/protocol_runs/paironly_simple_seed42.
All trains then evaluates; Train and Evaluate are also supported. Original CSV
and prior IAT/PS models are unchanged. No downloads or attack generation occur.
Use -Csv / -Output for explicit paths, -Device cpu for CPU, -MaxFeatures 0 for
all nonconstant source-TRAIN features (slower). Keep the same arguments on resume.
Verified training is skipped; changed inputs/schedules or partial training require
a fresh output. Evaluation recomputes reports within this run.

## Protocol

This is a small-data TUN adaptation/retention pilot, not a reproduction of the
published ASNM random-CV tables or the Burst 7-domain experiment. Known full TUN
counts are 177 benign, 130 direct attacks, 87 obfuscated attacks. Prediction is
binary benign/malicious; obfuscated traffic is malicious, not a third class.

- All labels, IDs, raw endpoint IP/MAC/ports, host-identity proxies and absolute
  timestamps are excluded using the shared audit policy. Numeric traffic behavior
  features remain. No metadata is silently used as an input.
- Exact-feature duplicate groups are kept together across a fixed 60/20/20
  TRAIN/VAL/TEST split. This does not establish independent sessions or times.
- Source is benign + direct; target is benign + obfuscated. Within VAL and TEST,
  entire benign groups are reserved for the source holdout. Remaining benign and
  obfuscated target groups are split into adaptation and fixed target audit.
  All audit features/IDs are excluded from adaptation. Both target classes must
  occur in adaptation and audit; invalid/small grouped splits are rejected.
- Fixed signed-log transformation handles ASNM's widely differing numeric scales.
  Missing-value medians, ANOVA selection (default maximum 64 features) and scaling
  fit ONLY source TRAIN. Constant source-TRAIN features are excluded. Feature
  names, excluded columns, row IDs/counts and transforms are recorded in Prepared.
  Cross-partition/audit duplicates are rechecked on the final model input.
- Original CNN architecture is trained from scratch on source TRAIN; VAL-only
  early stopping, at most 60 epochs. Original Context is pretrained against the
  frozen CNN on source TRAIN (maximum 10 epochs). No TEST data is loaded by TRAIN.
- New dataset-specific one-Context selector initialization uses dense TRAIN
  supervision for correctness/critic (source then target TRAIN, maximum 6 epochs).
  This replaces unavailable compatible pretrained weights and is recorded in
  training.json; it is not an imported IAT/PS selector or their three-bank warmup.
- Final one-Context, three-action PairOnly specialization trains update_value only
  with the existing paired advantage objective and paid TRAIN feedback; correctness
  and future heads remain fixed. Maximum 12 epochs; epsilon .05, gamma .95,
  Context LR .0005, alpha .8, feedback cap 20%, patience 5. An inherited epoch-zero
  candidate is considered. All model/selector selection uses temporary VAL
  adaptation and fixed VAL observers; target TEST never selects a model.
- Simple original Context update: one current acquired row with existing protected
  memory replay; no batch5, damage guard or protected routing replay. Final online
  prediction routing/query learning uses ordinary paid-feedback losses.

## Results

Results/summary.csv and summary.json include frozen vs PairOnly online accuracy
and macro-F1, label usage/cap, independent target-audit accuracy before/after,
target-audit macro-F1/attack recall, source accuracy before/after and source drop
in percentage points. Positive source_drop_pp means loss of original-domain
performance. CNN invariance is checked. Trace and stream metrics are saved too.

There is one target domain, so multi-domain forgetting/BWT are not reported.
The relevant retention measure is fixed source-holdout accuracy drop. With only
394 total rows, audits are small: report their sizes and treat a single seed as
a pilot. These are not the full-dataset benchmark outcomes until run locally.

Prepared/preprocessing.joblib and per-pool NPZs retain preprocessing and split
identities. Prepared files and trained models have integrity hashes. No synthetic
samples, oversampling or new attacks are generated.

## Validation

```powershell
python -m unittest discover -s AMCAL/tests -p test_tun_paironly.py -v
```

Three CPU tests use aligned-schema synthetic TUN CSVs, execute full training,
reload and evaluation, and verify no TEST access during TRAIN, frozen heads
in PairOnly specialization, preprocessing/metadata exclusions, disjoint audits,
label caps, frozen CNN, retention arithmetic, raw CSV preservation and resume.
A known-count fixture verifies the default seed pools. Another check perturbs held-out features and confirms source-TRAIN preprocessing
and partitions remain unchanged. Numerical fixtures do not establish real TUN
performance; PowerShell is reviewed statically.
