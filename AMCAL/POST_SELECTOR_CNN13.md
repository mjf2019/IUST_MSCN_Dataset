# CNN-only fine-tuning after selector training

This experiment keeps the current trained system and saves a second bundle after
an additional supervised pass over TRAIN levels 1/3. This describes the new stage;
it does not assume these levels were absent from historical CNN training.

- `Models/current_bundle.pth`: current CNN, healthy initial Context and trained DDQN.
- `Models/cnn13_bundle.pth`: fine-tuned CNN with exactly the same Context, query
  weights and query normalizer. Preprocessing is copied unchanged.

The existing source checkpoint files are never modified. Bundles contain all three
state dictionaries, query spec, class names, config and source/data provenance.
They use a separate format interpreted by this comparison script; do not replace
legacy model files with them.

## Source and fine-tuning stage

Defaults use the last discussed checkpoints: IAT beta=26, PS beta=55, from
`two_action_paper_gain_extra_seed42_p300_e3_v1/Candidates/beta_XX`. `-Beta` selects
another existing paper-plus-gain coefficient. Beta 0/10/100 default to the earlier
parent experiment folder. `-SelectorRun`, `-Teacher` and `-ActorRun` accept explicit
matching model folders. Hashes, seed, dataset, training inputs and architecture are
validated before copying anything. Original IAT CNN / Light PS CNN are retained.

Only a deep copy of the CNN is optimized, on raw preprocessed TRAIN 1/3 examples.
Context is bypassed during this stage, and neither Context nor DDQN is trained.
Defaults: 300 TRAIN rows from each level, 300 VAL rows from each level, at most
five epochs, AdamW LR 0.0001, weight decay 0.0001, ordinary cross entropy, batch32,
gradient clipping1 and patience2. There are no TRAIN level-0 rows in this new stage,
no extra CNN replay, no synthetic drift and no changed reward or query policy.

Use `-TrainRows 0` for all eligible TRAIN 1/3 rows. Source TRAIN retention-reference
IDs/features remain excluded, as in the source protocol. Official TRAIN/VAL IDs
must be disjoint. Existing official partitions are reused; feature-duplicate counts
are recorded rather than silently changing the benchmark's partitions.

The best **fine-tuned epoch** is selected by raw CNN VAL 1/3 CE only. Baseline raw
CNN VAL metrics are saved separately; a worse fine-tuned model is still retained
as an experimental comparator rather than presented as an improvement. TEST is
opened only after training. Epoch-boundary snapshots include CNN, Adam and RNG
state, so repeating the command resumes a partial fine-tune. The resulting CNN
is returned to eval mode and frozen before comparison.

## Paired online evaluation

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_post_selector_cnn13.ps1 -Dataset IAT -Beta 26
.\AMCAL\run_amcal_post_selector_cnn13.ps1 -Dataset PS -Beta 55
```

Both Current and CNN13 are evaluated on identical official TEST rows/order for
levels 5/7/10/12/15/17/20, default 300 rows per level, 20% label cap, fixed
Context -> CNN predictions, batch5, Context LR 0.00025 and replay weight0.1.
Query weights stay frozen, but their actions can change because CNN probabilities
and embeddings now differ. This is the intended post-selector compatibility test;
an increase in accuracy is a hypothesis, not assumed behavior.

For every variant/intensity, Context, Adam, feedback memory and cap reset from
that variant's initial bundle. Predictions precede feedback. Source anchors,
preprocessing and Context initial weights are identical. Gain probes are not
observed during TEST. Use `-Action Evaluate` to repeat only comparison after both
bundles are complete. `-TestRows 0` uses all official TEST rows in a separate result
subfolder and reuses the completed fine-tune.

Source forgetting is reported against a **common original-system reference**:

- `cnn_finetune_source_drop`: current initial source AC minus the variant's initial
  source AC, before online updates. This captures damage from CNN fine-tuning.
- `forgetting`: variant initial source AC minus its final online source AC, as in
  previous runs. This captures online Context adaptation alone.
- `total_source_drop`: current initial source AC minus the variant's final source
  AC. It includes both stages; it is not reset to hide pre-online CNN damage.

The same total-drop accounting is saved for PR/RC/F1. All source metrics use the
same independent original-domain TEST audit. They do not select a fine-tuning epoch.

Default output roots:
`protocol_runs/post_selector_cnn13_seed42_beta_26_trainp300_v1` for IAT and
`post_selector_cnn13_seed42_beta_55_trainp300_v1` for PS. `finetune_summary.json`
compares raw VAL metrics before/after. `Results/testp300/summary_by_level.csv`
contains 14 cases; `summary_mean.csv` compares the two bundles. Model hashes,
epoch logs, full causal traces and checksummed completion records are retained.
Console lines explicitly label dataset, beta, variant and level, and print online
AC/F1, label consumption, final target AC and both stages of source forgetting.

```powershell
.\AMCAL\push_amcal_post_selector_cnn13_results.ps1 -Dataset Both
```

The helper pushes this experiment's JSON/CSV only; bundle/optimizer binaries and
copied preprocessing stay local. No package is installed. No previous result or
manuscript table is automatically changed. This remains a single-seed pilot unless
expanded explicitly.

User-run tests cover copied-CNN-only optimization, per-epoch resume, identical
Context/query bundle states, prevention of bundle tampering, correct cumulative
source forgetting and TRAIN/VAL level-1/3-only development inputs. No imports,
tests, training or evaluation were run by the assistant while writing this change.
