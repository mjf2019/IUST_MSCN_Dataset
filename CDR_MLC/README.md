# CDR-MLC: paper-aligned, label-free implementation

This implements **Algorithms 1 and 2 and Figure 6** of the supplied
`CDR-MLC(2).pdf`. It fixes the previous notebook's label leakage and routing
mismatch. It does **not** claim to reproduce the paper's published scores.
The current CSVs and temporal metadata are insufficient for that claim.

## Quick start

From the repository root (Python 3.10+):

```bash
python -m pip install -r CDR_MLC/requirements.txt
python -m unittest discover -s CDR_MLC/tests -v
python CDR_MLC/run_cdr_mlc.py --scenario 1
python CDR_MLC/run_cdr_mlc.py --scenario 2
python CDR_MLC/run_cdr_mlc.py --scenario 3
```

Each run writes `results/cdr_mlc/scenario_N/` containing:

- `report.json`: metrics, cluster/class tables, missing classes, silhouette,
  distance diagnostics, dataset SHA-256 hashes, configuration, versions, and limitations.
- `predictions.csv`: the single routed prediction and cluster for each test row.
- `train_cluster_class_counts.csv`: class coverage of each expert.
- `model.joblib`: scaler, router, experts and feature schema together.

Please return `report.json` and `train_cluster_class_counts.csv` for analysis.
Do not compare the removed notebook outputs with these as if the protocols were identical.

For the complete scenario matrix:

```bash
python CDR_MLC/run_cdr_mlc.py --scenario all
```

| Scenario | Training | Test |
|---|---|---|
| 1 | Short level 1 | Short level 2 |
| 2 | Short level 1 | Short level 3 |
| 3 | Short level 2 | Short level 3 |
| 4 | Short levels 1, 2, 3, independent file boundaries | Long, notShuffle |
| 5 | Long, notShuffle | Short levels 1, 2, 3, independent file boundaries |

Scenarios 4/5 are implemented but were not run in the development environment;
the 62 MB Long CSV was not downloaded. The default uses `CDR-MLC-notShuffle.csv`,
not label-conditioned random interleaving. **The name `notShuffle` does not prove
chronology**; confirm capture timestamps/order upstream.

For independent new captures:

```bash
python CDR_MLC/run_cdr_mlc.py --train train.csv --test test.csv --stream-column capture_id --output results/new_captures
```

`capture_id` must be available before classification, not inferred from application
or congestion labels. CSV rows must already be in observation order. Windows
reset at every file boundary and every contiguous capture-ID change. Reappearing
IDs start a fresh window. Without a stream column, one file is treated as one
sequence. Do not shuffle or group by the target before extracting trends.
The runner rejects files with `Shuffle` in their name by default. The explicit
`--allow-shuffled` override is only for non-temporal diagnostics.

## Is clustering learning physical congestion?

Run a **fit-only diagnostic**, separate from any held-out evaluation:

```bash
python CDR_MLC/diagnose_clustering.py --files CDR_MLC/DATASETS/CDR-MLC/scale_1/Short/level_1.csv CDR_MLC/DATASETS/CDR-MLC/scale_1/Short/level_2.csv CDR_MLC/DATASETS/CDR-MLC/scale_1/Short/level_3.csv --sample-size 1000
```

The known level comes from these three filenames and is used **only after fitting**
for contingency tables and ARI/NMI. For other files, supply `--level-column` with
an annotation column. No level/class label enters MBK or RF features. This audit
fits on all supplied rows and is NOT a held-out classification experiment.
Do not reuse its fitted router in scenarios 1–3, as that would expose their test
levels during training.

On the current repository files, seed 42 and 1,000 diagnostic samples:

| Cluster ID | Level 1 rows | Level 2 rows | Level 3 rows |
|---|---:|---:|---:|
| 0 | 678 | 1,304 | 946 |
| 1 | 1,248 | 1,519 | 1,317 |
| 2 | 4 | 120 | 0 |

Silhouette is **0.4784**, but level ARI is **-0.0016** (approximately chance
agreement). Ward agglomerative silhouette on the same random sample is 0.4598.
Thus geometric separation is not evidence of three physical congestion states.
IDs are arbitrary: cluster 0 is not necessarily low congestion. No relabeling can
turn this contingency table into three well-separated levels.

These findings are conditional on the supplied CSV order, window and scaling
choices. They do not prove that the method cannot work with correctly captured
data. Single-level training can yield three geometric clusters, but cannot by
itself establish coverage of low/medium/high physical regimes. Likewise,
oversampling low-level observations does not create missing high-level regimes.

## Algorithm correspondence

| Paper operation | Implementation |
|---|---|
| Fig. 6 / Algorithm 1 line 1 | Trailing window of 3, current sample included; max/min/median/mean/std of AckDat, TcpRtt, SynAck = 15 features |
| Algorithm 1 lines 2–3 | Fit MiniBatchKMeans with k=3 on training trend vectors; route all training rows using its `predict` |
| Algorithm 1 lines 4–7 | Drop the 3 sensitive fields, train one RF on each cluster's 32 base fields |
| Section 5.1 | 20 trees per RF, not the previous notebook's 80 |
| Algorithm 2 | Reuse fitted preprocessing/router, select exactly one expert, predict without any target label |

The original Algorithm 1 line 3 writes the router as applied to `D^35`, although
it is fitted on `D^15_trendtrack`; this implementation uses the **15 trend
features**, consistent with the prose and Algorithm 2.

The default 35-feature allowlist is derived from the committed Short CSVs after
removing `IdleTime`. RF receives exactly 32 fields. The manuscript's Table 3
mentions names not identical to these CSVs (`sMeanPktSz` / `TcpOpt_MwsST`), and
does not enumerate the complete 35-column schema. This repository-compatible
mapping is explicit, not a claim that the exact original feature selection has
been recovered. Additional labels, encoded labels, source levels, row indices,
capture identifiers, and arbitrary extra columns are ignored. Missing required
features, nonnumeric values, NaN/Inf and degenerate clustering fail visibly.

### Choices the paper does not fully specify

- StandardScaler on trend features, **fitted on training only** (retained from the
  old code); `--no-scale` is an explicit ablation. RF sees unscaled base features.
- Warm-up uses 1 then 2 available observations, before the full 3-row window.
  This permits causal predictions immediately. The paper does not define warm-up.
- Standard deviation uses `ddof=1`, singleton std=0; `--std-ddof 0` is available.
- MBK uses seed 42, k-means++ default initialization, `n_init=10`,
  `batch_size=1024`, `max_iter=100`, `reassignment_ratio=0`.
- RF uses 20 trees, bootstrap=True and the installed sklearn defaults for other
  RF parameters. No label-dependent cluster balancing or class weighting is added.
- Silhouette is sampled (default 2,000 rows) for memory bounds; the seed and sample
  size are recorded. Agglomerative comparison uses Ward on that same sample.
  This differs from claiming full-data reproduction of the paper's 0.59/0.56.

The paper is not a complete executable specification; changing these choices
should be recorded as an ablation, not quietly tuned against the test labels.

## Confirmed problems in the old notebook

1. `label_encoded` remained among classification features: direct target leakage.
   With a custom target name, the original target could remain too.
2. `use_at_least_one_rule=True` chose a prediction using the true test label:
   an oracle, not Algorithm 2. Deployment also consulted the sample's label.
3. `balanced_cluster_assignment` and `ensure_label_coverage` moved training rows
   between clusters according to labels, while test routing used nearest centers.
4. The hand-written two-heap median used physical heap lengths despite lazy
   deletion, so its median could disagree with the actual moving window.
5. Some defaults used window=100 / 4 statistics and 80 RF trees, unlike the paper.
6. Some experiment cells used CSVs interleaved randomly by application/level.
   Preserving order *inside each label group* does not preserve a network timeline.
7. Index/source-level metadata could leak into RF; experts with very few rows
   were skipped and predictions silently fell back to a majority class.

The notebook now imports the tested module. Old output cells and oracle code
were removed. Compatibility wrappers reject unsafe legacy options explicitly.
Missing class coverage is reported; samples are never transferred to manufacture
coverage. An empty cluster is an error rather than a fabricated expert.

## Validation and limits

Tested with Python 3, numpy 2.3.5, pandas 2.2.3, sklearn 1.8.0.
`validation_summary.json` contains the exact environment, data hashes and counts.
12 automated regression tests cover causal windows, boundaries, medians, 15/32
feature dimensions, label invariance, unchanged fitted state after testing,
nearest-center training/inference consistency, missing classes, serialization,
stream/batch equivalence, custom targets, reports and invalid inputs.

Actual runs on committed `scale_1/Short` CSVs, **not the paper-sized dataset**:

| Scenario | Train rows | Test rows | Accuracy |
|---|---:|---:|---:|
| 1 | 1,930 | 2,943 | 0.7958 |
| 2 | 1,930 | 2,263 | 0.8542 |
| 3 | 2,943 | 2,263 | 0.8665 |

The manuscript instead lists 68,079 / 201,165 / 60,482 rows (329,726 total).
Each current Short file has exactly five application-label runs (class blocks),
so capture chronology is not established. The CSVs do not retain reliable
capture IDs/timestamps sufficient to reconstruct the original shared timeline.
We neither sort by labels nor invent timestamps to disguise this limitation.

There are 16, 8 and 17 test rows respectively with feature vectors identical to
training vectors. This is reported as an audit flag, not automatically interpreted
as duplicate capture records. Without provenance it is unsafe to silently remove
or declare them independent. Consequently these are engineering validation
scores, **not corrected publication results**.

For publication, supply ordered independent captures (and their identifiers),
verify feature extraction and sizes against Table 4, and rerun every comparison
under the same split protocol. Future test observations must not enter scaler,
router or expert training, even without their labels.

## Programmatic API and other datasets

```python
from cdr_mlc import CDRMLC, Config, evaluate
model = CDRMLC(Config(), stream_column='capture_id').fit(train_df, 'label')
metrics, predictions, clusters = evaluate(model, test_df, 'label')
live = model.stream()
result = live.predict_one(unlabeled_sample)  # includes capture_id
```

`predict` starts fresh history for each batch; `stream()` preserves history across
individual calls and resets at a capture change. If a stream column was configured,
it is required in every inference sample. Models saved by the runner use
`_capture_id`; provide a stable ID for the live capture. Reusing the same fitted
model never refits on test data.

For another dataset/schema, use an explicit numeric `feature_columns` allowlist
with all three sensitive fields. It must exclude labels and metadata. Never
silently rename unrelated ISCX features to AckDat/TcpRtt/SynAck. The attached
paper does not specify that dataset's exact mapping; Table 6 results and baseline
RF/CNN/AF/DFE experiments have not been reproduced by this change.

### راهنمای کوتاه فارسی

هستهٔ روش اصلاح شده است، اما داده‌های فعلی بازتولید دقیق نتایج مقاله را ممکن
نمی‌کنند. نشت مستقیم برچسب و انتخاب متخصص با پاسخ واقعی حذف شده‌اند. خوشه‌ها
بدون تغییر مصنوعی عضویت ساخته می‌شوند. سه خوشه الزاماً سه سطح ازدحام نیستند؛
جدول بالا نشان می‌دهد در داده‌های موجود سطح‌ها جدا نشده‌اند.

ابتدا سه دستور سناریوی ۱ تا ۳ را اجرا کنید و فایل‌های `report.json` و
`train_cluster_class_counts.csv` را برگردانید. برای ارزیابی علمی نهایی، ترتیب
زمانی و شناسهٔ capture و اختلاف تعداد نمونه‌ها با مقاله باید مشخص شود.
