# Leakage-safe CDR-MLC validation (`scale_1`)

This directory records a deterministic validation run of the leakage-safe
implementation with random seed 42.  These files are **not yet a replacement
for Table 5 in the manuscript**, because the repository's `scale_1` datasets
do not have the same sample counts as the datasets used for the published
table.

| Scenario | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| 1 | 0.788991 | 0.788126 | 0.816487 |
| 2 | 0.861688 | 0.857385 | 0.863796 |
| 3 | 0.867433 | 0.867732 | 0.872656 |
| 4 | 0.392518 | 0.429148 | 0.394404 |
| 5 | 0.638453 | 0.544434 | 0.647445 |

Command used:

```bash
python run_table5.py \
  --data-root DATASETS/CDR-MLC/scale_1 \
  --seeds 42 \
  --output-dir results/table5/scale_1
```

The four invariant tests in `tests/test_cdr_mlc_model.py` passed before this
run. They verify exclusion of target-like columns, label-independent
inference, identical train/inference routing rules, and causal window
features.
