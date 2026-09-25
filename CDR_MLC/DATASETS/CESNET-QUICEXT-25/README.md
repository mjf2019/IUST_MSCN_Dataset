# CESNET-QUICEXT-25: three-month preparation

This directory prepares June, July, and August 2024 from the official
CESNET-QUICEXT-25 Zenodo record. It produces one model-ready Parquet file per
month and a shared definition of seven cross-month evaluation scenarios.

Raw archives and generated Parquet files are ignored by Git.

## Expected input

Place the three official archives in `raw/` without renaming them:

```text
raw/2024-06.zip
raw/2024-07.zip
raw/2024-08.zip
```

## Install and run (PowerShell)

From the repository root:

```powershell
python -m pip install -r CDR_MLC/DATASETS/CESNET-QUICEXT-25/requirements-preprocess.txt

python CDR_MLC/DATASETS/CESNET-QUICEXT-25/prepare_quicext25.py `
  --max-rows-per-day 5000 `
  --sampling-seed 42
```

The official archive checksums are verified before processing. To replace an
interrupted or previous output, rerun with `--overwrite`. `--skip-checksum` is
available only when an independent checksum verification has already been
performed.

## Outputs

```text
processed/2024-06.parquet
processed/2024-07.parquet
processed/2024-08.parquet
processed/feature_schema.json
processed/scenarios.json
processed/dataset_manifest.json
processed/label_counts_audit.csv
```

The ZIP archives are read one daily Parquet member at a time. The default
50,000-row batch bounds memory use, and only one extracted daily file exists in
the temporary directory at a time. The recommended command applies a
deterministic, label-independent uniform sample of at most 5,000 source rows
per day. It preserves coverage of every available day while preventing
high-volume days from dominating the benchmark. Set `--max-rows-per-day 0`
only when a full-flow extraction is explicitly required.

`QUIC_SNI` is used only to derive the eTLD+1 class label. SNI, user agent,
network identifiers, connection IDs, timestamps, and protocol identifiers are
not copied into the feature matrix. The generated schema preserves:

- scalar bidirectional flow features;
- 30 packet positions for inter-packet time, direction, and size;
- PPI aggregate statistics;
- packet-size and inter-packet-time histograms.

All methods must use the same scenario manifest and model feature list. Class
selection is performed independently inside every scenario using development
training rows only. Target-label counts are audit output and must not be used
for class or hyperparameter selection.

## Seven scenarios

| ID | Development month | Test month | Direction |
|---|---|---|---|
| S1 | 2024-06 | 2024-07 | forward |
| S2 | 2024-06 | 2024-08 | forward |
| S3 | 2024-07 | 2024-06 | backward |
| S4 | 2024-07 | 2024-08 | forward |
| S5 | 2024-08 | 2024-06 | backward |
| S6 | 2024-08 | 2024-07 | backward |
| S7 | all three months | global chronological 80/20 | forward |

For S1--S6, the last 20% of the development month is reserved chronologically
for source validation. For S7, the first 80% of the globally ordered data is
development and the final 20% is test; the last 20% of development is reserved
for validation when a method requires it.

The candidate QUIC context features for CDR-MLC routing are `ppi_duration`,
`ppi_ipt_mean`, and `ppi_roundtrips`. Their rolling trend features are fitted
causally by the method runner; they are not computed across partition or month
boundaries during this preparation step.

## Fixed-class RF/CDR confirmation

`quicext25_classes.json` fixes one lexical 20-class ontology for every method
and every scenario. Each selected class has at least 1,000 prepared samples in
each of the three months. No scenario-specific class selection is permitted.

Start with S1 as a confirmation run from the repository root:

```powershell
python CDR_MLC/quicext25_rf_mf_cdr_mlc.py `
  --scenarios 1 `
  --window 3 `
  --congestion-window 50 `
  --expert-trees 20 `
  --utility-trees 10 `
  --meta-trees 20 `
  --rf-trees 110 `
  --seed 42 `
  --output CDR_MLC/outputs/quicext25_rf_mf_confirm
```

After the S1 split/model audit and results are inspected, run all protocols by
changing `--scenarios 1` to `--scenarios 1 2 3 4 5 6 7`.
