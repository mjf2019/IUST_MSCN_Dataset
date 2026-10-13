# Benchmark methods

This directory contains isolated implementations of the external methods used
to benchmark MF-CDR-MLC. All applicable runners use the same chronological
partition rules, training-only preprocessing and immutable test partitions.

Existing baselines:

- `AF/`: Adaptive Fingerprinting adapted to Clean-Valid flow features.
- `DFE/`: Deep Flow Embedding adapted to Clean-Valid flow features.
- `RF/`: standard Random Forest baseline.
- `1D_CNN/`: standard three-block 1D-CNN baseline.

Modern reviewer-requested baselines:

- `FT_Transformer/`: Transformer-based tabular classifier.
- `SCARF/`: self-supervised tabular representation learning followed by
  supervised fine-tuning.
- `GraphSAGE/`: inductive GNN over causal within-capture temporal graphs.

The modern runners support Scenarios 1--7 and are independently executable.
Model-specific assumptions, commands and dependencies are documented in each
directory. Shared code in `deep_common.py` is limited to data loading,
chronological partitions, training-only preprocessing, metrics and output
formatting so that all three methods receive identical inputs.

For IUST_MSCN, `--train-fraction` (or `--test-fraction` in the adapted
1D-CNN, AF and DFE runners) controls the chronological 80/20 split used by S7
and within-dataset protocols. The independent
`--target-test-fraction` option controls only the unseen target level in
S1--S3. Set it to `1.0` for zero-shot full-target evaluation. AF consumes a
labeled target prefix, so a one-percent AF budget uses
`--target-test-fraction 0.99 --fractions 0.01`, evaluating the disjoint
remaining 99 percent of the target level.

Every runner prints two tables. The first is a two-column legend that expands
the abbreviations used by that run. The second contains one physical row per
result using only short headings. The common performance and deployment fields
are:

- `Acc`, `BAcc`, `MF1`, and `WF1`;
- `FitS` and `InfS` for training and inference seconds;
- `us/R` and `R/s` for microseconds per input record and records per second;
- `RAM` and `GPU` for peak process and allocated GPU memory in MiB;
- `TTus` for causal TTFEF microseconds per input record where applicable.

The models operate on flow records, so `us/R` is the scientifically correct
counterpart of the reviewer's requested time-per-packet measure. CSV and JSON
artifacts retain descriptive field names, feature lists, configuration details
and partition-overlap audits.

## CESNET-QUICEXT-25

The modern benchmark runners auto-detect the prepared QUICEXT monthly Parquet
files when `--data-dir CDR_MLC/DATASETS/CESNET-QUICEXT-25/processed` is used.
They then apply the fixed lexical 20-class ontology and the same S1--S7
definitions as MF-CDR-MLC. All zero-shot methods are scored on the identical
causally window-eligible target record IDs. Preprocessing and validation use
development data only.

Dedicated QUICEXT runners are provided for 1D-CNN and DFE. AF is reported
separately because it requires labeled target samples: its default one-percent
budget reserves the earliest one percent of every target class and evaluates
the remaining disjoint target rows. It must not be described as zero-shot.
