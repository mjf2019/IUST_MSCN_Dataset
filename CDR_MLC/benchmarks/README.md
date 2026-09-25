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

Console output intentionally uses short headings:

`Scn`, `Sd`, `N`, `Acc`, `BAcc`, `MF1`, `WF1`, `FitS`,
`PredS`, and `us/R`.

CSV and JSON artifacts retain descriptive field names, feature lists,
configuration details and partition-overlap audits.
