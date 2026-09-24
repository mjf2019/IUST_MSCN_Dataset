# Benchmark methods

This directory contains leakage-safe implementations of external methods used
to benchmark Adaptive CDR-MLC. Each method is isolated in its own directory and
uses the same immutable target-test protocol where applicable.

- `AF/`: Adaptive Fingerprinting, adapted to Clean-Valid flow features.
- `DFE/`: Deep Flow Embedding, adapted to Clean-Valid flow features.
- `RF/`: standard 100-tree Random Forest baseline.
- `1D_CNN/`: standard three-block 1D-CNN baseline.

Method-specific assumptions, paper deviations, commands and output schemas are
documented in the corresponding README files.
