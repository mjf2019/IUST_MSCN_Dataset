# Standard Random Forest benchmark

This directory contains an independent, leakage-safe runner for the standard
Random Forest supplied in `Standard_RF.ipynb`. The adapted benchmark preserves
100 unrestricted-depth trees, balanced class weights, median imputation and
standard scaling. The scaling step is retained for notebook compatibility even
though it does not change ordinary tree split ordering.

The shared Clean-Valid schema excludes `IdleTime`, labels, congestion level,
capture identifiers and row metadata. For every scenario, the complete source
level and the requested chronological target prefix form the development data.
The final chronological target tail is fixed and is never used to fit either
the preprocessor or the classifier.

## Run independently

```powershell
python CDR_MLC/benchmarks/RF/standard_rf_iust_mscn.py --fractions 0 0.01 0.05 0.10 0.20 --test-fraction 0.20 --scenarios 1 2 3 --trees 100 --seed 42 --output CDR_MLC/benchmarks/RF/outputs/iust_mscn
```

Main output: `standard_rf_summary.csv`.
