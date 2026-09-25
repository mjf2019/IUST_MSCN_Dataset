# DFE benchmark for IUST_MSCN

This folder provides a leakage-safe adaptation of **DFE: Deep Flow Embedding
for Robust Network Traffic Classification** (IEEE TNSE, 2025).

The supplied notebook was not suitable as a benchmark. In particular, it
created random unregistered projection layers inside the compression loss,
normalized each test set using its own extrema, omitted the stride-2 layers,
averaged 20 embeddings into one centroid per class, and forced `k=1`. Those
choices materially differ from the paper.

The corrected runner retains:

- the 13-convolution residual backbone and three stride-2 operations;
- the 9x9 flow-feature image representation;
- equations (6)-(8) through a differentiable Gaussian compression estimator;
- the constrained triplet objective in equations (12)-(13);
- SGD, learning rate 0.001, batch size 32, at most 200 epochs and patience 10;
- 20 separate template embeddings per class;
- L2 nearest-template voting rather than centroid classification;
- template-only adaptation without backbone retraining.

The paper's exact 81 features are unavailable in IUST_MSCN, so this method
must be reported as **DFE-adapted**. Available Clean-Valid numeric fields are
ordered deterministically and zero-padded to 81.

## Run

```powershell
python CDR_MLC/benchmarks/DFE/dfe_iust_mscn.py --fractions 0 0.01 0.05 0.10 0.20 --test-fraction 0.20 --scenarios 1 2 3 --epochs 200 --seed 42 --device cuda --output CDR_MLC/benchmarks/DFE/outputs/iust_mscn
```

The Low backbone is reused by scenarios 1 and 2. For nonzero budgets, at most
20 chronological calibration embeddings per class replace the source template
library; the backbone is not retrained. At zero budget, source templates are
used. The target test tail is immutable for every budget.
