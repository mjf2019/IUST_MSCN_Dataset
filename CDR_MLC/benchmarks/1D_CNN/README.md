# Standard 1D-CNN benchmark

This directory contains an independent, leakage-safe adaptation of the
supplied `Standard_1D_CNN.ipynb` implementation.

Preserved architecture and defaults:

- Conv1D blocks with 64, 128 and 64 filters, kernel size 3;
- BatchNorm, ReLU and max-pooling by 2 after every convolution;
- dense layers of 128 and 64 units with 0.30 dropout;
- Adam with learning rate 0.001 and weight decay 0.001;
- ReduceLROnPlateau and early stopping with patience 15;
- 100 epochs and batch size 32 by default.

The original random validation split is replaced by the chronological tail of
each source capture. The target test tail remains fixed across every adaptation
budget. Imputation and scaling are fitted only on source-training rows plus the
allowed target calibration prefix. Labels and capture metadata cannot enter
the feature tensor. `IdleTime` and `DstWin` are excluded as in the supplied
legacy implementations.

## Run independently

```powershell
python CDR_MLC/benchmarks/1D_CNN/standard_1d_cnn_iust_mscn.py --fractions 0 0.01 0.05 0.10 0.20 --test-fraction 0.20 --scenarios 1 2 3 --epochs 100 --batch-size 32 --seed 42 --device cuda --output CDR_MLC/benchmarks/1D_CNN/outputs/iust_mscn
```

Main output: `standard_1d_cnn_summary.csv`.
