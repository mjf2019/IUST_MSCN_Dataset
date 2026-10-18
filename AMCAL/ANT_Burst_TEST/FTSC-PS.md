# Official FTSC-PS data

Run from repository root:

```powershell
python AMCAL/amcal_burst_prepare_ps.py --cache AMCAL/ANT_Burst_TEST/official_source_cache --output AMCAL/ANT_Burst_PS_TEST/Dataset
```

The script downloads the original FTSC dataset and the authors' saved PS UAPs, then exports clean train (3230), validation (1075), test (1076) CSVs and ten test levels (0, 1, 3, 5, 7, 10, 12, 15, 17, 20) under AdvBurst_FTSC-PS. Each CSV has 100 signed packet-size features and label. Output must be empty; IAT files are never modified. Requires numpy and pandas, no GPU or TensorFlow. Original source split, normalization and insertion conditions are retained; training data are not oversampled. The supplied manifest records the tested export hashes. This reconstructs saved attacks, not the gradient optimization. PS uses a separate model from IAT. PS and IAT have different row counts due to representation-dependent short-flow filtering; do not pair them by row index.
