# Adaptive Fingerprinting benchmark

This directory contains a clean implementation of **AF-SingleSource** from
*Adaptive Fingerprinting: Website Fingerprinting over Few Encrypted Traffic*
(CODASPY 2021).

## Paper protocol retained

- `M=25` labeled source traces per class.
- A target pool of 20 traces per class and a disjoint test of `T=70` traces.
- `N={1,5,10,15,20}` target traces per class.
- The same `N` target traces are treated as unlabeled during DANN pre-training
  and labeled when fitting the target k-NN classifier.
- 30 pre-training epochs, learning rate `1e-5`, GRL coefficient `1`, a
  512-dimensional embedding, `k=N`, and 10 repeated folds.
- No target-test row is used by preprocessing, DANN, feature extraction
  training, model selection, or k-NN fitting.

## Input modes

`direction` is the paper-faithful mode. Every feature column is a position in
the packet sequence and must contain `-1`, `+1`, or zero padding.

```powershell
python AF/af_single_source.py --source SOURCE.csv --target TARGET.csv --label-column label --input-mode direction --output AF/outputs/exact
```

`tabular` applies the AF training protocol to IUST_MSCN flow features. It uses
an MLP because the dataset does not contain the Tor packet-direction vectors
required by the DF CNN. Its output is therefore named `AF-MLP-adapted`.

```powershell
python AF/af_single_source.py --source LOW.csv --target MEDIUM.csv --label-column label --input-mode tabular --output AF/outputs/low_to_medium
```

Input files must have identical numeric feature columns and one label column.
The output includes per-fold metrics, mean/std summaries, a run manifest and a
row-level split audit.

AF has no valid zero-label result: the target k-NN requires at least one
labeled target trace per class. Report the zero-budget cell as `N/A`.

## IUST_MSCN benchmark

For direct comparison with S-Meta, use the adapted runner. It retains AF's
DANN/GRL and target k-NN mechanism, but uses an MLP over Clean-Valid flow
features, chronological percentage budgets, and the immutable 20% test tail.

```powershell
python AF/benchmark_iust_mscn.py --fractions 0 0.01 0.05 0.10 0.20 --test-fraction 0.20 --scenarios 1 2 3 --epochs 30 --seed 42 --output AF/outputs/iust_mscn
```

The zero-budget AF rows are emitted as `N/A`; no synthetic zero-shot AF method
is substituted. This runner should be used in the comparison table, under the
name **AF-MLP-adapted**.
