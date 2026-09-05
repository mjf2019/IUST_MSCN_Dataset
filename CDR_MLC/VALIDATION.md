# CDR-MLC validity checks

The submitted notebook contained two target-leakage paths:

1. `label_encoded` remained in `classification_features`, exposing the target to every
   expert classifier during both training and evaluation.
2. The optional `At Least One Correct` rule inspected the true test label to select a
   prediction from the expert outputs.

The revision branch removes both paths. Mini-Batch K-Means now assigns training and
inference samples using the same label-free nearest-centroid rule. The former
label-aware cluster rebalancing is no longer part of the evaluated pipeline.

Run the regression checks from the repository root:

```bash
python -m pip install -r CDR_MLC/requirements-validation.txt
python -m pytest CDR_MLC/tests -q
```

All manuscript metrics produced before this correction must be treated as invalid until
the corrected pipeline has been rerun with the documented train/test scenarios.

## First corrected baseline

The first smoke run uses Scenario 2 (low-congestion training data and
high-congestion test data), seed 42, three clusters, and a causal window of size 3.
It produces 32 classification features after excluding the three routing features.
The corrected single-run accuracy is `0.868758` and weighted F1 is `0.865043`.

The submitted Table 5 values for this scenario must not be reused. This smoke result
is stored in `results/validity/scenario_2_seed_42.json`; final manuscript values will
be based on repeated runs and statistical uncertainty estimates.
