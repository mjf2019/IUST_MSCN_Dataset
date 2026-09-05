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

## Existing smoke results

Results under `results/validity` and `results/table5/scale_1` are development checks
on the small `scale_1` data. They are not replacements for the manuscript's Table 5.
Final values must be generated from `scale_0.001`, with its original sample counts,
and should be reported over repeated deterministic seeds.

## Leakage-safe reference implementation

`cdr_mlc/model.py` is the reference implementation for all revised experiments. It
enforces the intended hard-gated mixture-of-experts contract:

1. The Mini-Batch K-Means router is fitted only on causal window statistics of
   `SynAck`, `AckDat`, and `TcpRtt`.
2. Training samples are assigned with the fitted router's `predict` method; traffic
   class labels never alter cluster assignments.
3. Each cluster fits one multiclass Random Forest expert using 32 non-congestion
   traffic features and 20 trees by default.
4. Inference selects exactly one expert using the router. Target columns and
   target-derived metadata are ignored even if supplied by a caller.

Run a Table 5 scenario with:

```bash
cd CDR_MLC
python run_table5.py --scenarios scenario_2 --seeds 42
```

Use the exact dataset scale and sample counts reported in the manuscript for final
comparisons. The runner records configuration, feature names, routing counts, runtime,
and metrics in machine-readable JSON files.
