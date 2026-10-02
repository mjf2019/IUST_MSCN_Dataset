# Primary analysis: congestion-sensitive features on S7

Use this analysis to investigate which original numerical traffic measurements
discriminate Low, Medium, and High congestion. The saved-model analysis below
answers a different question about application classification and meta inputs.

The current Clean-Valid schema contains **24 numerical fields**, as specified in
`build_clean_valid.NUMERIC`. The diagnostic model uses exactly these fields;
Flgs, State, TcpOpt, identifiers, and application labels are not model inputs.
No One-Hot encoding or artificial padding to 35 columns is performed.

The S7 partition is imported from the existing `ALL-80-20` implementation:
each application/level capture is split chronologically into an 80% development
prefix and a non-overlapping 20% test tail. An independent RF is trained to
predict **congestion level**, not application class. Development sample weights
give equal total weight to each application/level stratum, retaining all valid
development records. Imputation is fitted on development data only.

SHAP rows are sampled solely from held-out tails, equally across the three
levels and five applications. SHAP feature ranks are measured without assuming
that AckDat, SynAck, or TcpRtt must be the highest-ranked features. Importance
quantifies predictive association, not causal effects. Correlated predictors
may share attributions; interpret individual rankings accordingly.

## Pilot: 30 held-out records

```powershell
python -m pip install shap matplotlib
python CDR_MLC/explainability/shap_congestion_s7.py --rows-per-level 10 --trees 110 --seed 42 --output CDR_MLC/explainability/outputs/s7_pilot
```

The first USER execution trains and caches the diagnostic model locally.
No application RF/CDR/MF model, dataset, or existing runner is modified.

## Main: 300 held-out records

```powershell
python CDR_MLC/explainability/shap_congestion_s7.py --rows-per-level 100 --trees 110 --seed 42 --output CDR_MLC/explainability/outputs/s7_main
```

The matching cached diagnostic model is reused. A configuration/data mismatch
requires `--force-refit` or another `--cache` path; it never silently reuses
a model trained on different development records.

Results include complete global ranks (`congestion_shap_importance.csv`),
rankings by observed level, application, and output level, the three timing
feature ranks (`timing_features_shap.csv`), diagnostic full-test/sample metrics,
a confusion matrix, raw SHAP values, and a two-panel PDF/PNG figure.
The manifest records the 24 input names, split identities, sample composition,
training and SHAP times, and library versions. All outputs and cached models
remain in the ignored `outputs/` directory.

---

# SHAP interpretability analysis

This script explains the **saved models**, without training or changing them.
It uses `rf.joblib` and `mf-cdr.joblib` from the inference-resource benchmark.
The dataset must match their recorded Low-level training identity.

## Scope

- RF explanations quantify the contribution of the original classifier inputs,
  including AckDat, TcpRtt, and SynAck.
- MF explanations quantify the selected **meta-classifier's** inputs: expert
  outputs, cluster information, congestion descriptors, and estimated utilities.
  They do not explain the complete composite model or confidence fallback.
- Low records are training-level post-hoc examples, because the saved models
  were fitted using the full Low level. Medium and High are unseen levels.
- SHAP explains model associations, not causal effects of congestion.
- `tree_path_dependent` TreeSHAP uses training-path counts in the stored trees;
  no background data, fitting, or test-derived model selection is introduced.
- All causal history is built on complete captures **before** explanation rows
  are sampled. Explanation sampling is balanced by level and application class.
- Rankings average absolute SHAP values over records and model output classes.
  Group rankings sum signed SHAP values within a group first, then take the
  mean absolute value. Group and individual-feature rankings are distinct.

## Install and pilot

In the existing activated environment:

```powershell
python -m pip install shap matplotlib
python CDR_MLC/explainability/shap_analysis.py --rows-per-level 10 --seed 42 --output CDR_MLC/explainability/outputs/pilot
```

This explains 30 records: two per application class at each congestion level.
Check `timing.csv` and the console before increasing the sample size.

## Main analysis

```powershell
python CDR_MLC/explainability/shap_analysis.py --rows-per-level 100 --seed 42 --output CDR_MLC/explainability/outputs/main
```

This explains 300 records: 20 per application class at each congestion level.
No automatic retraining occurs if artifacts are missing. Supply `--models`
and `--data-dir` if they are stored outside their default locations.

## Outputs

- `rf_importance.csv`, `meta_importance.csv`: complete global rankings.
- `rf_by_level.csv`, `meta_by_level.csv`: separate congestion-level rankings.
- `*_by_output_class.csv`: importance for each predicted application class.
- `meta_grouped_importance.csv`: context/expert/utility group contributions.
- `shap_feature_importance.pdf/png`: two-panel individual-feature rankings.
- `shap_grouped_importance.pdf/png`: RF features and MF input-group rankings.
- `*_shap_values.npz`: original signed SHAP values and sample identities.
- `sample_audit.csv`, `manifest.json`, `timing.csv`: provenance and timings.

The script checks reconstruction of the RF class probabilities from SHAP values.
Generated outputs and saved models are not committed to Git.
