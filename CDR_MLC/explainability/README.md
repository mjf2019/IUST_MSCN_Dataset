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
