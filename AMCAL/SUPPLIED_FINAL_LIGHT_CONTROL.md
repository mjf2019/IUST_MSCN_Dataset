# Final Light CNN control on the four supplied-attack datasets

This adds separate Light CNN runs for SDNCampus, ISCX-TOR, IUST and IoT.
It does not modify any existing driver, checkpoint, reward or reported result.

The reference is `clean0_meta_beta_ddqn_seed42_v1` with the original CNN.
The new output is `clean0_light_meta_beta_ddqn_seed42_v1`.

## Preserved controls

- Same existing clean source recipe, partition IDs, labels, feature schema,
  TRAIN-only fitted scaler and source-preprocessing bytes.
- Fresh Light CNN (`light_ps_cnn_v1`, the approved PS/IAT architecture) and fresh
  Context trained only on clean TRAIN; clean VAL early stopping only.
- The source fitter and settings match the original supplied-data source stage:
  `amcal_batch5_pipeline.fit`, original maximum CNN/Context epochs and patience,
  CNN AdamW LR 0.001 and initial Context AdamW LR 0.0003. These are the source
  pretraining settings; subsequent adaptive Context updates retain LR 0.00025.
- Identical clean-only DDQN warmup and learned-beta code, query features,
  normalization procedure, reward, independent TRAIN gain/meta reserves,
  epoch selection, batch5 feedback replay and source replay weight 0.1.
- Full eligible TEST, 20% stream cap per intensity, fresh Context/optimizer/memory
  per intensity, frozen CNN/Q/beta. No perturbed CSV is loaded by Train-only.
- `Both` scopes by default: full pre-feedback Online evaluation, and separate
  Independent holdout evaluation. The latter adapts on approximately 80% of TEST,
  with 20% of that stream available as feedback; final scores use only the
  approximately 20% held-out groups. Do not mix these two reports.

CNN/Context training is necessary because original-CNN weights cannot be loaded
into Light CNN. No original/other-dataset model weights are imported. DDQN/beta
must also be trained for the changed predictor/state distribution, rather than
reusing a policy trained against the original CNN. All source supervision remains
level 0; TRAIN probes are additional offline supervision, not part of TEST budget.

The supplied dataset mapping provenance is inherited unchanged, including any
historical pool-order assumption already recorded by the reference preparation.
Do not describe that assumption as independently verified by this control.

Source CNN completion and its RNG are saved for deterministic Context restart.
An interrupted Context fit repeats from its original initialization. Q warmup,
meta training and evaluations resume using the unchanged shared driver's rules.
Changing the recipe requires a separate output directory; no results are deleted.

## User-run commands

```powershell
git pull origin experiment/paironly-paid-replay
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_supplied_light_meta_beta.py -v
```

Each command trains and tests one dataset. Separate terminals may run in parallel.

```powershell
.\AMCAL\run_amcal_supplied_light_meta_beta.ps1 -Dataset SDNCampus -Action All
```

```powershell
.\AMCAL\run_amcal_supplied_light_meta_beta.ps1 -Dataset ISCX-TOR -Action All
```

```powershell
.\AMCAL\run_amcal_supplied_light_meta_beta.ps1 -Dataset IUST -Action All
```

```powershell
.\AMCAL\run_amcal_supplied_light_meta_beta.ps1 -Dataset IoT -Action All
```

Use `-Dataset All` for sequential execution, or `-Action Train` and then
`-Action Evaluate` to separate stages. Use the same epoch arguments for both.

```powershell
.\AMCAL\push_supplied_light_meta_beta_results.ps1 -Dataset All
```

The push command stages only JSON/CSV from these new Light experiment roots.
It excludes model binaries and refuses unrelated staged changes.
