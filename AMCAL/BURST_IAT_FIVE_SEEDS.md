# Burst IAT: five online evaluation seeds

Run from the dataset repository with its Python virtual environment active:

```powershell
git pull --ff-only origin experiment/amcal-protocol
powershell -ExecutionPolicy Bypass -File AMCAL/run_burst_iat_five_seeds.ps1 -Group AMCAL
powershell -ExecutionPolicy Bypass -File AMCAL/run_burst_iat_five_seeds.ps1 -Group Baselines
```

The script resolves the dataset checkout from its own location; `-DatasetRepo` optionally overrides it. Existing trained checkpoints are reused, and only missing/incompatible evaluations are run. Defaults: online seeds 42, 48, 54, 62, 72; held-out levels 5, 7, 10, 12, 15, 17, 20; Context LR 0.0005; epsilon 0.05; 20% strict label cap; the approved original coupled seven-action balanced-reference policy. These are repeated online streams with fixed offline checkpoints trained with seed 42, not five independent trainings.

The approved AMCAL output is `structured_bank3_balancedref_lr0p0005_seed42`. Baseline checkpoint locations are listed in the script. Missing trained models produce an error rather than silently training a substitute. JSON configuration checks prevent reuse of quality-gate, decoupled-query, target-scaler, or different policy/LR/epsilon outputs. Numerical fields already saved in metrics JSON, including Macro-F1, are included in the bundles.

Each group creates a timestamped ZIP under `AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paper_five_seeds_<group>_<timestamp>.zip`, including selected full metrics, console log, and a manifest. Run groups sequentially on an idle machine for runtime comparison; `-Force` disables reuse and reruns all requested seeds. PowerShell/GPU execution has not been performed in the editing environment; script arguments were checked against the existing Python parsers.
