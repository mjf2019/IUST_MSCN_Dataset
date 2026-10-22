# PairOnly paid-feedback replay controls

Evaluation only; existing PairOnly checkpoints, one Context/three actions,
Context Adam LR 0.0005, gamma .95, epsilon .05, seed 42, 20% cap.
Default behavior is unchanged unless --paid-replay-mode is specified.

- sequential: current acquired row, then up to two distinct uniformly drawn old
  paid rows, with a separate Adam step for each (at most three).
- batch: current acquired row and the same kind of up-to-two old rows, with one
  Adam step. The primary loss averages CE over these rows before applying the
  existing Context objective against mean frozen-CNN CE.

Only recent/historical paid feedback is sampled, never source anchors or audit
labels. Current and duplicate inputs are excluded. Sampling happens before the
current row is inserted. Empty/short memories use available rows without padding.
A separate persistent RNG avoids consuming reference/protected-replay RNG state.
Existing protected replay is retained: the same reference replay batch is used
on every sequential step, or once in batch mode. Thus these experiments include
both extra paid replay and the existing protected loss. Sequential also increases
optimizer work; this is not a compute-matched comparison with batch or baseline.

Transfer gain/damage and the paired successor are computed after the whole update.
One paid label is charged, and routing/selector feedback occurs once per query.
The CNN remains frozen. Per-query Context version advances once after the whole
operation. Independent domains reset memory; sequential domains retain it.
Execution fingerprints include the mode, preventing baseline result collisions.

Run from the repository root, in order on the same idle GPU:

```powershell
python -m unittest discover -s AMCAL/tests -p test_paid_replay.py -v
.\AMCAL\run_amcal_paironly_paid_replay.ps1 -Dataset IAT -Mode sequential
.\AMCAL\run_amcal_paironly_paid_replay.ps1 -Dataset IAT -Mode batch
.\AMCAL\run_amcal_paironly_paid_replay.ps1 -Dataset PS -Mode sequential
.\AMCAL\run_amcal_paironly_paid_replay.ps1 -Dataset PS -Mode batch
```

Each command runs independent and persistent sequential evaluation (eight total).
No training is started. PS requires the completed PS PairOnly pipeline; use
-OutputRun if the checkpoint is elsewhere. Results remain under each existing
PairOnly run's Results/evaluation_* directory with separate tags/fingerprints.

Validation in authoring environment: Python syntax and update/reward/memory
insertion order checks passed. PyTorch is unavailable here; both added numerical
CPU tests were skipped, and full dataset/GPU runs were not performed. PowerShell
was reviewed statically. Run the CPU tests with PyTorch before dataset evaluation.
