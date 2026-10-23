# RAL-CD (journal reconstruction)

Reference: Wassermann et al., *Adaptive and Reinforcement Learning Approaches for Online Network Monitoring and Analysis*, IEEE TNSM 18(2), 2021, DOI 10.1109/TNSM.2020.3037486, section V-E.

This is a separate RAL-CD implementation; historical RAL, RAL-recent5 and AMCAL code/results are unchanged. No dependency is automatically installed.

## Algorithm

The existing RAL query committee, decision-power update, adaptive uncertainty threshold and epsilon exploration are reused. The committee is kNN(5), RF(10) and a decision tree; uncertainty=.9, epsilon=.025, eta=.01, reward=+1 and penalty=-1 (MAWI preset).

Prediction precedes query, reward computation and model update. Only acquired committee-query rewards enter ADWIN. Epsilon-only acquired labels train the model but do not generate a reward event. The detector monitors raw +/-1 rewards as described in the paper. It is not a detector on all stream labels or on classification accuracy (the latter belongs to ADAM).

On change, initial training data and old feedback are removed. The retained ADWIN window counts reward events, so its oldest event is mapped to the corresponding paid-label index. Acquired exploration labels inside that interval are retained too. Query weights/threshold are preserved. Each acquired label causes a single refit of the committee models and soft-voting predictor. Window size, drift events, retained rows and refits are saved.

**Important:** RAL-CD does not bound the memory to 5 or any other fixed count. Before the first detected drift it still refits initial data plus feedback. Between drifts memory can grow. It may detect no drift; it is not guaranteed cheaper or more accurate than RAL.

## Fidelity and explicit reconstruction choices

The paper does not release RAL-CD implementation details sufficient for a bit-exact reproduction. This implementation makes the reward-to-label-window mapping and retention of exploration labels explicit. ADWIN delta=.002 and clock=32 are library defaults, not claimed paper hyperparameters. Initial detector history is empty; initial model labels are not scored in-sample to initialize it.

The paper used scikit-multiflow ADWIN, selectable with `-ADWINBackend skmultiflow` if already installed. The default uses installed River ADWIN2. Its retained window is preserved using the underlying helper update interface: River's public wrapper otherwise resets the remaining window on the next event following a drift. Unsupported APIs raise an error. Backend/version are recorded. This compatibility detail is explicit; River is not claimed numerically identical to scikit-multiflow.

The runner applies the existing Burst benchmark: matched training levels 0,1,3; seven unseen test intensities; train-fitted preprocessing and disjoint source IDs; independent initial model/detector/counters per test intensity; 20% strict cap plus native RAL prefix budget gate. Global seeded permutation matches the existing Burst baseline runner. It is not claimed to preserve within-class order. This differs from the paper's small chronological bootstrap and postadaptation validation protocol. Test truth is used for scoring and only paid truth enters adaptation. Reference level-0 validation and same-level endpoint scoring are observers only.

The source-run supplies preprocessing, not its CNN or selector; no AMCAL training occurs. Default source is the historical seed42 matched protocol preprocessing. For comparisons using a different preprocessing split, supply that exact `-SourceRun` and a fresh `-Output`. Model seed is separate from the preprocessing split seed.

## Run

From repository root (existing environment with numpy, sklearn, joblib, torch and River or scikit-multiflow):

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_ral_cd.ps1 -Dataset IAT -Seed 42
.\AMCAL\run_amcal_ral_cd.ps1 -Dataset PS -Seed 42
```

For clean-only source training and test levels including 1 and 3:

```powershell
.\AMCAL\run_amcal_ral_cd.ps1 -Dataset IAT -TrainLevels 0 -Levels 1,3,5,7,10,12,15,17,20 -Seed 42
```

Existing output directories are protected. To run another backend/parameter set, supply a fresh `-Output`. After a completed Train stage use `-Action Test` with the same original arguments. Completed test levels cannot be overwritten; select only pending levels when continuing. Prior summary files retain all saved levels.

Outputs: `Models/ral_cd.joblib`, preprocessing artifact, training manifest, per-level `Results/level_*/metrics.json` and `trace.csv`, aggregate `summary.json`. Metrics include online pre-update accuracy, macro precision/recall/F1, confusion matrices, paid labels, endpoint gain, level-0 reference forgetting, drift logs and CPU timings including all paid refits. Gain is endpoint accuracy change, not online accuracy change. Endpoint all-row scores include queried rows; unqueried endpoint scores are also reported separately.

For timing comparisons run sequentially with the same machine/load/thread settings. Source fitting, endpoint observers and file I/O are excluded from online time.

## Push result artifacts (no model binaries)

```powershell
Get-ChildItem AMCAL/ANT_Burst_IAT_TEST/protocol_runs,AMCAL/ANT_Burst_PS_TEST/protocol_runs -Directory -Filter 'ral_cd_*' | ForEach-Object {
    $manifest = Join-Path $_.FullName 'training_manifest.json'
    if (Test-Path $manifest) { git add -f -- $manifest }
    $results = Join-Path $_.FullName 'Results'
    if (Test-Path $results) {
        Get-ChildItem $results -Recurse -File | Where-Object { $_.Extension -in '.json','.csv' } | ForEach-Object { git add -f -- $_.FullName }
    }
}
git diff --cached --stat
git commit -m "Add RAL-CD Burst results"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; stop here' }
git pull --rebase --autostash origin experiment/paironly-paid-replay
if ($LASTEXITCODE -ne 0) { throw 'Pull failed; stop here' }
git push origin HEAD
```

Only syntax and CLI help were checked during preparation; numerical experiments are left to the user. No claim of reproduced paper results is made.
