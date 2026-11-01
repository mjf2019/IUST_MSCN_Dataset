# Clean-only learned-beta DDQN on supplied attacks

This opt-in runner leaves all previous scripts, models and results unchanged.
It reuses the verified **original CNN and initial Context trained on clean level
0** for SDNCampus, ISCX-TOR, IUST and IoT. The two-action DDQN requests or skips
feedback; prediction always follows Context -> frozen CNN before feedback.
Context uses the existing competitive update, batch5 paid replay, source replay
weight 0.1 and learning rate 0.00025. Each episode starts with a fresh Context,
Adam optimizer, paid memory and independent 20% stream cap.

## Training

Default source root per dataset:
`AMCAL/<dataset-folder>/protocol_runs/clean0_AMCAL_attacks_seed42`.
The runner requires its completed clean `source` stage and prepared domains.
It validates model/preprocessing hashes, source-only training/VAL levels,
permitted features and source IDs. It does not retrain the CNN or import any
IAT/PS weights, learned beta or synthetic drift. Use `-SourceRoot` if the existing
clean source run has a different name. Missing source artifacts cause an error;
there is no silent fallback to attack-trained models.

The source recipe retains the original supplied-attack mapping provenance:
some legacy CSVs use an explicitly assumed historical pool order. Reusing this
recipe does not independently verify those historical row mappings.

1. Reserve disjoint clean TRAIN reference, gain probe, 300-row meta stream and
   separate meta gain probe. Group feature duplicates so they cannot cross roles.
   Use **every remaining TRAIN row** in the main trajectory; no main-TRAIN or
   TEST sample cap is applied. Probes are held out from current adaptive updates;
   they may have appeared in the historical supervised CNN/Context pretraining.
2. Fit frozen state normalization using uniform temporary adaptation on all inner
   TRAIN rows. Initialize a fresh dataset-local `2K+18 -> 16/GELU -> 2` DDQN;
   warm it up for 3 epochs with the existing original-paper reward.
3. Train for 3 learned-beta epochs with the unchanged `MetaBetaDDQN` objective.
   For each epoch collect uniform paid transitions on the independent 300-row
   clean meta stream (60 labels at 20%), providing at least 32 observations.
   The old three-intensity meta collector cannot be used unchanged when only
   clean level 0 is available. This clean-only collector supplies the same
   component targets without importing attack data.
4. Beta starts from TRAIN reward/gain RMS calibration and learns through the
   existing virtual-step meta objective. Main DDQN rewards are reconstructed from
   original-paper reward plus learned beta times signed independent-probe gain.
   A zero-gain meta sample fails explicitly; no artificial signal is invented.
5. Select the epoch using clean VAL only: online AC minus positive source
   forgetting; fewer labels break ties. VAL has a separate 20% reference reserve.
   VAL/TEST never update Q or beta. Only the Context adapts within each trajectory.

Three warmup plus three meta epochs are defaults, not CNN training epochs.
TRAIN probes and initial supervised training use additional TRAIN labels;
20% denotes the per-trajectory/test feedback cap, not total offline supervision.
The original CNN/Context files and their initial weights remain unchanged.
Normalization, warmup and learned-beta stages have resumable checkpoints.
An interrupted epoch is repeated from its last completed checkpoint.

## TEST protocols (separate reports)

All 12 supplied cases are tested: CW/DeepFool strengths **0.1, 1, 5** and
FGSM/PGD strengths **0.01, 0.1, 0.5**. Ordinal levels 1/2/3 map to these strengths.
They are never loaded by a Train-only invocation. Each case reloads the initial
Context; CNN, Q and learned beta stay fixed. A seeded shuffle matches the
existing baseline stream convention. No state or paid labels carry across cases.

- `Online`: **all eligible original TEST IDs**, matching the previous supplied
  attack evaluations. All metrics are pre-feedback online AC/PR/RC/F1. Re-scoring
  those seen rows is stored only as an explicit diagnostic, not as independent
  final-target accuracy.
- `Independent`: reserve approximately 20% target rows jointly across all cases
  using flow-ID/exact-feature groups. Adapt on the remaining approximately 80%,
  with a cap of 20% **of that stream** (about 16% of all target rows). Report both
  denominators. Final-target AC/PR/RC/F1 use only the independent holdout. Purge
  holdout IDs/features from source replay. Pre-existing models are not refitted.
  No group is broken to force exactly 20% rows.
- `Both` (default): run both scopes; results are in distinct directories and
  **must not be averaged together**. Original-domain forgetting is the before
  minus after metric on clean VAL, consistent with the existing baselines.

The runner prints dataset, phase, attack, strength, AC/PR/RC/F1, actual feedback
use, budget use, original-domain forgetting and independent final AC where valid.
It reports means over the three intensities of each attack and over all 12 cases.

## Commands (user runs; no installation)

```powershell
git pull origin experiment/paironly-paid-replay
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_clean_meta_beta.py -v
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset SDNCampus
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset ISCX-TOR
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset IUST
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset IoT
```

Use separate terminals for separate datasets if memory permits. `-Dataset All`
runs them sequentially. The wrappers use the project's own venv Python, avoiding
the WindowsApps interpreter. Rerunning the same command resumes completed epochs
and skips verified completed training/evaluation cases. No files are deleted.

Training and testing can also be separated:

```powershell
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset SDNCampus -Action Train
.\AMCAL\run_amcal_clean_meta_beta.ps1 -Dataset SDNCampus -Action Evaluate
.\AMCAL\push_clean_meta_beta_results.ps1 -Dataset All
```

Push exports only JSON/CSV from the new experiment roots, excluding Models.
It refuses unrelated pre-staged changes and never force pushes.
