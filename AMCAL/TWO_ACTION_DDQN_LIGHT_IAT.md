# Light CNN IAT experiment

This is a separate experiment using the exact `light_ps_cnn_v1` architecture from
PS, instantiated for IAT input features/classes. Existing original-CNN IAT and PS
drivers, models and results are unchanged. No weight conversion is attempted.

Prerequisite: the completed local Light IAT PairOnly source, normally
`AMCAL/ANT_Burst_IAT_TEST/protocol_runs/light_iat_paironly_batch5_trainseed42/paironly_batch5`.
The wrapper accepts `-SourceRun` for a different completed Light IAT source.
It does not retrain the CNN if this source already exists. All source/checkpoint
hashes and CNN architecture are checked before use.

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_ddqn_light_iat.ps1
```

The first run prepares architecture-matched dependencies:

1. Collect the same 256-row-per-TRAIN-level diagnostic normalizer cache using
   the Light IAT source. It never reuses the original-CNN normalizer.
2. Train the healthy Light IAT Context using the existing full-domain Natural
   standardized-Actor procedure: at most six epochs, original patience and VAL
   selection. This preparatory training is **not limited to 300 rows**. It is
   needed because the original healthy Context belongs to a different CNN.
3. Train the corresponding compact paper-reward Actor on the 300-row pilot for
   two rounds. Only TRAIN/VAL 0/1/3 are used by these preparation stages.
4. Train the same two-action sequential DDQN: default three epochs, 300 rows per
   TRAIN/VAL/TEST domain, budget cap 20% per episode. Run independent TEST on
   levels 5/7/10/12/15/17/20 with the same five matched controls as the main pilot.

Everything except the CNN architecture and its necessarily retrained dependent
models follows the original experiment. Prediction stays Context -> Frozen CNN;
batch5, Context LR 0.00025, replay weight 0.1, DDQN settings and reward table are
unchanged. Light CNN cannot be substituted into an original-CNN checkpoint at
evaluation time. Full matched preparation is needed before comparing architectures.

Completed preparation stages are verified and reused. The healthy Context trainer
retains its existing interruption behavior: a partial preparation run is preserved
and rejected, rather than silently restarted. DDQN resumes at completed epoch
boundaries. Selected epochs/rounds, hashes and metrics are written to separate
`light_iat_*` preparation folders and `two_action_ddqn_light_iat_*` pilot folders.

To repeat only TEST after completed training:

```powershell
.\AMCAL\run_amcal_two_action_ddqn_light_iat.ps1 -Action Evaluate
```

To push DDQN TRAIN/VAL/TEST JSON/CSV summaries and traces:

```powershell
.\AMCAL\push_amcal_two_action_ddqn_results.ps1 -Dataset IAT
```

The existing push helper also stages other new DDQN IAT JSON/CSV if still uncommitted;
it never stages model weights. The light pilot protocol records its teacher hashes.
The wrapper runs architecture-contract tests and the dedicated Light-IAT DDQN
tests before training. No packages are installed. Code/tests have been written
without running tests, training, imports or evaluation by the assistant.
