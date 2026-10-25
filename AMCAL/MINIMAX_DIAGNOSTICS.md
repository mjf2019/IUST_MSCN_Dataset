# Diagnose the v2 query target before changing it

The diagnostic entry point does not alter the deployed reward, cost (0.05),
threshold (0.5), architecture, cap, replay weighting, or optimizer definition.
The original `minimax_query.py` implementation is unchanged. Existing v2
models/results are read only; diagnostic outputs use a separate directory.

## First run: a short TRAIN pass, then a small Actor audit

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_minimax_diagnostics.ps1 -Dataset IAT
.\AMCAL\run_amcal_minimax_diagnostics.ps1 -Dataset PS
```

Use separate terminals if desired. The default visits at most 256 shuffled
TRAIN rows at each of levels 0, 1 and 3, with the same independent future and
forgetting probes as v2. A prefix retains its original full-domain budget cap
and position features. It is a diagnostic prefix, not a 20%-of-prefix benchmark
and not a reconstruction of the full v2 training trajectory. No validation or
TEST labels enter the Actor fitting data. It does not start six training epochs.

Outputs live under
`AMCAL/ANT_Burst_{IAT|PS}_TEST/protocol_runs/minimax_diagnostics_seed42_n256`:

- `TRAIN/train_epoch1_level*.csv`: every visited row, including masks and
  non-target rows. Logs `gap` (= d_B), weighted gain beta G, weighted harm lambda F,
  request cost, S, probability before/after the actor step, logit, actual clipped
  Actor gradient norm, request/skip, and original-domain budget state.
- `TRAIN/target_summary.json`: full-trace and per-level sign counts and term
  distributions, including zero values. Illegal/exhausted-budget and unavailable
  future-block rows have no invented S; their exclusion reasons are counted.
- `TRAIN/actor_cache.pt`: local, per-sign reservoir of actual TRAIN states and S,
  plus the original initialized query head. This file is not pushed. The cache
  is sign-stratified; use the complete CSV/summary for actual class prevalence.
- `actor_fit.json`: fits the exact original Actor objective on a small balanced
  cache, with unchanged input scaling and architecture. A separate BCE-on-sign
  control checks binary separability/optimization; it is diagnostic only and no
  BCE-trained model is saved or deployed. Both start from identical original
  initialization and see identical minibatches. Reports balanced accuracy,
  sign AUC, class recall, output distribution, continuous objective regret,
  initial outputs, value-head error and actual optimizer gradient/detach checks.
  If fewer than eight cached examples of either sign exist, the binary fit is
  marked inconclusive, not a failed Actor. Scoring uses the fitting rows and
  demonstrates capacity only, never generalization or final traffic accuracy.
- `routing_wiring.json`: checks route 0 against the Frozen forward, route 1
  against the Context forward, correctness-target orientation, actual routing
  loss gradient signs and parameter separation on TRAIN only.
- `routing_diagnostics.json`: posthoc per-level ranking/selection errors from
  the already completed v2 TEST traces. Separates a mapping error from choosing
  the wrong route despite correct mapping. TEST labels are observers only and
  do not fit thresholds, heads or rewards.

No conclusion that S is mostly negative should be generalized beyond the
logged TRAIN sample. If the prefix is inconclusive, increase the sampled
prefix (for example `-Rows 4096`, which covers each current Burst TRAIN domain)
or explicitly enable full training logs below. Stored source/code hashes
prevent silently reusing a different collection as the same diagnostic.

Repeat only the cached Actor audit, without recomputing counterfactual targets:

```powershell
.\AMCAL\run_amcal_minimax_diagnostics.ps1 -Dataset IAT -Phase Actor
.\AMCAL\run_amcal_minimax_diagnostics.ps1 -Dataset PS -Phase Actor
```

## Optional independent routing control

```powershell
.\AMCAL\run_amcal_minimax_diagnostics.ps1 -Dataset PS -Phase FixedRouter
```

This explicitly runs the seven TEST levels using the selected v2 Context and
Actor but the old source routing weights, fixed online. It preserves the 20%
per-level cap and reloads models at every intensity. It asserts that queried
rows and pre-feedback Frozen/Context predictions match the existing v2 run,
so the contrast concerns routing. It changes both routing initialization and
its online updating; it does not isolate those two effects from each other.
It is optional and is not run by default `-Phase All`. Results are separate
under `fixed_old_router`; no TEST-based model or threshold tuning occurs.

## Optional full TRAIN logging, after the small audit

```powershell
.\venv\Scripts\python.exe AMCAL/amcal_minimax_paironly.py --dataset IAT --action Train --train-diagnostics
```

This is an explicitly requested full training run with every visited TRAIN
row logged, rather than the short prefix. Its default output has a `_logged`
suffix so it cannot reuse an existing run lacking instrumentation. Logging has
an isolated reservoir RNG and no extra optimizer steps. A synthetic contract
test compares training with/without logging, including final parameter values.

## Send the short diagnostic outputs

```powershell
.\AMCAL\push_amcal_minimax_diagnostics.ps1 -Dataset Both
```

Only JSON/CSV inside the diagnostic run directories are staged. Cached tensors,
model files and raw datasets are excluded. Custom diagnostic output names should
retain the `minimax_diagnostics_seed...` prefix for this helper.

The runner checks dependencies without installing anything and runs the
`test_minimax*.py` preflight suite. Numeric Torch tests require the existing
experiment environment; a missing dependency is not reported as a passed test.
