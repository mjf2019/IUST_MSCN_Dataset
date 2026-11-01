# Matched clean0 full RAL and corrected author AF with the full 20% cap

This opt-in experiment does not train AMCAL or modify old notebooks, models,
results or baseline algorithms. It uses the verified prepared domains and
partitions of `clean0_AMCAL_attacks_seed42`, matching the latest AMCAL Online
experiment. All 12 cases are evaluated: CW/DeepFool 0.1/1/5 and FGSM/PGD
0.01/0.1/0.5. No TEST row cap, new attack generation or TEST-fitted scaler.

## RAL full

- Train on **all clean TRAIN rows**, with the existing author-core committee:
  KNN5, RF10 and a decision tree. No attack TRAIN rows, no sliding window.
- At each intensity create a fresh runtime from this source model. Predict the
  current row first; only a legally acquired label can trigger an update.
- Every paid feedback calls `acquire_label`, appends that row, and refits on
  **all initial clean TRAIN plus all acquired feedback**. This intentionally
  expensive source-plus-growing-feedback behavior is recorded explicitly.
- The original author implementation separately fits the committee members and
  its voting ensemble; this control preserves that work rather than optimizing
  away repeated fits. A refit event denotes one feedback update, not one tree fit.
- Budget ceiling is `floor(0.2 * full TEST rows)` independently at each intensity.
  Default `prefix` also retains the author's causal ratio gate. It may spend less
  than the ceiling; actual use is reported. `-RALBudgetMode global` is an explicit
  alternative cap ablation, not the default paper gate.
- Source forgetting uses clean VAL, exactly the existing matched baseline
  observer. Prediction/update timing includes full refitting and runs on CPU;
  human label latency and model loading are not included.

## AF author full20

This uses the **corrected author-structure AF** implementation, not the historical
MLP/GRL online extension and not a frozen network called AF.

- Select 25 clean TRAIN examples per class, as in the corrected author runner.
  Original-domain support and preprocessing use only level 0.
- For every target intensity select exactly `floor(0.2*N)` calibration labels,
  using capacity-aware balanced round-robin allocation. Each class retains at
  least one held-out query candidate. This is an offline class-balanced support
  selection protocol, not a label-blind online acquisition policy.
- Fit a fresh AF network with the unchanged corrected author extractor, two
  optimizers, flipped-domain labels, feature-layer reset, and target tuning.
  Default 1000 adversarial iterations and up to 30 tuning epochs are retained.
- The fixed five-shot support is replaced by budget-sized support. Remainders
  and rare classes can make counts unequal. Target-only cosine distance-weighted
  kNN uses `k=max(1,floor(total support / number of classes))`, fixed before fit;
  this generalizes the author's k=N rule to the budget-sized support. The support
  counts and k are saved. This is a documented budget protocol adaptation, not
  a claim of identical numerical reproduction of fixed N-shot AF.
- `AFParameters.target_pool_per_class` is populated per case from the largest
  actual support count. It is descriptive; no hidden 20-rows/class cap is used.
- Score only remaining query rows. Exact feature duplicates of calibration or
  clean source support are purged from queries and counted; no query label is
  passed to fit/tuning/kNN. Source, target support and query IDs are recorded.
- Results are **offline post-calibration**. Do not call them pre-feedback online
  accuracy. Source forgetting has no clean-only AF checkpoint baseline in this
  author structure and is reported as undefined, not zero.

Both methods reset independently at each attack/strength. The inherited
historical attack pool-order assumption, where applicable, is kept in provenance;
this runner does not independently verify those historical mappings.

## Outputs and resuming

- RAL: `protocol_runs/clean0_RAL_full20_seed42_v1`
- AF: `protocol_runs/clean0_AF_author_full20_seed42_v1`

Both store per-case predictions, metrics, completion fingerprints and 12-case
summaries with macro and support-weighted PR/RC/F1. Rerunning skips verified
completed models/cases. An interrupted AF case repeats deterministically; no
requested output is deleted. Different recipes require fresh output paths.

`-Action Train` fits the RAL clean source model. For AF it prepares the clean
support identities only: AF joint fitting necessarily occurs with each target's
calibration support in `-Action All` or `-Action Evaluate`.

## User-run commands (no installation)

```powershell
git pull origin experiment/paironly-paid-replay
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_supplied_ral_af20.py -v
```

One terminal per method; each command covers all four datasets sequentially:

```powershell
.\AMCAL\run_supplied_ral_af20.ps1 -Method RAL -Dataset All
```

```powershell
.\AMCAL\run_supplied_ral_af20.ps1 -Method AF -Dataset All
```

For individual dataset runs replace `All` with `SDNCampus`, `ISCX-TOR`, `IUST` or
`IoT`. RAL is CPU-bound; `-Jobs 2` can limit RF workers when running several
datasets in parallel. AF uses CUDA by default. Use sequential runs for a clean
runtime comparison without resource contention.

Push each method separately:

```powershell
.\AMCAL\push_supplied_ral_af20_results.ps1 -Method RAL -Dataset All
```

```powershell
.\AMCAL\push_supplied_ral_af20_results.ps1 -Method AF -Dataset All
```

Only new result JSON/CSV files are staged. The exporter commits, synchronizes the
branch with rebase/autostash, and pushes without force. It stops if synchronization
fails; existing results remain preserved.
