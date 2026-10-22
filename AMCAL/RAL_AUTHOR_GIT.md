# RAL: clean-0 training and independent attack-level evaluation

Use `run_ral_author_levels.ps1` for the requested AMCAL comparison protocol:

```powershell
.\AMCAL\run_ral_author_levels.ps1 -Dataset IAT -Seed 42 -BudgetFraction 0.2
.\AMCAL\run_ral_author_levels.ps1 -Dataset PS -Seed 42 -BudgetFraction 0.2
```

Initial supervised training uses level 0 only. The author example's .5% bootstrap
size is retained (fraction of the whole clean CSV, selected exclusively from
eligible TRAIN identities in file order). It does not fit all clean TRAIN rows.
Clean VAL is an observer only; it supplies no training labels or model selection.
All supplied attack levels 1,3,5,7,10,12,15,17,20 are TEST only before streaming.
Only the held-out TEST identities from the existing partition are streamed.

Save one clean-0 checkpoint, then deserialize that SAME checkpoint before EVERY
level. Each level starts with the initial source history, alpha, threshold and
zero acquired/seen counters. No feedback or fitted target model crosses levels.
Each level has its own floor(.2 * level TEST sample count) hard label cap, in
addition to the author's native prefix-ratio gate. This is a maximum, not a
forced quota. Predictions are recorded BEFORE acquiring the current label.
Only acquired labels enter adaptation. Every update retains all initial clean
bootstrap rows plus acquired feedback from the CURRENT level, as in author RAL.
The original author core file remains unchanged. The experiment adapter changes
the data split, stream order and hard per-level ceiling to the user's protocol;
it is not the original single-file example experiment.

The saved preprocessing artifact supplies feature schema and source identity
partitions only. No CNN/Context weights, saved scaler or attack-domain fitting
are used. Raw numeric permitted features are passed to the original classifiers.
Inputs, checkpoint hashes, split identities, installed versions and actual default
classifier parameters are recorded. The author uses default constructors;
installed sklearn determines RF tree count (not an explicit RF10).

Outputs for each dataset:
`protocol_runs/ral_author_clean0_levels_budget0.2_seed42/`
contains `Models/ral_author_clean0.joblib`, `execution.json`, and `Results/`
with per-level prequential metrics/traces, combined_results.csv and summary.json.
Source retention is measured on clean VAL before/after each independent level.
The final retrospective target score includes queried rows and is explicitly
separate from the pre-feedback online accuracy used for comparison.
Prior outputs are preserved; use a new --output / -Output for a repeat run.
No numerical experiments were run; Python syntax was checked.

The earlier `run_ral_author_git.ps1` below remains a separate single-file example
adapter. It is NOT the runner for clean-0-to-held-out-attack comparisons.

---

# RAL using the original author source and example protocol

Author: https://github.com/SAWassermann/RAL
Pinned commit: 2a34a4b1b1e11054a718233e19de0c2134fd68e8
baselines/ral_author.py is the complete unmodified upstream ral.py. Do not edit it.
Old matched-source RAL and recent5 results are preserved separately.

```powershell
.\AMCAL\run_ral_author_git.ps1 -Dataset IAT -Level 0 -Seed 42
.\AMCAL\run_ral_author_git.ps1 -Dataset PS -Level 0 -Seed 42
```

The original repository is data-agnostic: its main has empty data_X/data_y
placeholders. Our adapter supplies a single selected numeric Burst CSV and omits
unrelated identity/metadata features. Defaults use level 0. To supply another
CSV, use Python --csv; -Level selects that entire official file. Initial training
therefore comes from THAT file too. Running -Level 5 is NOT clean-only source
training followed by attack testing, and must not be described as such.

Unmodified example settings: first .5% of rows for initial fit; next 69.5% stream;
last approximately 30% validation, using the exact round/index slicing of the
original. No shuffling, scaling, class balancing, full-source training or added
global cap. Native query ratio budget .05, uncertainty .9, greedy .025, eta .01,
reward 1, penalty -1, update frequency 1. All initial and acquired rows remain in
online refits. No recent5. Initial and final validation evaluation uses the
author's evaluate_performance method, including its extra voting-classifier fit.

The same default constructors are used: KNeighborsClassifier(),
RandomForestClassifier(), DecisionTreeClassifier(). The upstream variable is
named rf10 but n_estimators is NOT explicitly passed in its source. Actual
RF/default parameters depend on the installed sklearn version and are recorded
in execution.json (modern versions typically create 100 trees). Do not silently
replace defaults with RF10 and call it exact source behavior. This is exact
source/settings reproduction, not a guarantee of historical numeric equivalence:
upstream pins no dependencies and has no input dataset. A Python/NumPy seed is
added only for reproducibility and explicitly recorded; constructors stay intact.

Additional pre-feedback predictions record online metrics without updating RAL
or exposing unqueried labels. Original endpoint validation accuracy is the primary
reported author-protocol metric. Supplied file order is retained; actual traffic
timestamp ordering is not independently established. No execution tests run.

Output: protocol_runs/ral_author_git_level0_seed42/{execution.json,summary.json,
summary.csv,trace.csv,split_manifest.csv}. Previous outputs are never overwritten.
This protocol differs from AMCAL's matched 60/20/20 + held-out-identity experiments;
its scores must not replace matched RAL results in the same comparison table.

Budget update requested by user: adapter default is now .2, passed to the
unmodified author's native budget argument. Every other setting stays unchanged.
Use -BudgetFraction 0.05 to reproduce the Git example's budget. New output names
include budget, e.g. ral_author_git_level0_budget0.2_seed42, so old .05 outputs
are preserved. Report the .2 run as original RAL code with an adapted 20% budget,
not all-original-example hyperparameters. No additional global cap is added.
No execution tests run; Python syntax checked.
