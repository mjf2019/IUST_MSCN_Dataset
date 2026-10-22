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
