# Causal window uncertainty sampling on full TEST

Reviewer1 requested DDQN versus uncertainty sampling and random selection; the
reviewer did not prescribe a75th-percentile threshold. This experiment adds
window-ranked controls without changing original code, checkpoints or results.
There is no TRAIN/VAL/TEST fitting, no threshold calibration and no installation.
The authoring assistant did not run tests, imports, training or inference.

## Defaults and causality

IAT beta26 / PS beta55; seed42; budget20% per level; every eligible official
TEST row of levels5/7/10/12/15/17/20. Historical TRAIN sizes and model architecture
remain unchanged. Original source audit exclusions are retained.
The window size is5: at20% budget this is the smallest regular window that can
rank five arrived samples and select one. It was not optimized on TEST.

All five predictions are fixed before requesting the selected label. The
highest arrival-time normalized Context prediction entropy is selected.
Only after window close is the label acquired and Context updated, affecting
subsequent windows. No unarrived sample or its label participates in ranking.
No score is recalculated after acquiring a window label.
Equal scores favor the earlier arrival. Selected labels are processed in arrival
order. The existing batch5 paid-feedback replay and source replay update remains
unchanged (LR .00025, replay .1); CNN and query weights stay fixed.
Window size5 is a query buffer, distinct from the existing Context replay batch.

Quota = floor(.2*window_end) - floor(.2*window_start), using zero-based exclusive
boundaries. This sums exactly to floor(.2*N). A partial final window is predicted
and ranked with its own quota, possibly zero. It never creates extra labels.

## Methods

* DDQNOriginal: unchanged per-row argmax request decisions and immediate feedback.
* UncertaintyWindow: highest entropy in each fully arrived window; no threshold.
* RandomWindow: uniform random priorities, same window quota/release times.
* DDQNWindowRank: highest Q_query-Q_skip, same window quota/release times.

The three window methods match quota and feedback release times. Sample-relative
delays depend on selected positions and are reported (zero to four rows by default).
DDQNWindowRank is an additional control, not a claim that forced top-quota selection
is the original DDQN policy. The difference from DDQNOriginal includes pacing and
feedback delay; do not attribute it solely to uncertainty or sample utility.
All methods use the same pretrained Context and source replay. These are query-policy
replacements, not independently retrained end-to-end AL ablations.
This is a buffered online protocol; whole-TEST ranking is not used.

TEST resets Context/Adam/memory/cap for every method and intensity. Online metrics
use the original pre-feedback predictions, including selected samples. Final target
metrics are separately evaluated on the same target rows after adaptation.
Trace rows store prediction step, feedback step, delay, arrival-time entropy/score,
quota and labeled counts. Labels of unqueried rows are observer-only.

## Run on the user's machine

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_causal_window_al.ps1 -Dataset IAT
    .\AMCAL\run_amcal_causal_window_al.ps1 -Dataset PS

Each command runs small CPU tests, then28 full TEST cases (four methods, seven levels).
No real-data training is started. Results are separate:
protocol_runs/causal_window_al_seed42_beta_26_w5_full_v1 (IAT) and
protocol_runs/causal_window_al_seed42_beta_55_w5_full_v1 (PS).

    .\AMCAL\push_amcal_causal_window_results.ps1 -Dataset Both

Only JSON/CSV results are staged. Earlier percentile controls and paper tables
remain unchanged pending review of these results. Completed cases resume only
after execution/metrics/trace checksum checks.

