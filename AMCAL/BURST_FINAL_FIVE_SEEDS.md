# Final Burst five-seed completion

Use run_burst_final_five_seeds.py, not the old run_burst_iat_five_seeds.ps1
(which evaluates the previous seven-action AMCAL and historical AF extension).

Online evaluation seeds: 42, 48, 54, 62, 72. All online methods reuse their
existing offline checkpoints trained with seed 42. This measures stream-order,
query/adaptation randomness, not variance across independent offline trainings.
Levels 5,7,10,12,15,17,20 reset independently, budget floor(0.2*N), original CNN
IAT and Light CNN PS, final AMCAL one Context/three actions/batch5, bonus1,
online Context LR .00025 and replay CE .1. No model or method is changed.

AF is the corrected author-structure offline reference, with 25 original-domain
source examples/class and N=5 target calibration examples/class. A fresh AF fit
is performed for each calibration seed; its post-calibration metrics are NOT
prequential metrics or a matched 20% online-budget comparison.

## Collect before running (no experiments)

python AMCAL/run_burst_final_five_seeds.py --dataset IAT --action collect
python AMCAL/run_burst_final_five_seeds.py --dataset PS --action collect

Reads existing metric JSON and prediction CSV files. Confusion matrices and
macro precision/recall are computed, and recomputed accuracy/F1 must agree
with saved metrics to 1e-10. Outputs missing metrics/traces remain explicitly
marked. Source checkpoint hashes and result provenance are preserved.

## Complete only missing groups

python AMCAL/run_burst_final_five_seeds.py --dataset IAT --group AMCAL --action run
python AMCAL/run_burst_final_five_seeds.py --dataset PS --group AMCAL --action run

python AMCAL/run_burst_final_five_seeds.py --dataset IAT --group Baselines --action run
python AMCAL/run_burst_final_five_seeds.py --dataset PS --group Baselines --action run

python AMCAL/run_burst_final_five_seeds.py --dataset IAT --group AF --action run
python AMCAL/run_burst_final_five_seeds.py --dataset PS --group AF --action run

Existing complete compatible results are reused; if only a trace is missing
locally, its matching evaluation is repeated to produce predictions. No
offline online-method model is retrained. Source checkpoint absence/mismatch
fails explicitly. AF output attempts get fresh folders, preserving partials.
Groups may run in separate terminals, but do not compare timing from competing
GPU/CPU runs. Within each group dataset commands can be run sequentially.

## Push the small, JSON-only evidence bundles (PowerShell)

git add -f -- AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paper_final_five_seeds AMCAL/ANT_Burst_PS_TEST/protocol_runs/paper_final_five_seeds
git commit -m "Add final Burst five-seed metrics and confusion matrices"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git pull --rebase --autostash origin experiment/paironly-paid-replay
if ($LASTEXITCODE -ne 0) { throw 'Pull failed' }
git push origin HEAD

Bundles contain original metric summaries, derived metrics, confusion matrices,
trace hashes and source paths, without model weights or the full trace CSV.
Per-group coverage manifests record missing entries, not imaginary zero SD.
