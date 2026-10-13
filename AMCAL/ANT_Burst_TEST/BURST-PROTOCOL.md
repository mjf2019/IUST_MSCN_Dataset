# Burst AMCAL runner

Use the existing oversampled train/validation/test CSVs unchanged. No split or
new oversampling is performed. Train-only fitted scaler and label encoder are
saved and reused for all levels. Original Burst CNN and single-token Context
architectures and weighted train sampler are preserved.

The online gate requires DDQN action 1, different CNN and Context predictions,
a maximum-confidence gap >= threshold, and remaining label budget. Defaults:
threshold 0.0005, online learning rate 0.0005, label fraction 0.20. Both parameters
are explicit experiment settings; the legacy Burst notebook also contains other
learning-rate and threshold settings, so this is not an exact numerical replay.

Corrected evaluation: predict before acquiring a label/updating; base CNN weights
and buffers remain frozen; load the initial checkpoint separately on each CLI
invocation. DDQN uses gamma 0.95, nonterminal same-input local transitions,
label-free zero reward for skips, greedy online decisions, and target updates
every 15 training epochs / 15 successful online optimizations. Context uses the
article loss ratio without clamping, consistent eval mode at validation, and
independent best-weight snapshots. These differ from legacy notebook bugs.
The single-token architecture is retained as requested and does not establish
feature-to-feature attention, a min-max guarantee, or convergence.

The prepared validation and test files are used as supplied. Source independence
is not certified. Online budget counts file rows, not verified unique original
sources. Full-level results can therefore diagnose behavior but do not certify
held-out attack provenance. No training or evaluation was run by the assistant.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst.py audit
python AMCAL/ANT_Burst_TEST/amcal_burst.py train --base-epochs 200 --context-epochs 200 --pretrain-epochs 20
python AMCAL/ANT_Burst_TEST/amcal_burst.py evaluate --per 20 --budget-fraction 0
python AMCAL/ANT_Burst_TEST/amcal_burst.py evaluate --per 20 --budget-fraction 0.20 --threshold 0.0005 --lr 0.0005
```

Outputs: protocol_runs/burst_original_seed42/Models and Results. Existing model
folders cannot be overwritten by train; use --output for a fresh configuration.
--per supports 0,1,3,5,7,10,12,15,17,20. Each evaluation writes pre-update prediction,
label query, gate, losses and budget counters to a CSV trace and summary JSON.
