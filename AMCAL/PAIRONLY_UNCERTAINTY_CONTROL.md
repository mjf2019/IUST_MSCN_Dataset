# Matched PairOnly uncertainty-query control

Use `amcal_final_enhancement.py evaluate --uncertainty-query` with the existing
one-Context batch5 checkpoint. This evaluation-only control replaces query
timing by least confidence: `1 - max(softmax(frozen_logits))`.

The threshold is the `(1 - budget_fraction)` quantile on level-0 VAL features.
Calibration uses no target features or labels and no validation labels.
For each arriving TEST row, request feedback when its score reaches the
threshold, until the per-level hard cap is exhausted. Future TEST scores are
not ranked or used to choose the current threshold. No tail quota filling is
performed; actual label use can be below the cap and must be reported.

Prediction routing on non-query rows remains the inherited learned policy.
Queried rows retain the ordinary Context prediction before receiving their
label. Paid batch5 Context optimization, protected replay weight 0.1,
Context LR 0.00025, and paid-only selector updates are unchanged. Bonus must
be zero because the threshold, rather than a query action score, controls
acquisition. The policy state records whether a query is allowed on the row.

Use seed 42, budget 0.2, levels 5 7 10 12 15 17 20, independent resets,
`--context-count 1 --modes learned --query-schedule policy
--update-policy learned --paid-replay-mode batch5 --query-budget-bonus 0
--online-memory-ce-weight 0.1 --context-lr 0.00025 --endpoint-metrics`.
Set a new evaluation tag such as `component_uncertainty_seed42`.

IAT uses `paironly_batch5_trainseed42/paironly_batch5`; PS uses
`light_ps_paironly_batch5_trainseed42/paironly_batch5`. No checkpoint or
architecture is retrained or changed. Outputs include calibration provenance,
threshold and per-row uncertainty, predictions, labels, and endpoint metrics.
Existing commands retain their behavior when the flag is absent.

The older `amcal_burst_uncertainty_query.py` targets the coupled-v1 structured
bank and is not a matched comparison for these final PairOnly checkpoints.

Validation: Python syntax checks only; experiments run locally by the author.
