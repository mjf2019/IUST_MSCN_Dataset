# Full replacement of the Active Learning Module

Both controls reuse the final one-Context batch5 checkpoint. IAT uses the
original CNN; PS uses its final Light CNN checkpoint. No offline retraining.
Each intensity resets all adaptive state. Budget cap 20%, Context LR 0.00025,
paid batch5 and protected replay weight 0.1 remain unchanged.

## Random

`--modes random_al` replaces both query decisions and prediction routing.
Draw a legal action uniformly (query probability 1/3 until the cap), then draw
Frozen or Context independently with equal probability. The independent
route is also used for the prediction on queried rows. A paid label still
updates Context after prediction. Neither selector forwards nor selector
optimization occur. This differs from the older random-action mode, which
couples queried rows to Context predictions, and from no_rl, which always
predicts with Context and uses a fixed uniform exact-cap query schedule.

## Uncertainty

`--uncertainty-query --uncertainty-routing --modes learned` replaces both
decisions. Despite the inherited mode name, no selector inference or learning
occurs. Compute each route's maximum class softmax probability. Predict with
the route having the larger confidence (ties choose Frozen), including on
queried rows. Query when 1 minus that confidence reaches a fixed threshold,
until the cap. The threshold is the 75th percentile of the same routed score
on original-domain VAL features with the initial CNN and Context. No VAL
labels or target data calibrate the threshold. Do not fill unused budget.
Actual label use may be below 20% and must be reported.

All predictions precede label access. Observer endpoints use the corresponding
random or confidence-based routing, never an RL head. Paid Context updates
and protected memory remain active. Output metadata and trace rows distinguish
these controls from previously reported query-only uncertainty results.

## Windows

Run `run_amcal_full_al_controls.ps1 -Dataset IAT` and the equivalent PS command
in separate terminals. Each runs Random and Uncertainty serially. Add
`-Control Random` or `-Control Uncertainty` to run just one.
After all four runs finish, use `push_amcal_full_al_controls.ps1`.

Validation: Python syntax checked; no experiments or package installations run.


The current runner passes `--uncertainty-quantile 0.75`; budget remains 0.2.
Calibration quantile and feedback cap are separate parameters. The new
`component_full_uncertainty_q75_seed42` tag preserves archived q80 outputs.
The push helper selects the existing Random control and the new q75 control.
