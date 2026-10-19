# Quality/yield feedback control for the structured 3-Context bank

This is an **optional evaluation-only hybrid controller** on an existing trained
checkpoint (`--query-schedule quality`). It is not a newly trained RL sample
selector. It leaves the 3 Contexts, 7 actions, network state width, route scores,
update-destination scores, losses, rewards and online learning rates unchanged.
The controller decides **when to acquire a label**. On that row, the existing
policy chooses among update actions 4--6; on other rows, among prediction actions
0--3. The same masking mechanism already used by `uniform` is used here.

`policy` and `uniform` retain their original behavior, defaults and result hashes.
No new source model, attack level or target-label bootstrap is needed. Changing
feedback naturally changes learned Context/selector trajectories; unchanged
scoring architecture does not guarantee unchanged accuracy.

## Offline reference and causal signals

1. Compute **one scalar** before the stream: frozen CNN accuracy on the clean
   SOURCE validation IDs, using the saved source preprocessing. It is a quality
   reference, not an attainable accuracy guarantee on attacked inputs.
   No validation example enters online gradients, replay, or transfer probes.
2. Reserve label-blind monitoring positions, using independent seeded RNGs.
   Bootstrap: 12 positions spaced across the first 15% of the stream. Audits:
   `ceil(0.1 * cap)` random positions outside bootstrap, bounded by the cap.
   These are paid target labels, charged exactly like every other acquisition.
3. At each monitor, record the **pre-feedback prediction-only route** that the
   existing router would use without a query gate (including the configured
   epsilon, using an isolated RNG). This requires no additional route forwards.
   Compare that shadow prediction with the now-paid label. Do not use the forced
   update-slot prediction as a proxy for overall router quality. The scored online
   output remains the actual chosen action's pre-feedback prediction.
4. Estimate recent no-query routed accuracy from the latest 32 paid monitors:

   `A_hat = (sum(correct_monitors) + 2*A_clean) / (n_monitors + 2)`.

   The clean accuracy supplies only a weak two-observation prior. This is a
   finite-sample proxy for prediction-only routing, not exact full-test accuracy.
   Adaptive-query labels do not enter this quality window. Shadow predictions on
   unqueried rows cannot affect the controller.
5. Measure adaptation yield on the existing paid-history transfer probe,
   excluding the current and feature-identical examples. No source-validation
   or unqueried target label enters that probe. Before/after normalized CE change
   is the existing `(L_before-L_after)/(L_before+L_after+1e-8)`.
   For each paid update with a valid probe, store

   `net_transfer = gain_weight*gain - retention_weight*harm`.

   Use the latest 12 valid updates. No-probe updates are **unknown**, not zero
   evidence. A finite trial ends after `max(bootstrap_count, 2*min_return_updates)`
   paid labels (16 by default): if no valid probes exist then, adaptive requests
   pause while reserved monitoring continues. Probe measurements concern previously queried examples and can be
   selection-biased; they do not guarantee future generalization.

## Budget gate

```
deficit = max(0, A_clean - A_hat)
need = clip((deficit - 0.02)/0.15, 0, 1)
yield_factor = 1                         # fewer than 8 probes, within finite trial
yield_factor = 0                         # trial ended, no valid probe
yield_factor = min(1, n_probes/8) * clip((mean(net_transfer)-selection_cost)/0.001, 0, 1)
                                         # otherwise
p_adaptive = need * yield_factor
```

On reserved bootstrap/audit positions, request a label regardless of the signals,
provided the global cap remains. Otherwise, request with `p_adaptive` only when:

- `used < floor(cap*(position+1)/stream_rows) + 4`, limited by the total cap;
- enough labels remain reserved for **every future monitor**.

The four-label burst permits limited early adaptation. Unused released capacity
carries forward; there is no forced final exhaustion. Monitoring is never
cancelled merely because recent yield is poor, so later changes can be detected.
If adaptation is persistently harmful or at a plateau, a large clean-accuracy
gap alone cannot trigger unlimited extra acquisitions. The 12/32/8 windows,
10% audit reserve, 2 pp tolerance, 15 pp gap scale, and 0.001 yield scale are
explicit initial controller settings, **not performance-tuned claims**.

For PS (886 rows, cap 177), the defaults reserve 12 bootstrap and 18 audit labels
inside the 177-label cap. For IAT (1,073 rows, cap 214), they reserve 12+22.
Small/zero caps are explicitly bounded. The controller is reset for each level.

The next gate is planned once after the current paid feedback, before the next
sample's truth is accessed. Behavior and actual DDQN transition backups receive
the same next gate. Counterfactual skip backups hold that realized external next
gate fixed; they are controlled one-update comparisons, not a simulation of a
different controller history. Selection and controller RNGs are isolated.

## What a comparison means

- `learned` retains online route supervision and DDQN update-destination learning.
- `random` uses random legal routes/destinations without selector learning, with
  the **same controller algorithm and parameters**. Realized correctness/yield
  can differ, so the two modes can acquire different rows and label counts.
- `uniform` is still the matched-feedback routing/update control. Keep its prior
  PS result (177 labels, 58.63% online RL accuracy) as an external reference, not
  as a value inserted into the new metrics.
- A successful trade-off requires lower label consumption **and** sufficiently
  preserved online accuracy/retention. Saving labels alone is not success.
  Attribute feedback control to this explicit hybrid, not purely to RL.

## Test PS with the existing checkpoint (PowerShell)

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/amcal_burst_structured_bank_protocol.py evaluate --burst-dir AMCAL/ANT_Burst_PS_TEST/Dataset_official_v1/AdvBurst_FTSC-PS --output AMCAL/ANT_Burst_PS_TEST/protocol_runs/structured_bank3_policy_lr0p0005_seed42 --levels 5 7 10 12 15 17 20 --context-lr 0.0005 --budget-fraction 0.2 --query-schedule quality --update-policy learned --eval-epsilon 0.05 --selection-cost 0.0001 --gain-weight 1 --retention-weight 1 --modes learned random --seed 42 --device cuda
```

For IAT, substitute `ANT_Burst_IAT_TEST` and `AdvBurst_FTSC-IAT` paths.
No retraining is required for this controller ablation. The output still points
to the existing Models directory. Quality settings/reference/version enter the
configuration hash, so the new results do not overwrite uniform/policy files.

Per-level console output adds the reference, monitored no-query quality, recent
transfer yield, bootstrap/audit/adaptive counts, need and yield factors. Standard
online, budget, timing and retention tables remain. Per-row trace columns begin
with `quality_` and preserve decision-time signals, monitoring route/prediction,
release/reserve values and reasons. Full configuration and final controller
diagnostics are also saved with each metrics file; no JSON is dumped to console.

Optional flags use the defaults listed above:
`--quality-bootstrap-labels`, `--quality-audit-fraction`, `--quality-window`,
`--quality-return-window`, `--quality-min-return-updates`, `--quality-tolerance`,
`--quality-gap-scale`, `--quality-gain-scale`, `--quality-burst`.
Choose hyperparameters on development data, not by repeatedly optimizing test
level results. `--bank-utility` retains its prior optional probe definition;
it is not enabled by this controller.

## Checks

```powershell
python -m unittest discover -s AMCAL/tests -p test_quality_budget.py -v
python -m unittest discover -s AMCAL/tests -p test_structured_bank_protocol.py -v
```

Checks cover weak/useful versus weak/harmful/plateau/good streams, gradual release,
reserved monitoring, strict and zero/small caps, paid-only feedback and sequential
gate planning. Tiny neural integration tests verify frozen CNN preservation,
state/reward compatibility, actual DDQN successor masks, save/evaluate reports,
and identical actions, weights and controller state when all unqueried labels
are changed. Existing bank tests cover routing semantics and previous controls.
These are correctness checks, not full PS performance measurements.
