# Learned seven-action acquisition (opt-in v2 checkpoint)

This experiment replaces the external `quality` query controller with a learned
query policy. It does **not** guarantee better accuracy or a desired minimum
label count. The cap is an upper bound, independently for each test level.
Old checkpoints and their `policy`, `uniform`, `fixed`, and `quality` controls
retain the old semantics unless `--budget-aware-policy` is enabled.

## Action semantics

| Action | Scored prediction BEFORE feedback | Update after paying for a label |
|---|---|---|
| 0 | Frozen CNN | None |
| 1, 2, 3 | Corresponding Context then Frozen CNN | None |
| 4, 5, 6 | Best of the four routes according to the learned correctness head | Corresponding Context 1, 2, 3 |

Querying a currently weak Context no longer forces a poor current prediction
through that Context. All route logits are already available as in the previous
bank. The correctness head selects the query action's pre-feedback route;
no true label is used to choose it. Random control draws query prediction routes
uniformly, and draws its seven actions uniformly until the cap, then four routes.
Epsilon exploration remains part of the learned policy. Only zero remaining
budget masks query actions; there is no audit schedule, bootstrap quota, minimum
spending, or accuracy gate. Query slots and prediction routes are both in traces.

## State and learning

The previous shared correctness network and its route features remain intact.
TD gradients do not enter that network. Only the budget/update value heads gain
36 features: eight global resource/quality features and seven per route.
These include pre-feedback accuracy estimates, confidence/support and freshness,
past reference gain/harm, remaining labels per remaining row, spending pressure,
and the resource price. The query score is now

`Qquery_k = common_future + max_route(2*p_route-1) + update_advantage_k`.

In addition to ordinary DDQN loss, a paired loss directly learns

`Qquery_k - max(Qskip) = utility_k + gamma*(V(actual_next)-V(no_query_next))`.

The no-query successor uses the pre-update bank, unpaid reliability state,
unspent label and no-query resource-price transition. The actual successor
includes the selected update, paid reliability and consumed label. Both use the
same next input and terminal flag. This is one-step counterfactual credit, not
an oracle query policy, future-label access, or global adaptation optimum.
Replay is cleared when banks reset between training episodes; test levels each
start from the saved policy, empty replay, fresh optimizers and fresh banks.

## Quality estimates and reference data

Query probability under epsilon-greedy behavior is logged **before** feedback:
`epsilon*(legal_query_actions/legal_actions) + (1-epsilon)*I(greedy_is_query)`.
Random behavior uses `legal_query_actions/legal_actions`. Only a paid label
supplies four pre-feedback route outcomes. Recent estimates use clipped
inverse-propensity weights (max 20), prior shrinkage and effective sample size.
Online route-supervision losses use the same clipped importance weights;
offline dense TRAIN supervision does not need selection weighting.

Clipping trades bias for variance: these estimates are not guaranteed unbiased
population accuracies. Epsilon zero can leave regions without any feedback.
After exhaustion there are no new labels to measure drift; freshness and
support are exposed to the policy rather than fabricated outcomes.

The saved quality prior is Frozen/initial-Context accuracy on source **TRAIN**
rows, not a claimed heldout clean accuracy. It may be optimistic. Quality need
is bounded and cannot create a reward on its own. Source validation labels are
not used to build priors or the training-reference memory.

At most 256 balanced clean TRAIN anchors are selected once, and their source
IDs are saved. Before/after probes mix these with an independent TRAIN batch
offline, or previous paid rows from this level online (up to 16 each). When the
paid history is empty, probes use TRAIN anchors. Feature-identical current rows
are excluded from both pools. References receive **no Context gradient**; only
the current paid row is used in the Context objective. This is reference memory,
not Context replay or inter-level replay. Unqueried test labels never enter it.

## Utility and resource price

For the same reference rows, let `Lbefore_i,k` and `Lafter_i,k` be the selected
Context's CE, and let `w_i,k` be its pre-update predicted route weight.
Weights are `0.9*softmax(correctness_logits) + 0.1/4`, frozen for the probe.

`delta_i = w_i,k * (Lbefore_i,k-Lafter_i,k)/(Lbefore_i,k+Lafter_i,k+1e-8)`

`G = mean(max(delta_i,0)); H = mean(max(-delta_i,0))`.

This rewards a routing-weighted loss surrogate, not an all-route true-label
oracle. Loss normalization already existed in the older implementation; the
change is attribution, state, paired credit and the prediction/update separation.

`utility = gain_weight*(1+need_weight*need_k)*G - retention_weight*H - selection_cost - price`

`need_k` is the larger confidence-weighted positive deficit of the system or
destination against the TRAIN Frozen prior (bounded by 1). No measured transfer
means no gain bonus. Harm is never turned into positive benefit by a deficit.
Source-memory retention and target adaptation share the reference batch; the
reported independent clean validation observer remains separate.

The multiplier updates after each action using

`price_next = clip(price + price_step*(query - remaining_cap/remaining_rows)/sqrt(1+position/32), 0, price_max)`.

The default step is 0.005, maximum 0.2. Fast spending raises its future cost;
waiting reduces it. This is a soft pacing pressure in the policy's reward/state,
not an additional action selector or an obligation to spend. The base cost
remains when the price reaches zero. It is a proposed heuristic, not a theorem
that this price is the true optimal shadow cost. Tests verify the mechanics,
not benchmark improvements.

## Short separate runs (PowerShell)

Use the already prepared data and trained structured-bank source; nothing is
downloaded and Frozen/initial Context checkpoints are copied unchanged. The
trained correctness scorer is reused; new value heads must be trained.
The helper defaults to 12 epochs, patience 5, one correctness warmup episode,
training levels 0/1/3, LR 0.0005, cap 20%, and gamma 0.99. Its longer discount
horizon is an experimental choice rather than an established improvement.

```powershell
git pull --ff-only origin experiment/amcal-protocol

# IAT: training only (training seed 42)
powershell -ExecutionPolicy Bypass -File AMCAL/amcal_burst_budgetrl_run.ps1 -Command train -Representation IAT

# IAT: evaluation only (evaluation seed 48; epsilon 0.1)
powershell -ExecutionPolicy Bypass -File AMCAL/amcal_burst_budgetrl_run.ps1 -Command evaluate -Representation IAT -Seed 48 -EvalEpsilon 0.1

# PS: training only (training seed 42)
powershell -ExecutionPolicy Bypass -File AMCAL/amcal_burst_budgetrl_run.ps1 -Command train -Representation PS

# PS: evaluation only (evaluation seed 48; epsilon 0.1)
powershell -ExecutionPolicy Bypass -File AMCAL/amcal_burst_budgetrl_run.ps1 -Command evaluate -Representation PS -Seed 48 -EvalEpsilon 0.1
```

Default source per representation:
`protocol_runs/structured_bank3_policy_lr0p0005_seed42`.
Default new output:
`protocol_runs/structured_bank3_budgetrl_v1_lr0p0005_seed42`.
Use `-SourceRun` / `-OutputRun` to override existing paths; a training output must
be fresh. Changing evaluation seed changes online randomness, not source splits.
Full per-level console tables, runtime, budgets, retention, oracle observers,
and CSV/JSON artifacts remain available. JSON is saved, not dumped to console.

## Interpretation and acceptance

The cap is never a target lower bound. Compare accuracy, Macro F1, clean and
same-level retention, actual labels, and runtime, using the same source models,
splits, order and evaluation seeds. For a matched-resource routing comparison,
keep the old uniform-feedback control; this new policy learns acquisition too.
Random and learned policies can acquire different rows/counts. Multiple seeds
and validation-only hyperparameter selection are needed before claiming a
dataset-level improvement. Reusing previously explored test levels means these
are development results, not a new untouched final test.

CPU checks:

```bash
python -m unittest discover -s AMCAL/tests -p 'test_*budget*.py' -v
python -m unittest discover -s AMCAL/tests -p 'test_structured_bank_protocol.py' -v
```

Background motivating selection weighting and budget-conditioned value learning:
[Importance Weighted Active Learning](https://arxiv.org/abs/0812.4952),
[Budgeted Reinforcement Learning](https://budgeted-rl.github.io/).
These works do not establish guarantees for this neural non-stationary bank.
