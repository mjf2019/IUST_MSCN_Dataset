# Structured Burst bank experiment

Entry point: `AMCAL/amcal_burst_structured_bank_protocol.py`.
The previous bank runner is unchanged. This experiment retains three independent
Contexts, one frozen CNN and seven actions: direct prediction, three Context-only
predictions, and three query/update actions. Predictions are scored before feedback.

## Value representation

For routes 0..3, `Q = B + 2*p_correct - 1`. For query/update actions 4..6,
`Q = B + 2*p_correct_context - 1 + U_context`. Thus non-update route ranking is
exactly correctness ranking. A shared scorer handles each expert and its current
statistics; a frozen-route indicator distinguishes the fixed CNN. Context ordering
is permutation equivariant. TD optimization cannot change the correctness scorer.

Features include the existing CNN embedding, all route probabilities, confidence,
agreement, input distortion, and paid-feedback reliability summaries. No current
true label enters the state. The reliability summaries use pre-update predictions.

## Supervision and budget

Offline, existing TRAIN labels supervise correctness on every training row. These
are labels already available to train the source CNN. Context gradient updates
still obey the episode cap. This is an explicit change from paid-only offline
correctness supervision and must be reported when comparing methods.

Warmup episodes train correctness under random bank updates, followed by joint
DDQN training. Training order, query locations and exploration are resampled between
episodes. Validation partitions remain fixed; temporary online validation learners
are discarded after scoring. Source IDs are disjoint across train/validation/test.

Online, all supervised and TD updates require a paid label. The same paid label
reveals pre-feedback correctness for all four prediction routes; no extra label is
charged for those counterfactual prediction targets. Unchosen Context update effects
are not fabricated. Context replay is disabled. Router replay is bounded and weighted
by recency; hidden-bank TD replay is reset between offline episodes. Every evaluation
level starts with fresh bank weights, optimizers and replay.

Budget exhaustion disables query actions but does not terminate the value of future
predictions. Source CNN and initial Context files are copied unchanged and hashed.

## Controls and metrics

- `--query-schedule uniform`: exact label-blind global cap, default 20%.
- `--update-policy learned`: DDQN chooses the update destination (default).
- `--update-policy fixed`: label-blind common update destinations, uniform-only
  diagnostic; learned/random policies have identical Context update trajectories.
- `--bank-utility`: optional separate experiment using the minimum reference CE
  across the bank rather than selected-slot reference CE. Run without it first.
- `--context-lr`: Context adaptation LR; it is distinct from the router's LR.

The legal oracle respects the actual action mask and any fixed-update constraint.
The all-route oracle reports union correctness over all four pre-feedback routes.
Both are observers along the actual bank trajectory, not global adaptation optima.

`FixedRL` is a retrospective control: preserve actual query actions/destinations,
but use the best single fixed route on non-query rows. It is not an independently
trained policy. Reference scores show both routed accuracy and per-Context retention;
routed retention alone cannot establish that individual Contexts retained knowledge.
Reference state uses actual paid-feedback statistics rather than zeroing them.

Only concise reports and the combined table are printed. Metrics JSON and detailed
CSV traces are saved to Results. Trace files contain all route predictions and legal
oracle outcomes; they are never replay inputs. Configuration hashes separate outputs.

## Validation

CPU unit checks:

```bash
python -m unittest discover -s AMCAL/tests -p test_structured_bank_protocol.py -v
```

Checks cover exact route ranking, context permutation equivariance, gradient
separation, legal masks, identical-bank fixed controls, unchanged frozen weights,
and invariance to changes in unqueried test labels. A small end-to-end CPU training
and evaluation smoke run also exercises the real CNN/Context classes. These are
correctness checks, not evidence of GPU performance or benchmark accuracy.
