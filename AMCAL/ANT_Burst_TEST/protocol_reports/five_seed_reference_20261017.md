# Five-seed reference results

Evaluation seeds: 42, 54, 67, 81, 96. Training checkpoint: three_action_cost0p1_ep100_pat15_seed42 (one training seed). Levels: 0,1,3,5,7,10,12,15,17,20. Context LR .25; window 1; selection cost .1. Mean and sample SD below are across five seed-level means, each averaged over ten perturbation levels. These are evaluation seeds, not independent training runs.

## Separate class-as-flow Contexts; shuffled; global 20% cap

| Policy | Accuracy (%) | Macro F1 (%) | Mean labels |
|---|---|---|---|
| Adaptive, epsilon .1 | 84.870457 +/- 1.651943 | 82.349104 +/- 1.492717 | 206.76 |
| Random three actions, epsilon 1 | 77.101584 +/- 1.040114 | 76.792904 +/- 1.287372 | 214 |
| Frozen DDQN, epsilon .1, full budget | 69.164958 +/- .826896 | 67.473631 +/- 1.050352 | 214 |
| Rolling uncertainty, full budget | 72.300093 +/- .480184 | 71.203990 +/- .377471 | 214 |

Adaptive accuracy exceeds each comparator for all five evaluation seeds. Mean accuracy gaps: random 7.768872 pp; frozen 15.705499 pp; uncertainty 12.570363 pp. Frozen and uncertainty use causal forced-tail completion: mean forced queries 46.32 and 185.8 respectively. Equal final label counts do not mean equal feedback timing. Adaptive consumes fewer labels in four of five seeds. Group identity in this controlled dataset is true class, explicitly marked oracle_context_routing in outputs; measured real Flow IDs were not available.

## Separate protocol: shared Context; input order; offline class quotas

| Policy | Accuracy (%) | Macro F1 (%) | Mean labels |
|---|---|---|---|
| Adaptive epsilon 0 | 77.170550 +/- 1.473238 | 72.452287 +/- 1.153309 | 141.08 |
| Random three actions epsilon 1 | 73.058714 +/- 1.548627 | 71.618410 +/- 1.906658 | 212 |

Adaptive accuracy exceeds random for all five seeds. Mean gap 4.111836 pp; 33.452830% fewer labels. This supports joint-policy label efficiency, not isolated sample informativeness because routing also differs.

## Source logs

First protocol: Pasted text(20261016-203138).txt; Pasted text (2).txt; Pasted text (3).txt; Pasted text (4).txt. Second protocol: Pasted text (5).txt; Pasted text (6).txt. Uploaded 2026-10-17 (Tehran). Do not pool the two protocols.
