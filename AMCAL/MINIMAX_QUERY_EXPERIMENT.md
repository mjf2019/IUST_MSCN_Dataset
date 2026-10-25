# Experimental adversarial query selection

This is an opt-in method variant. It neither changes the existing PairOnly
runner nor overwrites any source model or previous result. No dependencies are
installed by the scripts.

## What is optimized

For each TRAIN state, first fix the prediction using the larger of the two
pre-feedback routing scores. The current-row gap is
`d = CE(Context(x), y) - CE(Frozen(x), y)`.

Copy the Context together with its Adam state, execute one candidate update,
and measure:

- `G`: signed mean CE reduction on the next independent TRAIN block;
- `F`: mean positive CE increase on an independent clean TRAIN reference;
- `S = d + gain_weight*G - forgetting_weight*F - cost`.

There are no updates within the future block and no bootstrap. All three
quantities have CE-loss units. The default weights are pilot settings, not a
claim that the components have equal distributions or that the settings are
optimal. The future block is of length 16 before removing duplicate features;
the final 16 TRAIN positions have no future-block target.

The query actor directly minimizes `-sigmoid(actor(s))*stopgrad(S)`: this is
the max step of a frozen-round, two-action full-information surrogate. A
parameter-disjoint value head also regresses `S` for diagnostics. The value-head
regression is not described as performing the max optimization. TEST decisions
use the actor, not the diagnostic value head.

The Context minimizes the same current-row gap, with prior-feedback/source
replay as a retention regularizer. A single forward/backward combines the
current row, up to four previously paid labels (batch5), and protected replay.
Current-row CE has weight one; retained rows have mean weight 0.1. This replaces
the legacy nonlinear alpha objective intentionally, so its gradient matches
the gap used in the adversarial component.

**Scope of the min-max claim:** it is the shared current-row gap in alternating
query-max / Context-min optimization, augmented by detached one-update utility
and harm for the actor and a selected-label replay regularizer for the Context.
It is not an implementation of differentiation through the entire future
adaptation process, a convergence guarantee, or a claim that the separately
supervised routing objective is part of one zero-sum game. In particular, replay
is conditional on a paid update; it is not the unconditional replay term of the
idealized discussion equation.

## Data and state isolation

- Existing IAT original-CNN / PS Light-CNN preprocessing, CNN, pretrained
  initial Context and PairOnly routing initialization are reused. The new
  query actor/value heads start fresh; this is not a neutral pre-RL ablation.
- TRAIN intensities are 0, 1 and 3. Reserve approximately 10% of clean TRAIN
  IDs, stratified by class, for forgetting probes and exclude those IDs from
  experimental updates/future probes at all TRAIN intensities. These rows may
  have been seen during existing source pretraining; independence applies to
  the experimental update being measured, not to the full source pretraining.
- Current row, future block, and reference are excluded from replay using
  source IDs and feature hashes. Paid batch rows and protected replay are also
  disjoint. Feature-identical copies of the current row are removed from the
  future block. Empty future blocks do not create a target.
- Keep the existing protected-memory source/recent/historical capacities and
  reservoir rules. Probe exclusions change eligible replay rows. This pilot
  samples the eligible protected rows uniformly, not with the legacy segment/
  class quotas, so it is a new experimental protocol.
- The cloned Context/Adam branch never mutates the live model, optimizer,
  memory or Torch RNG. If the pre-feedback decision requests a label, commit
  exactly that measured branch, including its Adam moments. Otherwise discard
  it. TRAIN future labels are never inserted into paid memory by probing.
- VAL selects the best epoch using mean online accuracy minus positive
  clean-source accuracy drop. Horizon, cost and weights are explicit pilot
  choices; compare runs on VAL before fixing them. Do not select on TEST.

## TEST protocol

Each intensity 5, 7, 10, 12, 15, 17 and 20 starts from the selected experimental
Context, routing and query checkpoint, with fresh Adam, reliability and paid
memory. The cap is 20% at each level, with no forced quota filling. Prediction
always precedes label access. Query and skip use the same best-scored route.

Context and routing learn only from requested labels. Query actor/value and
legacy future/update-value heads stay fixed. Legacy future/update-value heads
are unused. There is no bonus. Snapshots assert these fixed weights and the
Frozen classifier remain unchanged.

Outputs include macro AC/PR/RC/F1 in percent, actual label usage, clean TEST
before/after metrics and original-domain forgetting for every metric. The
before/after target endpoint reuses scored target rows: it is descriptive, not
an independent generalization estimate. Hidden TEST labels are used for
reporting only, except the requested labels made available after prediction.

## Run

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_minimax_paironly.ps1 -Dataset IAT
.\AMCAL\run_amcal_minimax_paironly.ps1 -Dataset PS
```

Use separate terminals for the two runs if GPU memory permits. Each runner
first performs the isolated CPU preflight checks below. Completed
training/evaluation is reused; an interrupted training directory is preserved
and requires a fresh `-Output`. An interrupted evaluation resumes at the first
level without a completed metrics/trace pair. Changed training parameters need
a different output directory; the default name includes horizon and cost.

```powershell
.\AMCAL\push_amcal_minimax_results.ps1 -Dataset Both
```

Only these experimental JSON/CSV outputs are staged. Unrelated staged files
cause an explicit stop; model checkpoints and raw datasets are not added.

Optional isolated CPU checks, using the existing environment:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_minimax_query.py -v
```
