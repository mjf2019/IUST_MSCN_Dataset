# CALMID and MicFoal: equation audit and Python implementation

Entry point: `AMCAL/amcal_burst_baselines.py --methods calmid micfoal`.
Implementation: `baselines/calmid_micfoal.py` and `baselines/weighted_arf.py`.
The existing five baselines and AMCAL model architectures are unchanged.

## Primary sources

- Liu et al., **A comprehensive active learning method for multiclass imbalanced
  data streams with concept drift**, Knowledge-Based Systems 215 (2021) 106778,
  DOI `10.1016/j.knosys.2021.106778`. Read Algorithms 1--3 (PDF pp.6--7),
  Eqs.1--4 (pp.7--8) and default parameters in Sec.4.2.
- Liu et al., **Multiclass imbalanced and concept drift network traffic
  classification framework based on online active learning**, Engineering
  Applications of Artificial Intelligence 117 (2023) 105607,
  DOI `10.1016/j.engappai.2022.105607`. Read Algorithms 1--3 (PDF pp.7--8),
  Eqs.1--3 (p.6), and periodic performance adaptation in Sec.3.5 (p.9).
- Author CALMID Java: https://github.com/sunnyweike/CALMID/blob/master/CALMID.java
  (blob `7813f131587bf051c5d88c44bebf6a3a2e1bcac5`).
- Author MicFoal Java:
  https://github.com/sunnyweike/MicFoal/blob/master/Source%20code/MicFoal.java
  (blob `87aa7e31af0cacdac16e04aa34f102db3336e5fd`).
- Python backends: River **0.26.1** HoeffdingTreeClassifier, ADWIN and ARFClassifier.

The equations and printed pseudocode are the primary specification. The public
Java implementations were checked as additional evidence, not copied blindly:
they have several differences from the published algorithms, listed below.
This is an equation-based Python implementation with an explicitly declared
River backend, not a claim of bit-for-bit equivalence with MOA or reproduction
of the numerical results on the authors' original datasets.

## Shared benchmark protocol

- Initialize with **all available source TRAIN rows**, from the same domains,
  partitions, train-fitted scaling and class mapping as the matched AMCAL run.
  Shuffle those combined TRAIN rows once with the training seed. No test or
  validation row is used for fitting. The paper's first-500 stream-label
  bootstrap is replaced by already available source training: neither method
  receives 500 free test labels. Training labels are outside the target budget,
  just as for every existing baseline.
- Source fitting trains the configured base learners with native Poisson(6)
  bagging and unit source weights. It fills the latest source-label windows;
  CALMID caches recent source examples with neutral bootstrap weights.
  CALMID source ADWIN detectors receive post-update errors. This is supervised
  benchmark initialization, not active querying on source rows.
- Before each target level, deep-copy the entire trained model and histories,
  reset stream query counters and AL RNG, and set the native starting budget
  to `--budget-fraction`. Source-window contents remain available initially.
  No target buffer or learned weights transfer between test levels.
- Predict and score before feedback. `request(x, can_query)` accepts no true
  label. Only `update(x, paid_y, decision)` receives one. Threshold correction,
  class discovery, weight computation, training, performance monitoring and
  drift detection cannot use unqueried truth.
- Every target row ages the label windows, including budget-exhausted rows;
  unavailable labels are placeholders. Random-only label counts estimate
  class proportions; uncertainty-selected labels do not contaminate that
  estimator. Source labels are known bootstrap data, not queried target data.
- Preserve the papers' native query-ratio rules, including random requests
  outside the native ratio gate. Independently impose the common strict
  `floor(budget_fraction * target_rows)` cap on **all** requests. Thus adaptive
  MicFoal's internal budget can change while actual target use never exceeds
  the common 20% ceiling. There is no forced budget exhaustion.
- Clean reference and same-level all/unqueried retention observers only predict;
  they never supply labels, alter windows or call a model update. The common
  synchronized CPU timing includes query decisions, window maintenance, paid
  updates and periodic performance adaptation, but excludes endpoint observers.

## CALMID

Defaults: ten Hoeffding trees, label window 500, threshold matrix initialized to
0.5, random probability 0.1, threshold step 0.01 and ADWIN delta 0.002.

For top classes `c1,c2`, margin `m=p(c1)-p(c2)`, and paid truth `y`:

| Component | Implementation |
|---|---|
| Algorithm 1 | Random request first. Otherwise uncertainty/selective requests require `queried/seen < budget`. The uncertainty comparison is `m <= M[c1,c2]`. |
| Algorithm 2 | Correct uncertain prediction decreases `M[c1,c2]` by `(1-s)`, and again if imbalance exceeds 0.5; a second-class error decreases it once if imbalance exceeds 0.5. Selective second-class errors set it to `max(theta, M*(1+s))`. The opposite matrix entry is not changed. |
| Eq.1 | Difficulty is `1-m` when correct, `(1+m)*exp(1)` for second-class errors, `(1+m)*exp(2)` for other errors. |
| Eq.3 | Imbalance is `random_count[y] * C / number_of_random_labels`. Use the author's count floor of one for an absent class; use neutral imbalance one for a wholly empty random window, avoiding undefined divisions. |
| Eq.2 | Sample weight is `log(1 + difficulty + 1/imbalance)`, without an added unit weight. |
| Online bagging | Independently draw `Poisson(sample_weight)` per member; pass this count to weighted incremental learning, not a full-data refit. This follows the paper's explicit Poisson(weight) construction. |
| Sample memory | Per-class FIFO capacity `ceil(window * random_ratio / C)`; only available source and paid target examples are stored. |
| Eq.4 / Algorithm 3 | On drift, sort cached examples by arrival, recompute imbalance-dependent weights and multiply by `exp((arrival-current)/window)` before Poisson resampling. Replace the member with largest ADWIN error and reset its detector. |
| Drift ordering | Use per-member post-update classification error, as in the paper's train-then-drift sequence and the public Java code. Only paid errors enter detectors during testing. |

### Differences in the public Java implementation

The Java version uses `<` for uncertainty versus `<=` in Algorithm 2, places
random selection inside its budget check versus outside in Algorithm 1, and can
leave its random-label window frozen after native budget exhaustion. It also
uses `Poisson(6) * (1 + computed_weight)` instead of Algorithm 3's explicit
`Poisson(weight)`. Those are not silently presented as equivalent. Python uses
the paper comparisons/order/Poisson construction and ages every row. The paper
does not round its fractional per-class capacity; Python uses the author's ceil
rule. The numerical zero-count guards are explicit implementation choices.

## MicFoal

Defaults: ARF with ten trees and Poisson(6), three time-aligned windows of size
500, initial per-class least-confidence threshold 0.9, random ratio 0.1,
paper threshold step **0.1**, periodic low/high Macro-F1 bounds 0.6/0.9.

| Component | Implementation |
|---|---|
| Algorithm 2 | Native ratio `sigma=queried/seen <= b` gates uncertainty and selective sampling. Uncertainty means `confidence <= V[predicted]`. Otherwise selective probability is `(b-sigma)/(b-sigma+confidence-V[predicted])`. Independently draw the random request afterwards; preserve both flags if both request. |
| Algorithm 3 | `n_class` is the number of distinct **paid/available true classes in current Wt**, not the lifetime global class count. `n_all` and `n_y` count random-only labels in Wr. |
| Difficulty | Correct: `1 + V[y] - confidence`; incorrect: `2*(1+n_class)`. |
| Imbalance/weight | If `n_y=0`, `mu=0`, weight `1+log(1+difficulty+n_all)`. Otherwise `mu=n_y*n_class/n_all`, weight `1+log(1+difficulty+1/mu)`. No artificial extra count is added to these denominators. |
| Eqs.1--2 | A correct uncertain prediction multiplies its threshold by `0.9**2` for `mu>=1`, or by `0.9` for `mu<1`. |
| Eq.3 | An incorrect non-uncertainty-selected prediction multiplies its threshold by `1.1`. No Java-only floor `max(initial_threshold, ...)` is inserted and thresholds are not silently clipped at one. |
| Class evolution | A newly acquired truth expands the threshold vector to that new class with initial value 0.9. No unqueried label can expand it. |
| Period evaluation | Every 500 stream rows, calculate mean class F1 over classes present in Wt, using only the paid pre-feedback predictions in Wp. If Wt has no labels, skip the budget correction. |
| Budget correction | Below 0.6 increase `b` by factor `1+(0.6-F1)/10`; above 0.9 reduce by `1-(F1-0.9)/100`. Bound to `[min(epsilon,b0), min(1,2*b0)]`. Sec.3.5 specifies direction, but not numerical rates/bounds; these are drawn from the official Java PeriodicEvaluate/adjustBudget implementation. Bound checks are implemented by clipping, avoiding the Java branch that can reduce an enlargement at its upper bound. |

### Weight support in River ARF

River 0.26.1 ARF accepts `**kwargs` but ignores an outer `w`. Passing the
MicFoal training weight to unmodified ARF would therefore silently discard a
central component. `WeightedARFClassifier` retains River's original ARF loop
and wraps its tree factory. A shared scalar multiplies **each foreground and
background tree's** native Poisson count exactly once by the current sample
weight. Warning-created trees and drift replacements use the same factory.
The weight can be fractional; it is not rounded into repeated learn calls and
does not trigger another Poisson draw. The scalar resets after each call.
Deep-copy/joblib preserves this shared reference. With outer weight one,
predictions and tree training counts match ordinary River ARF exactly in tests.

### Differences in the public Java implementation

Its confidence-threshold step is 0.01, while published Eqs.1--3 specify 0.1;
its minority cutoff is 0.5, while Eq.2 specifies 1. Its weight implementation
uses lifetime classes and cached/smoothed imbalance ratios instead of the exact
current-window counts in Algorithm 3. It includes a new-class budget boost not
specified in the paper and requires an external caller for periodic evaluation.
Python uses the printed equations/current-window counts, expands classes without
that extra boost, and performs periodic evaluation automatically after each
completed window. Label-dependent corrections are moved strictly after payment.
These choices are deliberate and tested, not claimed to match Java bitwise.

The River ARF and Hoeffding-tree feature observers, split policies, memory
management and random-number sequences are not identical to the MOA engine.
The configured native learner is **MicFoal-ARF**; this implementation is not
labelled MicFoal-LB. Macro-F1 on the queried window can be selection-biased;
no source-validation or unqueried test labels are used to "repair" it.

## Run only the two new methods (PowerShell, repository root)

```powershell
git pull --ff-only origin experiment/amcal-protocol
python -m pip install river==0.26.1
```

Train in a **fresh output**, without retraining the existing five baselines:

```powershell
$burstIAT = "AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT"
$activeRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/calmid_micfoal_levels1_3_seed42"
python AMCAL/amcal_burst_baselines.py train --source-run AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_policy_lr0p0005_seed42 --burst-dir $burstIAT --output $activeRun --methods calmid micfoal --train-levels 0 1 3 --budget-fraction 0.2 --active-models 10 --active-window 500 --active-random-ratio 0.1 --seed 42 --device cpu
```

Test after training:

```powershell
python AMCAL/amcal_burst_baselines.py evaluate --burst-dir $burstIAT --output $activeRun --methods calmid micfoal --levels 5 7 10 12 15 17 20 --budget-fraction 0.2 --progress-every 200 --seed 42 --device cpu
```

All thresholds, window sizes and backend configuration are training parameters
saved in the model. `--micfoal-fixed-budget` selects a separate **MicFoal_hw**
training ablation; default MicFoal includes the adaptive native budget while
still respecting the common target cap. Query flags can overlap in MicFoal;
their displayed counts should not be summed as a label count. Actual acquired
labels and cap utilization are reported separately, together with reference
retention, paired unqueried retention and split runtime tables/CSV.

For PS, use its own structured source run and `ANT_Burst_PS_TEST` data/output
paths. Neither method requires CUDA. The existing matched-run convenience
script's explicit five-method list is retained for compatibility with ongoing
PS runs; the baseline runner accepts the two additional names.

## Checks

```powershell
python -m unittest discover -s AMCAL/tests -p test_calmid_micfoal.py -v
python -m unittest discover -s AMCAL/tests -p test_burst_baselines.py -v
```

Checks cover difficulty/weights, asymmetric thresholds, current versus lifetime
class counts, native/random budget rules plus the hard cap, placeholder aging,
new classes, periodic budgets, worst-tree replacement and chronological decay,
fractional foreground/background weights, unit-weight equivalence to River,
save/load, observer isolation and invariance to every unqueried true label.
End-to-end train/save/load/evaluate checks use a small fixture. They verify
correctness, not performance on the full Burst GPU/CPU experiment.
