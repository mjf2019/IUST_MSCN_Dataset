# Matched baselines for the final structured AMCAL bank

Entry point: `AMCAL/amcal_burst_baselines.py`. Existing AMCAL runners,
notebooks and CDR_MLC implementations are not modified.

## Common experiment

- Use the final AMCAL run's immutable `protocol_preprocessing.joblib`: same
  original source IDs, training/validation/test partitions, class mapping,
  feature order, train-fitted scaler, and official cross-intensity row maps.
- Train/develop on level 0 and the same two source-training shifts (default
  1/3). Only train IDs enter fitting/template construction; only validation IDs
  enter early stopping and k selection. All seven unseen intensity levels use
  only heldout test IDs. No test row, label, extrema, or future target feature
  enters offline learning.
- Independently reset each method, model, optimizer and feedback history at
  every test level. Stream permutation is NumPy default_rng(seed), identical
  to structured AMCAL. Score each prediction before feedback.
- Global label cap is floor(0.2*n), not per class or per-flow. Unique source
  IDs occur once per level. RAL chooses labels by its existing policy. AF/DFE/ARF
  use label-blind uniform query positions with the same scheduler seed as the
  structured uniform diagnostic. CNN is a static zero-label baseline. Equal
  cap does not imply equal actual query positions/counts; record both.
- Save per-row traces, metrics JSON (not printed), a compact combined table,
  clean-reference retention and paired initial/final same-level retention on
  all rows and unqueried rows. Final scores are retrospective, not online.
  Unqueried subsets differ by policy and are not an independent fixed holdout.
- All five methods report actual acquired `Labels`, allowed `Cap`, `Label%`
  (100*Labels/stream rows), and `CapUsed%` (100*Labels/Cap), per level and in
  the combined table. CNN uses zero labels. Metrics also save remaining labels;
  zero-cap utilization is defined as zero. A floor cap of 214/1073 is 19.94%
  of rows, not exactly 20%. These reporting additions require evaluation only.
- `online_seconds` excludes the endpoint observers. AF's DANN/re-embedding
  work is included; it is not presented as free inference.

## Runtime reporting (evaluation only)

All five baselines and the structured bank print prediction, selection and online
update wall seconds per level. A separate combined `*_timing.csv` table also
reports Other(s), total Online(s), ms/row, rows/s and update ms/label, with per-method
means. The metrics JSON retains full precision and device information. CUDA is
synchronized at stage boundaries; CPU-only RAL/ARF are timed on CPU even if the
other methods use CUDA. Timers do not change queries, random draws or gradients.

The timed interval excludes model loading/initialization, CSV input/output and
initial/final retention observers. Update includes RAL full refits, AF's paid
DANN/re-embedding, DFE template refresh, ARF learn_one and AMCAL reward probes and
selector learning. AMCAL frozen logits/embedding cache construction is charged
once to prediction and total; Cache(s) is a subset of prediction, not an extra
charge. Other(s) includes trace construction, oracle bookkeeping and progress
printing. Human label turnaround time is not simulated. Endpoint oracle forwards
are not timed. The existing batched frozen cache is retained; these results
describe this implementation, not an uncached request-at-a-time service benchmark.
Run methods sequentially on the same idle machine for timing comparisons; record
CPU/GPU differences. Training is unnecessary to add timings to existing evaluations.

For prepared **PS** data, an entry point separates complete training and testing:

```powershell
python AMCAL/amcal_burst_matched_run.py train --representation PS --seed 42 --device cuda
python AMCAL/amcal_burst_matched_run.py evaluate --representation PS --seed 42 --device cuda
```

Training first creates (or reuses a completed) PS-specific two-stage source run
on 0/1/3, including twenty second-stage Context pretraining epochs. It then trains
the final structured bank (three Contexts, seven actions, policy queries, Context
LR .0005), followed by CNN/RAL/AF/DFE/ARF matched baselines. Testing uses only
5/7/10/12/15/17/20, epsilon .05 for the bank, per-level independent resets and a
20% cap. Source CNN/Context are never taken from IAT. No downloads are performed.
Existing final AMCAL/baseline outputs are not overwritten during training.
Flags `--source-run`, `--amcal-run`, `--baseline-run`, and `--dataset` override paths.
The same entry point supports `--representation IAT`; existing separately trained
baselines can instead be evaluated individually with the original runner.

## 1D-CNN

The CNN architecture is exactly the one in `ANT_Burst_TEST/1D_CNN.ipynb` and
the current shared core (six convolutions; three pooling blocks; dense 256/128).
Reuse the identical final AMCAL frozen CNN checkpoint, already trained on the
declared source domains, rather than wastefully retraining a different random
CNN. Record its SHA256. It remains in eval mode with no online updates and
uses zero target labels. This is the static baseline, not CNN fine-tuning.

## RAL

`baselines/ral_existing.py` contains the two classes extracted unchanged from
`ANT_Burst_TEST/RAL.ipynb`, checked against the author's `SAWassermann/RAL`
implementation. Use the paper's k-NN (k=5) / RF (10 trees) / decision-tree committee,
soft vote, uncertainty threshold 0.9, greedy probability 0.025, eta 0.01,
reward +1 / penalty -1, multiplicative alpha/threshold updates and refitting
on all source plus acquired rows after every query. Estimator seeds are fixed.

The old notebook's evaluation called only `evaluate_metrics`, so did not
perform online acquisition at all. The new runner calls `ask_committee`,
`should_label`, and `acquire_label` after the scored prediction. By default,
retain the paper's prefix-ratio gate: acquired labels / samples seen must be
less than the budget fraction before a new query. Also enforce the shared
floor(fraction*n) total cap externally, preventing a final rounding overshoot.
`--ral-budget-mode global` disables the prefix gate for an explicit ablation.
Endpoint observers call predict only, never refit or acquire labels.

The earlier wrapper inherited modern sklearn's default RF=100; older sklearn
used 10, as in the paper's committee. Retrain RAL in a fresh output directory;
old RAL checkpoints are rejected. The other methods need no retraining for this
correction. Hyperparameters above follow the paper's MAWI configuration, with
the common Burst 20% budget and splits. Predict-before-feedback and retention
are shared benchmark metrics, additional to the paper's validation evaluation.

## AF-Burst-causal

AF means **Adaptive Fingerprinting**, not Adaptive Random Forest. Reuse the
corrected `CDR_MLC/benchmarks/AF/af_single_source.py` classes directly:
`TabularFeatureExtractor`, `AFDomainNetwork` and `GradientReversal`. GRL sign
is retained: discriminator minimizes domain CE, extractor receives its negative
gradient. Continuous Burst IAT/PS is not a -1/+1 packet-direction sequence, so
use the explicitly adapted MLP rather than mislabel the direction DF CNN.

Offline DANN uses clean TRAIN IDs as labeled source and level-1/3 TRAIN IDs
as unlabeled domains (30 epochs, Adam 1e-5, batch 32, GRL 1, embedding 512).
The initial normalized k-NN support can use all labeled source-development
TRAIN rows from 0/1/3, matching the full-source fairness policy. Choose voting
k from 1/3/5/7 using validation only.

At each uniformly acquired test label, append the paid sample to k-NN support.
Once two paid samples exist, take one DANN minibatch using labeled clean TRAIN
samples and ONLY already-acquired target features. Re-embed source/paid support
after each extractor update; never keep stale embeddings. No future/unqueried
target feature is used to fill a batch. This causal streaming wrapper, merged
support, global percentage budget and validation-selected k are benchmark
adaptations; they are not the paper's batch few-shot M=25/N-per-class protocol.
They are labeled `AF-Burst-causal` in metrics, not an exact original AF result.

## River ARF

`--methods arf` runs `river.forest.ARFClassifier`, default ten trees and one
source-training pass, as in the earlier standalone ARF benchmark. Source rows
from the declared levels 0/1/3 are combined and shuffled for each training pass.
The saved forest includes its drift/warning detectors, background trees and RNG.
At every test level, deep-copy the trained forest; do not use River `clone()`,
which would instead construct an untrained estimator.

ARF predicts before feedback and calls `learn_one` ONLY on paid uniform query
positions, shared with the AF/DFE baselines. The global cap is exact. Endpoint
observers only predict; unqueried labels cannot update trees or detectors.
`--arf-trees` and `--arf-train-passes` configure training and are saved together
with the River version and samples/pass. Unlike the old standalone benchmark's
hard-coded 0/3/12 source levels, this runner takes the final AMCAL training levels.
ARF is a CPU algorithm; using `--device cpu` avoids unnecessary GPU allocation.

## DFE-Burst-adapted: paper audit

Reference: Wang et al., *DFE: Deep Flow Embedding for Robust Network Traffic
Classification*, IEEE TNSE 12(3), 1597-1612 (2025), DOI
10.1109/TNSE.2025.3535577. Implementation: `baselines/dfe_burst.py`.

Retained from Fig.2 / Sec.III / Sec.IV-C:

- Thirteen main 3x3 convolutions, stride 2 at convolutions 3/6/10, residual
  stages, adaptive average pooling, fc256 and a 256-dimensional embedding.
- Euclidean-distance constrained triplet loss (Eq.12), inter-class margin
  alpha1=0.5, intra-class threshold alpha2=2.0. Retain literal max(dp,alpha2),
  rather than subtracting the constant eta*alpha2 from printed loss values.
- Total loss = Gaussian compression bound + lambda*triplet loss (Eq.13).
  Six output stages define six conditional Gaussian mixture bounds (Eq.6),
  using diagonal Gaussian KL (Eq.7), a stable log-sum-exp including self terms.
  KL is between samples/components in the SAME transition/output space, not
  between unrelated means from different-width input/output layers.
- SGD 0.001, batch 32, at most 200 epochs, patience 10. Triplets exclude the
  anchor from positives and draw negatives only from other labels. All offline
  source TRAIN labels are available to this supervised method.
- Twenty separate randomly selected training templates per class. Classify by
  L2 nearest-template majority vote (Eq.14). Do not collapse them into centroids.
  Online adaptation changes templates only; backbone and BN buffers are frozen.

Explicit reproduction choices / limits:

1. The paper extracts 81 handcrafted features into a 9x9 image; Burst provides
   100 sequence features. Preserve all 100 in a 10x10 image in original order.
   For another width, zero-pad to ceil(sqrt(width)) squared; never drop columns.
   Fit MinMax scaling on source TRAIN rows only, clip transformed values to
   [0,1]. No per-test-set normalization.
2. The paper does not specify a practical tensor estimator for its Gaussian
   noise parameters. Use per-sample channel spatial means/variances, with a
   declared variance floor 1e-4. There are no random or unregistered projection
   layers in the loss. This implements a declared estimator of its equations,
   not verified bit-for-bit author code.
3. Six beta coefficients are indexed, but Sec.IV-C lists only five numbers.
   Default (.01,.02,.03,.04,.05,.05), explicitly repeating the final value.
   Eta=.5 and lambda=1 are declared choices because their numerical values
   are not reported. CLI flags expose all these values.
4. The paper does not give voting k or a causal template replacement rule.
   Select k among 1/3/5/7 on validation, retain all twenty individual templates,
   and replace the oldest same-class template when a paid example arrives.
   No class-balancing oracle is used for test selection; current true labels
   become available only after payment/prediction.

The older Burst notebook used squared differences of scalar log variances as
its compression term; that is not Eq.6/7. It is not reused for this benchmark.
No claim of exact reproduction of the paper's original dataset results is made.

## Run (PowerShell, repository root)

```powershell
git pull --ff-only origin experiment/amcal-protocol
python -m pip install psutil
python -m pip install river==0.26.1
```

Training only (fresh output):

```powershell
python AMCAL/amcal_burst_baselines.py train --source-run AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_policy_lr0p0005_seed42 --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/baselines_levels1_3_seed42 --methods cnn ral af dfe arf --train-levels 0 1 3 --seed 42 --af-epochs 30 --dfe-epochs 200 --patience 10 --arf-trees 10 --arf-train-passes 1 --device cuda
```

Testing only (after training):

```powershell
python AMCAL/amcal_burst_baselines.py evaluate --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/baselines_levels1_3_seed42 --methods cnn ral af dfe arf --levels 5 7 10 12 15 17 20 --budget-fraction 0.2 --seed 42 --device cuda
```

To run one method in another tab, use `--methods af` or `--methods dfe` with
a distinct fresh output directory, and use the identical output when testing.
Do not change AF online LR between comparisons without recording it (included
in execution hashes). RAL fits its 10-tree RF both separately and inside the
soft-voting committee after every acquired sample, as in the author code.
Keep update frequency 1 for the main comparison. Progress prints every 200
stream rows; set `--progress-every 0` to disable it.

CPU correctness checks:

```powershell
python -m unittest discover -s AMCAL/tests -p test_burst_baselines.py -v
```

These test math, architecture, template voting/replacement, label causality,
budget enforcement, AF GRL/paid-only DANN, frozen DFE weights and all four
train/save/load/evaluate paths on a tiny fixture. They are not benchmark
performance measurements; full CUDA training/testing is run by the researcher.
The additional ARF check uses a real River forest to verify save/load, independent
clones, observer isolation, exact query cap and invariance to unqueried labels.
