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
  IDs occur once per level. RAL chooses labels by its existing policy. AF/DFE
  use label-blind uniform query positions with the same scheduler seed as the
  structured uniform diagnostic. CNN is a static zero-label baseline. Equal
  cap does not imply equal actual query positions/counts; record both.
- Save per-row traces, metrics JSON (not printed), a compact combined table,
  clean-reference retention and paired initial/final same-level retention on
  all rows and unqueried rows. Final scores are retrospective, not online.
  Unqueried subsets differ by policy and are not an independent fixed holdout.
- `online_seconds` excludes the endpoint observers. AF's DANN/re-embedding
  work is included; it is not presented as free inference.

## 1D-CNN

The CNN architecture is exactly the one in `ANT_Burst_TEST/1D_CNN.ipynb` and
the current shared core (six convolutions; three pooling blocks; dense 256/128).
Reuse the identical final AMCAL frozen CNN checkpoint, already trained on the
declared source domains, rather than wastefully retraining a different random
CNN. Record its SHA256. It remains in eval mode with no online updates and
uses zero target labels. This is the static baseline, not CNN fine-tuning.

## RAL

`baselines/ral_existing.py` contains the two classes extracted unchanged from
`ANT_Burst_TEST/RAL.ipynb`. Retain its k-NN / RF / decision-tree committee,
soft vote, uncertainty threshold 0.9, greedy probability 0.025, eta 0.01,
reward +1 / penalty -1, multiplicative alpha/threshold updates and refitting
on all source plus acquired rows after every query. Estimator seeds are fixed.

The old notebook's evaluation called only `evaluate_metrics`, so did not
perform online acquisition at all. The new runner calls `ask_committee`,
`should_label`, and `acquire_label` after the scored prediction. Disable the
class's original prefix-ratio budget and enforce the common global cap outside
it; this is an explicit budget-protocol adaptation, not a silent algorithm
change. Endpoint observers call predict only, never refit or acquire labels.

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
```

Training only (fresh output):

```powershell
python AMCAL/amcal_burst_baselines.py train --source-run AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_policy_lr0p0005_seed42 --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/baselines_levels1_3_seed42 --methods cnn ral af dfe --train-levels 0 1 3 --seed 42 --af-epochs 30 --dfe-epochs 200 --patience 10 --device cuda
```

Testing only (after training):

```powershell
python AMCAL/amcal_burst_baselines.py evaluate --burst-dir AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT --output AMCAL/ANT_Burst_IAT_TEST/protocol_runs/baselines_levels1_3_seed42 --methods cnn ral af dfe --levels 5 7 10 12 15 17 20 --budget-fraction 0.2 --seed 42 --device cuda
```

To run one method in another tab, use `--methods af` or `--methods dfe` with
a distinct fresh output directory, and use the identical output when testing.
Do not change AF online LR between comparisons without recording it (included
in execution hashes). RAL refits a 100-tree RF twice per acquired sample, as
in the supplied implementation; CPU runtime is expected to exceed a single
neural minibatch. Keep update frequency 1 for the main comparison.

CPU correctness checks:

```powershell
python -m unittest discover -s AMCAL/tests -p test_burst_baselines.py -v
```

These test math, architecture, template voting/replacement, label causality,
budget enforcement, AF GRL/paid-only DANN, frozen DFE weights and all four
train/save/load/evaluate paths on a tiny fixture. They are not benchmark
performance measurements; full CUDA training/testing is run by the researcher.
