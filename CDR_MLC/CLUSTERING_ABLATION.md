# Paired removal of congestion clustering

Run from the repository root in the existing experimental Python environment:

```powershell
python CDR_MLC/clustering_ablation.py --seeds 42 --output CDR_MLC/outputs/clustering_ablation_seed42
```

The default runs all seven protocols. Scenarios 1–3 use the **entire** unseen
target congestion level (`--target-test-fraction 1.0`); LM-H, LH-M, MH-L and
chronological ALL-80-20 use the existing shared protocol builder. Window sizes
are 3 for TTFEF and 50 for the retained congestion descriptors. Both variants
score exactly the same complete-window records and use the same chronological
55/15/15/15 development partitions. Cold-start records stay excluded in both.

## Intervention

- `MF_Full`: the unchanged production MF-CDR-MLC, with MBK-trained geometry and
  three congestion-specific experts.
- `MF_No_Clustering`: no MBK or router scaler is fitted. The same eligible
  development records receive independent uniform draws from {0, 1, 2}
  using a seeded NumPy PCG64 generator. Filenames, record identities, features
  and application/congestion labels do not enter the random draw. No quotas,
  stratification, balancing, rejection or redrawing are applied.
  Existing draws are saved and preserved during final refit; newly eligible
  development records receive draws from the continued RNG stream.
  Integer DataFrame indices are only lookup keys for saved assignments.
  Assignments are never used at inference.
  All three experts are evaluated for every eligible input record.
- Centroid distances and the three one-hot cluster indicators are removed from
  utility and meta-fusion inputs. Expert probabilities and confidence evidence
  remain, as do the separate causal congestion descriptors, utility models,
  meta-fusion, balancing and confidence fallback. The fallback selects the
  expert with maximum estimated utility.
- Both use the separated expert inputs and the same 110-tree budget:
  3×20 expert trees, 3×10 utility trees and 20 meta trees. Selection is performed
  separately for each variant using the same development-only procedure.

This is a **random-partition control for congestion-based expert specialization**.
It is not a single pooled RF, and it does not remove all congestion information.
Three experts are retained so expert count and tree budget do not change.
It measures the combined effect of congestion-based expert partitioning and
the geometry features produced by clustering; it cannot separate those effects.
TTFEF is retained to match eligibility, so this experiment is an accuracy
ablation, not a claim about speed gains from eliminating TTFEF.

## Outputs

- `per_seed_metrics.csv`: both methods for every completed protocol/seed.
- `summary.csv`: protocol-level mean and standard deviation across seeds.
- `paired_deltas.csv`: **no clustering minus full** on absolute score units;
  multiply by 100 for percentage points. Negative values favor clustering.
- `protocol_means.csv`: unweighted means across completed protocols per seed.
- `mean_paired_deltas.csv`: mean paired differences across completed protocols.
- `manifest.json`, `input_audit.csv`, per-pair `pair.json` and predictions:
  data/code hashes, partitions, expert group counts, selection trials and
  checks that geometry was removed while context and scored records matched.

Until all seven protocols finish, the printed means are partial means. A
single-seed result is descriptive; do not present it as a multi-seed significance
test. For ten paired seeds:

```powershell
python CDR_MLC/clustering_ablation.py --seeds 42 52 62 72 82 92 102 112 122 132 --output CDR_MLC/outputs/clustering_ablation_10seeds
```

Use `--resume` with the identical command to continue an interrupted run.
Existing output directories require `--resume` or a new `--output`; results
from different data, code or configurations cannot be silently combined.

Synthetic verification (not paper results):

```powershell
python -m unittest discover -s CDR_MLC/tests -p "test_*ablation.py" -v
```

`use_clustering` defaults to `True` in the production configuration. Existing
experiment commands retain their MBK path unless the new option is explicitly
disabled by this runner.

## Small MBK initialization comparison

Only the number of MBK initialization candidates changes. The default remains
10. Run the current and higher setting against the same No_Clustering control,
using fresh output folders after pulling this change:

```powershell
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 10 --output CDR_MLC/outputs/clustering_ninit10_seed42
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 50 --output CDR_MLC/outputs/clustering_ninit50_seed42
```

Compare protocol_means.csv and paired_deltas.csv in the two folders.
No_Clustering does not use this parameter and should have identical predictions
for identical data, seed and other settings. The fitted router's n_init is
recorded in pair.json and the requested value in manifest.json. This parameter
selects among initialization candidates, not multiple complete MBK fits.
Use development validation to select a setting; test comparisons are descriptive
and must not be used to choose the final configuration.

## MBK batch size and iteration limit

The runner also accepts --mbk-batch-size (default 1024) and --mbk-max-iter
(default 100). Both must be positive integers and affect only the MBK variant.
They are recorded in manifest.json and the fitted router audit in pair.json.
Increasing max_iter only raises the iteration limit; convergence may stop sooner.

For one-factor comparisons at seed 42 and n_init=26, run each setting in a new
output folder. Every command evaluates both MF_Full and MF_No_Clustering:

```powershell
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 26 --mbk-batch-size 1024 --mbk-max-iter 100 --output CDR_MLC/outputs/mbk26_b1024_i100_seed42
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 26 --mbk-batch-size 2048 --mbk-max-iter 100 --output CDR_MLC/outputs/mbk26_b2048_i100_seed42
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 26 --mbk-batch-size 1024 --mbk-max-iter 300 --output CDR_MLC/outputs/mbk26_b1024_i300_seed42
```

For a combined exploratory setting, pass --mbk-batch-size 2048 and
--mbk-max-iter 300 together and use another output folder.
All other preprocessing, windows, experts and evaluation protocols are unchanged.

## RNG control version

This replaces the previous filename/row-identity hash control. Historical
results are a different control experiment and must not be resumed or pooled
with new results. Use a new output directory. Each pair.json stores the final
random assignments, integer row lookup keys and generator state for auditing.
Source filenames remain available to data-split/overlap checks, but are never
read by the RNG partition function.

This seeded uniform design is reproducible and independent of labels.
Group sizes are random, often close to equal by chance; it does not
force unequal sizes or deliberately weaken the control.

## Optional class-covered MBK expert partitions

--mbk-label-aware adds a supervised training-only constraint: each observed
application class must appear at least once in every expert partition.
MBK still fits its three centers on the earliest eligible expert partition
using the original TTFEF geometry. After fitting, constrained assignments
minimize total squared distance to those frozen centers subject to class
coverage. Distinct class-specific anchors are selected with a minimum-cost
assignment; all other records keep their nearest-center assignment.
No record is duplicated, oversampled or discarded; equal sizes or class
proportions are not required.

The same constraint is applied to authorized full-development expert refit.
If an observed class has fewer than three eligible records, training fails
explicitly rather than silently duplicating data or relaxing coverage.

This is MBK plus label-constrained training partitioning, a new variant, not
ordinary unsupervised MBK. Inference does not require labels and uses the
frozen centers/geometry as before. For the moved training records, expert
membership differs from nearest-center membership; this mismatch and the
performance impact must be assessed, not presumed beneficial. Since the
fusion evaluates all experts, it does not require a test label to select one.

The random control remains independent uniform RNG and ignores this flag.
pair.json records initial/refit per-class coverage, number/fraction reassigned,
distance penalty, and initial/final expert class counts. No later utility,
meta, selection or test labels are used in preliminary constrained assignment.

```powershell
python CDR_MLC/clustering_ablation.py --seeds 42 --mbk-n-init 26 --mbk-batch-size 1024 --mbk-max-iter 100 --mbk-label-aware --output CDR_MLC/outputs/clustering_classcovered_seed42
```

Use a new output folder. The original MBK remains the default without the flag.
Results of this new variant must not replace the original method's results
without updating its definition, training algorithm and experimental reporting.
