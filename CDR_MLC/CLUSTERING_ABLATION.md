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
  development records are assigned to three approximately balanced random
  expert partitions using a seeded SHA256 hash of their opaque record IDs.
  This assignment uses neither features nor application/congestion labels,
  stays fixed during the final expert refit, and is never used at inference.
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
