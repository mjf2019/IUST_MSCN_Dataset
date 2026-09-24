# CDR-MLC — paper implementation

## Active work

- `DATASETS/CDR-MLC/New_Version/`: 15 Argus CSV exports, five applications × three congestion levels; includes corrected SMTP Low.
- `CDR-MLC.ipynb`: new paper-aligned implementation and three available cross-congestion scenarios.
- `CDR-MLC-New-Version-Feature-Analysis.ipynb`: exploratory feature audit. Its static initial-run notes refer to older data.
- `outputs/` and `analysis_outputs/`: generated locally and ignored by Git.

## Run

Use Python with numpy, pandas, scipy, scikit-learn, matplotlib and Jupyter installed. Open `CDR-MLC.ipynb` and Run All. The notebook locates the dataset relative to the repository or notebook directory and writes reports under `outputs/paper_implementation/`. Git stores the notebook without cell outputs; its initial validation note records the executed results and versions.

The core uses 15 trailing-window timing statistics, training-only scaling and
MiniBatchKMeans with three clusters, and three 20-tree Random Forest experts on
original non-timing features. All clustering-first implementations use the
shared `minibatch_clustering.py` factory (`k-means++`, batch size 1024,
`n_init=10`, `max_iter=100`, `reassignment_ratio=0.01`); full-batch `KMeans`
is not used. Two 100-tree pooled RF references are evaluated on the same test
records. Raw datasets are not modified.

## Cross-level injection pools

[CROSS_LEVEL_INJECTION.md](CROSS_LEVEL_INJECTION.md) documents the leakage-safe
pool builder. With its default settings, each level receives chronological 20%
prefixes from both other levels while the final 20% of every capture remains a
separate immutable test set. Physical congestion labels are retained and the
destination is stored separately in `pool_level`.

```powershell
python CDR_MLC/build_cross_level_injection_pools.py --injection-fraction 0.20 --test-fraction 0.20 --output CDR_MLC/DATASETS/CDR-MLC/Cross_Level_Injection_20
```

## Fidelity and limitations

- Scenarios 1–3: Low → Medium, Low → High, Medium → High.
- Scenarios 4–5 require the independent second-server capture and are not simulated from the first-server dataset.
- The supplied paper does not enumerate all 35 original fields or specify every implementation setting. The notebook explicitly records the available feature schema and assumptions, including a three-record window. This is not a claim of exact numerical reproduction.
- Additional paper baselines, including AF, DFE and CNN, require separate faithful implementations.
- Scaling, imputation, category encoding and clustering are fitted only on training data. Application and congestion labels are evaluation metadata, not prediction inputs.
- Windows stay within externally identified capture sequences. Runtime sequence identifiers must not be inferred from unknown application labels.
- Validation checks batch/streaming equivalence and prediction invariance to hidden test labels.
- Initial execution passed all eight code cells on 106,512 filtered records. CDR-MLC did not outperform the pooled RF references in these three runs. Report results as observed; do not select features to weaken a baseline.
- Timing sensitivity alone is not proof of a causal congestion effect.

## Preserved previous work

The complete pre-cleanup repository is preserved on branch `archive/cdr-mlc-before-restart-20260918`, commit `0a556328237ffa9c42d01d7b49f835a040d02c13`. Old notebooks, datasets, models and intermediate results remain accessible there. No Git history was rewritten.

## Per-application sliding-window comparison (new data)

See [WINDOW_SWEEP.md](WINDOW_SWEEP.md) for the 300-fit study on all 15 `New_Version` captures.
`CDR-MLC-Window-Sweep.ipynb` and `window_sweep.py` compare 1/3/5/10/20/50/100-record windows separately for each application across all three congestion levels.
Two protocols distinguish dependent within-capture temporal validation from a conservative five-tuple-purged sensitivity check. Committed reports include all seeds, contingency tables, hashes and plots; the reserved tail was not scored. This is exploratory clustering analysis, not a change to the paper classifier or proof of independent-capture generalization.

## Adaptive label-conditioned CDR-MLC

[ADAPTIVE_CDR_MLC.md](ADAPTIVE_CDR_MLC.md) documents the research extension in `adaptive_cdr_mlc.py`. It selects three congestion-routing features and a trailing-window size separately for each application using train/validation only. At inference, a preliminary label-probability gate combines the label-specific router/expert banks; true test labels and congestion levels are never routing inputs. The fixed paper implementation remains available for the required baseline comparison.

The adaptive implementation was committed without execution or testing at the user's request. Start with the documented reduced search, then return the generated configuration, trial, metric, prediction and manifest files for analysis.
## Adaptive cross-congestion scenarios

[ADAPTIVE_SCENARIOS.md](ADAPTIVE_SCENARIOS.md) documents `adaptive_cdr_mlc_scenarios.py`, which runs Scenario 1 (Low → Medium), Scenario 2 (Low → High), and Scenario 3 (Medium → High). Feature/window selection is performed separately per application using only a temporal split of the source congestion level; the target level remains untouched until final evaluation.

```bash
python CDR_MLC/adaptive_cdr_mlc_scenarios.py --scenarios 1 2 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 42 --gating soft
```

This runner was committed without execution or testing at the user's request.
## Whole-dataset feature bias and Argus audit

[FEATURE_BIAS_ARGUS_AUDIT.md](FEATURE_BIAS_ARGUS_AUDIT.md) documents `feature_bias_argus_audit.py`. It audits all 15 revised captures for invalid/constant/identifier-like fields, missing and negative values, application dominance, within-application congestion sensitivity, capture fingerprints and near-duplicate numeric features. Label-aware results are review evidence and never automatic feature-removal rules.

```bash
python CDR_MLC/feature_bias_argus_audit.py
```

The audit was committed without execution at the user's request.
## Clean-Valid comparison

[CLEAN_VALID_COMPARISON.md](CLEAN_VALID_COMPARISON.md) documents the two-stage reproducible pipeline. `build_clean_valid.py` creates 15 cleaned captures without modifying the revised source files. `compare_clean_valid.py` then compares fixed CDR-MLC, the primary all-valid-feature RF, an expert-input RF ablation, and Adaptive CDR-MLC in scenarios 1–3 on identical target rows.

```bash
python CDR_MLC/build_clean_valid.py
python CDR_MLC/compare_clean_valid.py --scenarios 1 2 3 --fixed-window 3 --windows 3 10 20 --ranking-top-k 5 --selection-seeds 42 --gating soft
```

Both scripts were committed without execution or testing at the user's request. The comparison now also includes `sensitive_cdr_mlc.py`: source-only temporal feature/window selection, unsupervised severity modulation, soft expert routing, and a global-RF blend. True target congestion levels are never used as weights.
## Oracle routing diagnostic

[ORACLE_ROUTING_SWEEP.md](ORACLE_ROUTING_SWEEP.md) documents `oracle_cdr_mlc_sweep.py`, a deliberately level-leaking upper-bound diagnostic. It trains a Low expert on all Low records and Medium/High experts on their calibration prefixes, then routes each held-out tail to the expert matching its true congestion level. Oracle results are diagnostic only and are never deployable performance.

```bash
python CDR_MLC/oracle_cdr_mlc_sweep.py --fractions 0 0.20 --output CDR_MLC/outputs/oracle_level_experts
```
## Learned CDR-MLC router

[LEARNED_ROUTER.md](LEARNED_ROUTER.md) documents a minimal learned gate that preserves the fixed CDR-MLC MiniBatchKMeans router and three RF experts. Gate targets come from a chronological holdout scored by experts that did not train on those records. The sweep compares actual, learned, oracle, and RF routing at 0% and 20% calibration.

```bash
python CDR_MLC/learned_router_sweep.py --fractions 0 0.20 --scenarios 1 2 3 --window 3 --output CDR_MLC/outputs/learned_router_sweep
```
## Selective correction router

[SELECTIVE_ROUTER.md](SELECTIVE_ROUTER.md) documents the conservative follow-up to the learned gate. MiniBatchKMeans remains the default route; a source-only error detector can override it only when the learned error probability and alternative-expert confidence satisfy validation-selected controls.

```bash
python CDR_MLC/selective_router_sweep.py --fractions 0 0.20 --scenarios 1 2 3 --window 3 --output CDR_MLC/outputs/selective_router_sweep
```
