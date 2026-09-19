# CDR-MLC — paper implementation

## Active work

- `DATASETS/CDR-MLC/New_Version/`: 15 Argus CSV exports, five applications × three congestion levels; includes corrected SMTP Low.
- `CDR-MLC.ipynb`: new paper-aligned implementation and three available cross-congestion scenarios.
- `CDR-MLC-New-Version-Feature-Analysis.ipynb`: exploratory feature audit. Its static initial-run notes refer to older data.
- `outputs/` and `analysis_outputs/`: generated locally and ignored by Git.

## Run

Use Python with numpy, pandas, scipy, scikit-learn, matplotlib and Jupyter installed. Open `CDR-MLC.ipynb` and Run All. The notebook locates the dataset relative to the repository or notebook directory and writes reports under `outputs/paper_implementation/`. Git stores the notebook without cell outputs; its initial validation note records the executed results and versions.

The core uses 15 trailing-window timing statistics, training-only scaling and MiniBatchKMeans with three clusters, and three 20-tree Random Forest experts on original non-timing features. Two 100-tree pooled RF references are evaluated on the same test records. Raw datasets are not modified.

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
