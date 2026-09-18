# CDR-MLC — clean restart

## Active work

- `DATASETS/CDR-MLC/New_Version/`: 15 Argus CSV exports, five applications × three congestion levels. Includes corrected SMTP Low.
- `CDR-MLC-New-Version-Feature-Analysis.ipynb`: exploratory feature audit. Run All to regenerate local reports; static initial-run notes refer to older data and are not current results.
- `analysis_outputs/`: generated locally and ignored by Git.

## Preserved previous work

The complete pre-cleanup repository is preserved on branch `archive/cdr-mlc-before-restart-20260918`, commit `0a556328237ffa9c42d01d7b49f835a040d02c13`. Old notebooks, datasets (including ISCX), models and intermediate results remain accessible there. No Git history was rewritten. This cleanup reduces the active folder, not historical repository size.

## Implementation sequence

1. Read the submitted paper and record its exact feature definitions, window construction, clustering, classifiers, hyperparameters and evaluation scenarios. Mark ambiguities explicitly.
2. Validate service endpoints and schemas before labeling records. Keep source data immutable and report exclusions.
3. Define train/validation/test sequence boundaries before preprocessing or overlapping windows. Fit scaling, feature selection and congestion clustering on training data only.
4. Implement the paper from scratch in `CDR-MLC.ipynb` with explicit, documented corrections for leakage. Keep exploratory variants outside the primary implementation.
5. Evaluate identical splits/features for comparisons and report both routing and classification results. Never select features to weaken a baseline.

The new paper implementation is not yet included. File-level congestion labels and application labels are evaluation metadata, not classifier inputs. Timing sensitivity is not proof of causal congestion effects or traffic-class separability.
