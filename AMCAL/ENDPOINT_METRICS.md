# Four-metric Burst endpoint audits

`run_burst_endpoint_metrics.py` runs one method and dataset, using the five evaluation seeds 42,48,54,62,72 and seven heldout levels 5,7,10,12,15,17,20. It collects a separate `paper_endpoint_metrics/METHOD` JSON bundle for that method. No package is installed. Existing offline models are reused for online methods; no CNN, Context, RL, RF, ARF, CALMID, MicFoal or DFE offline retraining is performed.

For each test level the original model and feedback state are reset; labels are capped at floor(0.2*N). RAL uses the existing recent-five-feedback variant. IAT uses the existing original-CNN AMCAL checkpoint and PS the existing Light-CNN AMCAL checkpoint. AMCAL uses one Context, three actions, batch5 paid updates, bonus1, protected replay weight .1 and online Context LR .00025. Training intensities remain 0,1,3.

Only evaluation output is augmented. Endpoint classification runs outside timed online adaptation, does not feed its labels into learning and does not choose policies, query schedules or hyperparameters. Existing result paths are preserved by distinct endpoint tags. Independent methods have distinct result tags. Audit reruns skip seed/level pairs with complete four-metric endpoint output, allowing continuation after an interrupted process.

## Quantities

At each endpoint, accuracy, macro precision, macro recall and macro F1 are computed from the full confusion matrix, using the fixed six-class source vocabulary and zero for undefined per-class precision/recall/F1.

- Target gain, in percentage points: final minus initial metric on exactly the same unqueried target TEST rows. The final query mask is applied retrospectively to both endpoints; it does not select training samples or change the query policy.
- Reference drop (net forgetting), in percentage points: initial minus final metric on the fixed original-domain level-0 VAL reference. Negative drop means improved reference performance. These are endpoint net changes, not a peak-to-final or multi-domain BWT definition.
- Online scores remain pre-feedback metrics recomputed from prediction traces.

Each paired output contains both confusion matrices and every gain/drop value; accuracy-only summaries are retained for compatibility. The existing classification/adaptation sequence remains unchanged. The collector compares rerun online accuracy/F1/label count to the previous saved evaluation and reports any discrepancy rather than silently mixing trajectories.

## AF boundary: stage audit, not whole-method forgetting

AF is a separately calibrated offline few-shot method with no trained source-only predictor before its DANN stage in the existing author-structure implementation. Inventing such a predictor would change the experimental protocol. Whole-method pre-source forgetting is therefore still not defined by the current AF protocol.

The optional AF audit observes the **end of DANN, before feature-layer reset and supervised target tuning**, then the final model. Both endpoints use the same target-only cosine/distance kNN rule (k=5) and the same five labelled calibration examples per class. Original-domain VAL rows and target TEST rows are observers only. This measures the effect of feature reset plus supervised tuning, **not the complete AF adaptation procedure**. Its phase is recorded in every JSON bundle and must be footnoted explicitly if placed beside online-method endpoint changes. Do not label these AF values as whole-method source-only forgetting.

The callback adds no optimizer step, trainable module, training data or RNG sampling. It restores extractor train/eval mode afterwards. AF fitting/calibration must be repeated to capture the missing intermediate state; a final checkpoint alone cannot reconstruct it. The final evaluation uses the unchanged author-structure method. No additional source-only training procedure is introduced.

## Run one method

From implementation repository root:

```powershell
python AMCAL/run_burst_endpoint_metrics.py --dataset IAT --method AMCAL
python AMCAL/run_burst_endpoint_metrics.py --dataset PS --method AMCAL
.\AMCAL\push_burst_endpoint_results.ps1 -Method AMCAL
```

Replace AMCAL with CNN, CALMID, MICFOAL, RAL, AF, DFE or ARF. Each command defaults to all five seeds and seven levels. `--seeds 42` or `--levels 5` can be used for a pilot, but the push helper expects complete 35-pair coverage for both datasets. Use `--device cpu` if needed. CPU-only tree methods are explicitly run on CPU regardless of the wrapper's device option.

`--action collect` rebuilds the evidence bundle from completed endpoint runs without another evaluation. Re-running the normal command resumes missing levels. Both datasets' matching method folders are the only paths committed by the push helper; model binaries and unrelated staged changes are not included in that commit.

Online execution is repeated, not inferred from historical accuracy totals. It can take time, especially for RAL refits. For timing comparisons run methods sequentially; concurrent execution shares resources.

## Validation during preparation

No model training or adaptation experiment was run and no dependencies were installed. Python syntax, CLI help and focused confusion-matrix arithmetic/serialization checks were performed. The user executes the actual model experiments in the existing Windows environment.

## Known-source AF update

The endpoint runner now explicitly supplies `--source-levels 0 1 3` to AF. It retains the author-style source budget of 25 examples per class **in total**, assigning 9 to level0 and 8 each to levels1 and3. Underlying flow IDs are distinct across these source supports. Only TRAIN partitions enter this selection, and source flow IDs are excluded from target calibration even at another intensity. The 5-shot-per-class target calibration remains an offline AF step at each heldout intensity; target TEST rows never enter fitting. Thus this is not source-only zero-shot AF and is not a 20%-budget prequential method.

The original standalone AF default (`--source-levels 0`) is preserved for reproducibility. New endpoint AF runs record their source levels, per-class counts, per-row source intensities and all source-domain hashes. The endpoint collector rejects previous AF audit runs lacking explicit source levels 0,1,3. The reference observer remains original-domain level0 VAL, as for other methods; known training levels and the forgetting reference are distinct concepts.
