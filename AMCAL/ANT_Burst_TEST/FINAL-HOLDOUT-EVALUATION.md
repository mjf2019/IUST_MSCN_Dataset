# Final reserved holdout evaluation

Enable with evaluate --holdout-fraction 0.2. Default 0 preserves prior protocol. Select the same first max_samples original rows, then reserve the last ceil(fraction * class size) rows within each class (at least one, leaving at least one for adaptation). Both subsets preserve original per-class order. Membership/source row IDs are recorded. All checkpoint classes must exist in the reserved set, otherwise evaluation fails.

Reserved rows are excluded from RL state/action/replay/reward, online Context/DQN updates, query quotas, and adaptation accuracy curves. The adaptation subset alone uses the existing budget and per-class caps. Checkpoint/model/optimizer state is retained between adaptation classes. One final frozen Context evaluation is run after the entire adaptation stream for each perturbation level; no model reload, training or optimizer calls occur during holdout inference. Labels are used only by offline splitting and scoring.

With holdout enabled, scaler and class-weight counts are derived exclusively from oversampled_train_dataset.csv for both learned and random evaluation. Prior target-PER0 preprocessing could include reserved rows; using training-only preprocessing prevents that. This is a protocol change relative to old results and is identical for both controls.

Final results include heldout accuracy, macro/weighted F1, all-class precision/recall/F1 and confusion matrix, plus frozen CNN accuracy on the same holdout. Output files: *_heldout_results.json and *_heldout_predictions.csv. Combined tables include Heldout Accuracy and Heldout Macro F1. Results have _holdout0.2 suffix. Reserved evaluation consumes zero query budget.

These are unseen rows with respect to ONLINE adaptation. Disjoint source-row membership does not establish independence from original offline training data or duplicate flows in the dataset; that requires dataset provenance/deduplication checks. Initial checkpoints remain reused. No model selection/early stopping is performed on heldout scores.

Validation: source compile; exact disjoint complete partition; all-class coverage of both subsets; FIFO preservation; deterministic matching split; zero/invalid fraction and singleton checks; no optimizer/backward call in holdout evaluator. On the uploaded 1073-row sequence, 20% reserves 217 rows (class counts 21,16,57,44,16,63), leaving 856 adaptation rows. Real PyTorch inference was not run locally.
