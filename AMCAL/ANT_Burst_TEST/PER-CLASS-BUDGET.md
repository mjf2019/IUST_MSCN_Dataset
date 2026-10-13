# Per-class sequential budget experiment

Use evaluate --budget-mode per-class. This forces original input order (class blocks), loads the checkpoint once per perturbation level, and retains Context/RL/optimizer/replay state between classes. It does not reset or retrain between blocks.

The configured total budget is split across checkpoint class IDs: divmod(B,C); the first remainder class IDs receive one extra label. For B=214,C=6 quotas are 36,36,36,36,35,35. Each label updates one class counter and the global counter. Neither cap may be exceeded. Unused quotas are not transferred. Default total mode remains available.

This is an offline class-block evaluation protocol. Block identity is derived from the evaluation labels during preparation and is available only to the quota controller/scoring, never RL state or model prediction inputs. Therefore this is not a deployment protocol with unknown class identity. Random and RL receive the same external class-block quotas.

Pure random --selector-mode random --query-gate none samples min(quota, class size) distinct rows within each class without replacement, using only the offline block definition and seed. It loads only the same CNN/Context checkpoints; its Context objective, online LR and order match learned evaluation. Omit --selection-cost when invoking random because it computes no rewards.

Per-class CSV results include sample count, actual labels consumed, allocated quota and pre-update accuracy. Class curve CSV contains each observed sample's cumulative class accuracy versus consumed labels. All predictions are scored before their current sample's update; post-update whole-stream scores remain separately available. Quota allocation is independent of prediction accuracy. Consuming fewer labels is not an improvement unless accuracy is comparable/better.

Results use _budget_per_class suffix. Existing models are reusable and existing result folders are retained. Tests cover quota conservation, integer remainder allocation, zero/tiny/214 budgets, hard class caps, no quota borrowing and pre-update curve arithmetic. Real PyTorch evaluation was not run locally.
