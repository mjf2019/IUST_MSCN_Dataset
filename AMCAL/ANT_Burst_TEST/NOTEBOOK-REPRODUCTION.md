undefined
Legacy source compatibility: absent reward_generalization_weight means the historical 0.0 (disabled) setting. Explicit nonzero values still fail matching; no weights or reward computations are changed by this fallback.

Selector-only checkpoint selection includes epoch-0 source initialization as a validation candidate. If no fine-tuned snapshot beats that candidate, preserve its policy weights with fresh/empty replay and mark selected_snapshot='source initialization' explicitly; never claim this is an improved trained policy. Selection uses validation utility then value residual, not attack results. This guard does not guarantee target-stream performance.
