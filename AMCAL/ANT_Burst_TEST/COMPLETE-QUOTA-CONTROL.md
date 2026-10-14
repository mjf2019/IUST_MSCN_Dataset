# Complete quota diagnostic

Use --quota-policy complete with evaluate --budget-mode per-class --query-gate none. The default optional policy preserves freely chosen consumption. Complete mode retains RL requests, then forces action=1 only when all remaining rows in an offline class block are needed to fill its remaining quota. Original RL requests and forced choices are recorded separately (requested_action, forced_query), and forced_queries is reported. The effective replay action reflects the actual query.

This uses offline block size metadata, not a deployment policy or unconstrained RL. Classes with fewer rows than quota consume at most their available rows. Total/class caps remain enforced. The model is not reset between classes. Holdout remains excluded, including from remaining-row counts. Pure random already samples exactly min(quota,class size) per class, and complete mode marks/separates these comparison results.

This control tests equal-count Context adaptation, not lower-label superiority. Tail completion may concentrate forced updates late in a block; report forced counts and final heldout results. Syntax and quota completion under all-skip/all-select/mixed requests, zero quotas and short blocks were checked; no full PyTorch evaluation was run locally.
