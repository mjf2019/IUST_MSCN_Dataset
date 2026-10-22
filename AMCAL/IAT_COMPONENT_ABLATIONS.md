# IAT principal component removal

```powershell
.\AMCAL\run_amcal_iat_component_ablations.ps1 -Seed 42
```

Three independent and three continual evaluations reuse the same original-CNN
paironly_batch5 checkpoint, identical source IDs, order seed and audit pools.
This is post-training component removal, NOT separately trained ablation models.
Legacy route tensor shape is retained only for checkpoint compatibility.

- no_context: bypass the Context entirely (no Context forwards/updates), retain
  RL online learning/query policy and Bonus1. Legal choices: frozen prediction
  or frozen prediction plus acquired label. Both predict the same frozen CNN;
  without any adaptable predictor, accuracy must equal Frozen, though label use
  differs. Feedback may train the selector, never CNN. Context utility is zero
  before annotation cost. A dummy duplicate frozen route packs the old state;
  it is not a second predictor or a Context execution.
- no_rl: no selector forward calls or learning; always predict through the
  Context. Fixed seeded uniform-without-replacement query schedule uses the
  exact 20% cap. Context adapts with batch5, LR .00025, protected replay .1.
  This removes both RL sample selection and routing. Observer endpoints also
  use Context only; no labels choose a prediction route.
- frozen: frozen CNN predictions only; no RL forward/learning, no Context
  execution/updates and zero labels. Nominal requested budget is .2 in config,
  effective stream label_cap=0. Retention should be zero on fixed audit rows.

All predictions precede feedback. No audit labels enter updates. Persistent
runs retain adaptive state only where it exists. Per-domain uniform budgets
in no_rl restart at each domain, matching the full method's recharge protocol.
The no_rl schedule consumes the full cap, while the full method may use less;
report actual label counts rather than claiming exact annotation equality.

Results get component_no_context / component_no_rl / component_frozen tags,
architecture_ablation metadata and distinct fingerprints under the original
IAT batch5 Results directory. Baseline is the previously completed Bonus1,
replay .1, LR .00025, budget .2 run. No baseline or model file is overwritten.
No numerical/execution tests were run, as requested; Python syntax only.
