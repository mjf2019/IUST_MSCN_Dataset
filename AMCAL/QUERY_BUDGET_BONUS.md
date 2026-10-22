# Isolated PS batch5 query-score experiment

```powershell
.\AMCAL\run_amcal_ps_query_budget.ps1 -Bonus 2 -Seed 42
```

Uses the existing light_ps_paironly_batch5_trainseed42/paironly_batch5 checkpoint,
not synthetic0: those are the results actually uploaded. No CNN/Context/selector
retraining. Independent and continual evaluation use the original levels and IDs.

Continual baseline: after level 5, each domain used only 12/141 labels. At level 10
no greedy query was selected in 709 steps. All 12 queries came from epsilon
exploration. This supports under-querying, but does not prove extra queries help.
The independent level-12 result uses 177/177 labels and reaches 81.72%; this
cross-domain observation alone cannot establish causation.

Opt-in action-score correction:
bonus = strength * clip((cap*(position+1)/N - used)/max(1,.1*cap),0,1).
Add this only to query action scores when legal and budget remains. The target
pace is the existing cap fraction; no query is forced, no cap is exceeded, and
current truth is not used. Raw reward/critic/TD targets stay unchanged. Default
strength=0 is the original behavior. Strength 2 matches the route reward's full
[-1,1] range as a single diagnostic setting, not a test-optimized parameter.

This is an evaluation-only query-score control, not a trained RL fix or evidence
of generalization. Compare accuracy/F1, source retention, BWT, labels, and timing.
A new execution fingerprint separates results. Trace logs record raw scores,
modified selection scores, and applied bonus. A query still predicts through
its original Context route; this experiment does not decouple routing from
querying. The uploaded oracle gaps indicate routing is another possible issue.

Architecture remains Light CNN for PS and original CNN elsewhere. No execution
or numerical tests were run, per user instructions; Python syntax only.

Next controlled test: keep Bonus=1 and increase only the online protected replay
CE coefficient from checkpoint .1 to .2:

```powershell
.\AMCAL\run_amcal_ps_query_budget.ps1 -Bonus 1 -ReplayWeight 0.2 -Seed 42
```

Tag budget_bonus_1_replay_0.2; unchanged checkpoint, batch5, label cap, Context
LR and memory size/sampling. This scales the existing protected replay loss,
not the routing replay loss. Evaluate accuracy and source retention together.
ReplayWeight=-1 (default) omits the override and preserves earlier commands.
Only syntax checked; no execution tests run.

Next experiment: keep Bonus=1, replay .1 and batch5, but halve only online
Context LR from .0005 to .00025. Reuse the same checkpoint, no retraining:

```powershell
.\AMCAL\run_amcal_ps_query_budget.ps1 -Bonus 1 -ReplayWeight 0.1 -ContextLR 0.00025 -Seed 42
```

Tag: budget_bonus_1_replay_0.1_lr_0.00025. The default ContextLR remains .0005,
so previous commands/tags are preserved. No execution tests run.

IAT evaluation on the existing original-CNN batch5 checkpoint is also supported:

```powershell
.\AMCAL\run_amcal_ps_query_budget.ps1 -Dataset IAT -Bonus 1 -ReplayWeight 0.1 -ContextLR 0.00025 -Seed 42
```

Dataset defaults to PS to preserve existing commands. IAT uses
paironly_batch5_trainseed42/paironly_batch5 and its original CNN checkpoint;
no architecture/model training change. Same independent/continual levels
5,7,10,12,15,17,20 and same evaluation tag; no synthetic0 checkpoint is loaded.
No execution tests run.
