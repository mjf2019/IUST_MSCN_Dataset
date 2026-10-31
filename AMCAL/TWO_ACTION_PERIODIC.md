# Periodic feedback controls at 20% budget

This evaluates two deterministic replacements for the query decision, with the
same strong healthy Context and classifier as the paper DDQN comparison:

- `PeriodicSkip1`: query rows 1, 3, 5, ... (query one, skip one).
- `PeriodicSkip2`: query rows 1, 4, 7, ... (query one, skip two).

Positions refer to arrival order in the same seeded shuffled TEST stream. Both
start at the first input and stop requesting after `floor(0.2*N)` labels. At
N=300 this is 60 labels: last query at row 119 or 178, respectively. The budget
is 20% per attack intensity, not a total shared across intensities.

The initial pretrained Context is retained. Prediction is always Context -> Frozen
CNN before the current label. A requested label triggers the unchanged batch5
Context update with source replay weight 0.1 and Context LR 0.00025. There is no
Actor/Q forward, uncertainty threshold, reward-based selection, TD update or
retraining of CNN/initial Context. The existing choose() environment interface is
used through an explicit schedule adapter; its `mode='DDQN'` plumbing does not call
a DDQN model. Saved metadata explicitly marks query learning disabled. Paper-table
rewards in traces are observer diagnostics only; the operative reward field is zero.

This is an inference-time query-selection control, not an independently retrained
no-RL ablation. Original IAT CNN and Light PS CNN, preprocessing, test rows/order,
source audit and healthy initial Context match the preceding comparison. Every
control/intensity resets Context, Adam, source/paid memory and the label cap.
No adaptive state carries over to the next case.

## Run both controls on each dataset

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_periodic.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_periodic.ps1 -Dataset PS
```

Each command evaluates both controls on official TEST levels 5/7/10/12/15/17/20,
with 300 rows per level and seed 42. It prints dataset, method, pattern, budget,
level, actual labels, last query position, online AC/PR/RC/F1, final target AC/F1
and original-domain AC forgetting. It requires the same existing local healthy
and parent-Actor checkpoint files for source loading/provenance; the Actor is
not used to make decisions. No DDQN checkpoint is needed by these schedule-only
controls. No package is installed.

For separate single-control runs:

```powershell
.\AMCAL\run_amcal_two_action_periodic.ps1 -Dataset IAT -SkipCounts 1
.\AMCAL\run_amcal_two_action_periodic.ps1 -Dataset IAT -SkipCounts 2
```

Use `-PilotRows 0` for full official TEST. Default output folders include dataset,
seed, row count and requested skip counts, e.g.
`protocol_runs/periodic_query_seed42_p300_skip1_2_b20_v1`.
`summary_by_level.csv` and `summary_mean.csv` report results per level and averaged
across seven levels. Full causal traces and checksummed completion records are
stored for each control/level. Repeating the same command reuses verified completed
cases. Existing experiments and source checkpoints are never overwritten.

```powershell
.\AMCAL\push_amcal_two_action_periodic_results.ps1 -Dataset Both
```

The push helper stages only the new periodic experiment JSON/CSV files, excluding
models and rejecting unrelated staged files. CPU tests cover both exact masks,
start/end positions, hard caps, real Context updates, absent query training,
independent resets, console labeling, cache reuse and corruption detection.
No tests, imports, training or evaluation were executed by the assistant while
preparing these files.
