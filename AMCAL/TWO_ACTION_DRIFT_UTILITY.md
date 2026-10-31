# Two-action utility DDQN with synthetic development drift

This isolated variant addresses the previous utility experiment's development
distribution: a previously trained Context had little reason to adapt on familiar
TRAIN/VAL 0/1/3, and the fixed mixed-domain probe often discouraged useful changes.
The original strong IAT CNN, Light PS CNN and healthy Context checkpoints remain
unchanged. Only the query DDQN is trained; no CNN/initial Context retraining occurs.

## Synthetic TRAIN and VAL

The same sampled known-level 0/1/3 TRAIN rows receive label-blind feature-wise
affine drift plus small Gaussian noise. The scale of each feature is estimated
from the adaptive TRAIN feature values only. Seeds, severities and transformations
are declared in advance, independent of labels, classification performance and
TEST. This is feature stress augmentation, not a claim of physically valid packet
manipulation or a substitute benchmark for official Burst attacks.

Each TRAIN epoch has a new transformation for each known-level domain. Default
severities are 0.75, 1.0 and 1.25 across the three epochs, with scale amplitude
0.3, shift amplitude 1.0 TRAIN standard deviations and noise amplitude 0.03.
The fixed 24-row independent TRAIN probe for that level receives the **same**
episode transformation. Sample IDs and feature duplicates were reserved outside
the union of adaptive TRAIN streams and excluded from TRAIN replay anchors.
Row-ID seeded noise is invariant to sample order or stream/probe batching.

The reward observes the effect of one actual batch5 Context update:

`reward = decrease in macro CE on the matched shifted TRAIN probe
          - lambda * positive increase in macro CE on the clean retention probe
          - request cost`.

The original-domain retention probe stays unshifted and disjoint from updates.
Each episode's gain is measured on its own matching probe, rather than averaging
the gain over three familiar unshifted domains. There is no future rollout,
counterfactual update copy, auxiliary head or sign-only target. Skip reward stays
zero. DDQN bootstrap, features, normalized input encoder, epsilon schedule, batch5,
replay weight 0.1 and Context LR 0.00025 are unchanged. Probe observation is TRAIN
only; it never fits or updates a model on VAL/TEST labels.

VAL uses official VAL partitions from levels 0/1/3, with independent fixed drift
seeds and severities 1.0/1.25/1.5. No TRAIN/VAL IDs are mixed. All query controls
see the exact same shifted VAL rows/order and unshifted source audit. TEST uses
official levels 5/7/10/12/15/17/20 with **no synthetic transformation**. No TEST
feature statistics, labels or metrics choose transformations, costs or epochs.

## Selection guard

The development scenario is declared valid only when front-loaded adaptation
improves mean VAL online AC over NoQuery by at least 0.5 percentage points. A
utility candidate must also improve AC over NoQuery by at least 0.5 points, stay
within 0.5 points of front-loaded AC, and remain within the existing 0.5-point
tolerance on positive original-domain forgetting. Among eligible candidates,
choose the lowest actual label use. There is no forced query floor in the policy.

Zero-request behavior cannot be reported as successful adaptation. If no candidate
qualifies, `validation_goal_met=false` and the best accuracy-minus-positive-
forgetting fallback is explicitly reported. TEST is still evaluated once for
diagnosis; this does not turn a failed VAL scenario into successful selection.
Reduced labeling is called a VAL success only if the eligible candidate actually
uses fewer labels than the front-loaded reference. TEST success remains empirical.

## Run

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_two_action_drift_utility.ps1 -Dataset IAT
.\AMCAL\run_amcal_two_action_drift_utility.ps1 -Dataset PS
```

Run IAT/PS in separate terminals if desired. Default is 300 rows per domain,
three epochs **per candidate**, and a 20% cap reset per episode. Four utility
settings (lambda 0.5/1.0 and cost 0/0.001) plus a matched paper-reward DDQN are
trained on identical synthetic episodes. The original trained Actor, uncertainty,
random, front-loaded and no-query controls are also evaluated on the same official
TEST inputs. Only utility candidates are selected on synthetic VAL. The paper
control receives the same drift and exclusions; it does not load the previous
paper DDQN trained on familiar inputs.

The prior local healthy and parent-Actor model binaries are required. No packages
are installed. Repeating the command resumes DDQN at completed epoch boundaries;
the recipe deterministically restores that epoch's drift/probes. `-Action Evaluate`
with identical settings repeats only selected TEST evaluation. Outputs use
`protocol_runs/two_action_drift_utility_seed42_p300_e3_v1`, separate from every prior
experiment. Code, source-model, data, probe and synthetic-recipe provenance are
recorded. This remains a seed-42 small-data pilot; it does not establish superiority
or full multi-seed benchmark results.

```powershell
.\AMCAL\push_amcal_two_action_drift_utility_results.ps1 -Dataset Both
```

The helper stages only this variant's JSON/CSV and rejects unrelated staged files.
Model binaries are excluded. `Results/comparison_summary.csv` reports all seven
methods. Console summaries include actual label consumption, reward gain/harm/cost,
NoQuery versus FrontLoaded versus selected VAL AC, scenario validity and independent
TEST AC/PR/RC/F1, final target AC and original-domain forgetting.

Closed-loop interaction and competitive-gap Context descent are retained. The
utility query objective does not establish a formal shared min-max objective.
Synthetic feature drift may not represent the official attack shift or produce
beneficial adaptation; the validity guard reports this instead of changing TEST
or forcing labels to manufacture a result.

The wrapper runs existing DDQN/reward tests and new CPU tests covering label-blind
augmentation, TRAIN-only statistics, row-ID deterministic noise, independent VAL,
unmodified TEST, matching episode probes, per-epoch drift refresh, frozen TEST
query weights and rejection of no-query success. No tests, imports, training or
evaluation were run by the assistant while writing this change.
