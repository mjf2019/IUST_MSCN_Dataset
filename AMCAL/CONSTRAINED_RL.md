# Constrained sequential RL pilot

This is an opt-in experiment. Original PairOnly, minimax-query, budget-Actor,
and manuscript files are untouched. It is not a validated replacement method.

## Objective and min-max meaning

Let b be acquired labels / scored rows, A pre-feedback online accuracy, M online
macro F1 over all configured classes, and D source Context accuracy before minus
after adaptation. The goal is to minimize E[b] subject to lower bounds on A/M
and an upper bound on D. The Lagrangian is

`min_policy max_lambda>=0 E[b + lambda_A*(A_min-A) + lambda_M*(M_min-M) + lambda_D*(D-D_max)]`.

This min-max is explicitly between the policy and constraint multipliers, not
the former selector-versus-Context adversarial game. Context continues to
minimize its existing paid batch5 CE-gap plus protected replay objective.
No zero-sum equivalence to the previous formulation is claimed. Neural DDQN,
partial state summaries and projected dual updates are an approximate solver;
the formulation does not imply global convergence or constraint satisfaction.

There is a hard 20% per-episode/intensity query cap and no exact-spending quota,
forced tail, first-20%-window restriction, or additive query bonus. A policy may
request fewer or no labels. Infeasibility is recorded rather than hidden.

Matched FrontLoadedContext runs provide the quality reference. Default lower
bounds are reference A/M minus 1 percentage point; the forgetting upper bound
is reference D plus 0.5 point. These tolerances are fixed before the pilot.
TRAIN references are recomputed on exactly the same sampled rows and ordering
as each TRAIN policy episode. Dual ascent uses TRAIN violations only, lr=0.5,
initial multipliers [5,5,5], projected to [0,100]. Projection is numerical
stabilization, not a hard quality guarantee. VAL never updates the multipliers.

## Preserved classifier and adaptation

The full healthy Natural checkpoint initializes every episode independently:
`minimax_query_batch5_seed42_v2_h16_c0.05_actorstd_natural` in each dataset root.
CNN and preprocessing provenance are checked. IAT requires the original CNN;
PS requires the existing tagged Light CNN. No CNN or initial Context is retrained.
Each episode copies the healthy Context, creates fresh Adam/history/feedback
memory, and keeps CNN/Context in eval mode. Context updates use only the current
paid label, up to four previous paid rows, and the same protected replay update
with the healthy checkpoint's learning rate and replay weight. Frozen CNN
parameters never require gradients. Ordinary update sampling does not see or
exclude a future block; the previous one-update oracle machinery is unused.

## Policy and information flow

The policy is a small three-action DDQN: (24+2*K) inputs, 32 GELU hidden units,
three Q values, target network and TD replay. For six classes this is 36 inputs
and 1,283 trainable parameters. Adam policy lr is 0.001; target synchronization
is every 20 optimizer steps. No learned input encoder is added.

State includes both class-probability vectors, confidence, margins, entropies,
JS disagreement, modulation size, distance from TRAIN source embedding statistics,
distance from recent paid embeddings, paid-label coverage/support, reliability,
recent update size, elapsed time, an absolute-cap encoding and remaining budget.
The absolute-cap feature distinguishes short and long episode horizons.
The 128-wide raw CNN
embedding is not duplicated into the state. Distances use Frozen embeddings;
source means/scales are fit on TRAIN anchors only. Only actually paid labels
enter history. This is a compact partial-observation summary, not proof of a
fully observed Markov state.

Actions 0/1 predict through Frozen/Context without updating. Action 2 retains
the original three-action meaning: Context prediction, then label acquisition
and a Context update. Its prediction route is fixed rather than changing with
the Q0/Q1 parameters; this keeps a fixed action definition for TD replay. Q2 is
learned directly from sequential returns, not manually tied to Q1 plus a
one-update score. All rows, including queried rows, are scored before their own
feedback. Context changes affect the next actual stream state. Budget exhaustion
masks action 2. Label-free routing and query controls use the same cap and rows.

Initialization imitates the uncertainty controller for 100 small TRAIN-only
gradient steps, including a route preference loss. Demonstration states are
collected from three temporary TRAIN episodes. The q75 entropy threshold is
calibrated from unlabelled VAL 0/1/3 features, as a fixed calibration setting;
no VAL truth is used for imitation. Initialization is not called RL or min-max.
RL then uses real multi-update trajectories, not fixed one-update utility labels.

The dense reward components are
`[-queried/N, correct/N, change_in_macro_F1, terminal_negative_source_forgetting]`.
For current multipliers their weighted sum over an episode is exactly
`-b + lambda_A*A + lambda_M*M - lambda_D*D`, i.e. negative Lagrangian up to
policy-independent constraint constants. Gamma=1 in the finite horizon. The
buffer stores component rewards; sampling reweights them using current duals,
avoiding stale scalar rewards after multiplier changes. One-step Double DQN
selects the legal next action with the online network and values it with the
target network. Terminal states do not bootstrap. No duplicate future-gain term
is added. Epsilon decays from 0.30 to 0.05 across TRAIN episodes.

All TRAIN truth may be used by the *observer* for rewards/confusion counts,
including unqueried rows, but unqueried truth never enters the adaptation memory
or policy state. TD learning occurs after the episode. During VAL/TEST, policy,
target and multipliers are fixed; only Context adapts to acquired feedback.

## Splits, audits and scope

TRAIN and VAL levels are 0/1/3; TEST 5/7/10/12/15/17/20 is loaded only with an
explicit Evaluate/All action. Existing artifact split identities are retained.
The default pilot uses up to 300 rows per domain and 12 TRAIN episodes, cycling
levels and using different seeded subsets/orderings. The first cycle (three
episodes) uses up to 300 rows to expose the audit-matched horizon; the remaining
nine use up to 96 rows. Within the TRAIN pools, a fixed 50% class-covered hard
curriculum uses higher initial healthy-Context CE losses, with the rest sampled
uniformly. HardFraction=0 disables hardness preference while retaining class
coverage. These are real TRAIN rows, not invented features or attack labels.
VAL/TEST receive no hardness selection. TRAIN truth for curriculum generation
is offline information and never enters the action state. Repeated
rows across episodes create new adaptive histories, not independent new flows.

TRAIN anchors exclude the healthy checkpoint's held-out source IDs/features;
a fixed stratified subset of up to 128 held-out TRAIN reference rows measures D.
For VAL/TEST, reserve a stratified source-audit subset (10% of eligible original
domain rows); audit features must be disjoint from TRAIN replay anchors. Audit
IDs and feature duplicates are removed from the target streams at every level.
Their labels are never adaptation inputs. Each pilot subset is sampled after
this separation. All compared policies use exactly the same rows and ordering.
Streams are fully shuffled; within-class temporal order is not preserved.

This independent-audit partition changes the scored population relative to
earlier experiments. Compare the matched controls in this run; do not attribute
differences from previous full-data tables to the policy alone. D uses the same
fixed Context route for every policy, preventing routing from hiding parameter
forgetting. Source initial/final accuracy is always reported. Routed-system
final source AC is also measured separately without requests or updates; it is
not substituted for Context D in the constraint. D is not a claim about routed
system-level forgetting, multi-domain BWT or final-target
heldout generalization. The pilot's primary target metrics are pre-feedback
online AC/PR/RC/F1.

## Selection, controls and resumability

Every four episodes (and at the last), VAL checks each level against its matched
FrontLoadedContext reference. A candidate is feasible only if every checked
quality constraint passes. Among feasible learned policies choose minimum mean
label fraction, then maximum online AC. If none is feasible, choose the smallest
worst violation and mark the checkpoint infeasible. Only RL-stage checkpoints
are selected; no baseline is silently substituted as a learned policy.

Controls are FrontLoadedContext, Uncertainty (both query and route), Random
(both query and route), and NoQueryContext. Each TRAIN episode also records
FrontLoadedContext minus NoQueryContext AC/F1 to show whether adaptation is
actually useful on that episode. A teacher already perfect on TRAIN need not
benefit from queries; the tool does not hide or force this headroom.
Controls share the healthy initial Context, update function,
independent audits and cap. These are matched *deployment controls*, not a claim
of complete no-RL-from-scratch Context training ablations. No control is weakened
to manufacture RL superiority. Report actual consumption as well as the cap.

Episode-boundary trainer checkpoints include policy/target/Adam, TD replay and
its sampler RNG, multipliers, logs and selection state. Repeating the same command
resumes from the last completed episode. An interrupted episode is repeated.
Protocol mismatches require a new RunTag and preserve existing results. Completed
runs display saved summaries. JSON/CSV outputs are pushed; model/trainer binaries
remain local. Console reports each episode, all VAL metrics, multiplier values,
worst constraint violation and a cross-method mean table. CUDA work is synchronized
for the reported amortized benchmark row time. Batched Frozen-cache computation,
sequential Context/policy work and observer scoring are included; checkpoint
loading and source audit passes are excluded. This pilot timing is not a claim
about real-time per-packet latency or production throughput.

```powershell
git pull origin experiment/paironly-paid-replay
.\AMCAL\run_amcal_constrained_rl.ps1 -Dataset IAT -Action Train
.\AMCAL\run_amcal_constrained_rl.ps1 -Dataset PS -Action Train
.\AMCAL\push_amcal_constrained_rl.ps1 -Dataset Both
```

Default output: `constrained_rl_seed42_p300_e12_n96_pilot_v1` under each dataset's
protocol_runs. No packages are installed. Real data/model experiments are run
by the user, not by the development tests. A validated quality improvement is
required before recommending full TEST or manuscript changes.
