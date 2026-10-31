# Full TEST fixed-beta sweep

Two datasets: IAT and PS. Betas10/15/20/25/30; seed42; feedback budget20%
per level; TEST levels5/7/10/12/15/17/20. Every eligible official TEST row is
used, retaining the original source-audit exclusions. No TEST subsampling flag
is exposed. The original CNN and initial Context are reused, without CNN tuning.
Context/Adam/memory/budget reset per beta and level. The Q policy is frozen on TEST.

Existing candidate checkpoints are validated and reused. This runner performs
evaluation only, and stops on any missing checkpoint without starting training.
Beta5 is excluded.
TEST does not select or automatically adopt a beta.

The full TEST size changes; TRAIN sizes and all original update settings do not.
Provenance/checksum checks prevent mismatched source models and stale results.
Original candidate checkpoints and earlier results remain unchanged.

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_fixed_beta_fulltest.ps1 -Dataset IAT
    .\AMCAL\run_amcal_fixed_beta_fulltest.ps1 -Dataset PS
    .\AMCAL\push_amcal_fixed_beta_fulltest_results.ps1 -Dataset Both

Each dataset has35 TEST cases. The console and CSV report AC, PR, RC, F1,
final target AC, source forgetting, labeled-stream percentage and cap utilization.
Outputs: protocol_runs/fixed_beta_fulltest_seed42_v1.
The wrapper executes small candidate/provenance tests on the user's machine.
No tests, imports, training or inference were executed by the authoring assistant.

