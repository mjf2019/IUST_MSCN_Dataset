# Selected-beta TEST budget sweep

Evaluation only. Existing composite bundles are required; no CNN, Context pretraining,
Actor training or DDQN training is run. During TEST only Context adapts, using
the same batch5 / LR 0.00025 / replay 0.1 update as the saved comparison.

Defaults: IAT beta 26, PS beta 55, Current bundle, seed 42, 300 TEST rows per
level, levels 5/7/10/12/15/17/20, budgets 30/40/50 percent. Each call to the
original episode runner creates fresh Context/Adam/memory and a fresh budget.
The CNN and query policy remain fixed. The same seed and TEST rows are used
at every budget. Actual label use can be below the cap.

Results are saved separately under the existing post_selector_cnn13 run:
Results/testp300_budget30, Results/testp300_budget40, Results/testp300_budget50.
Original training protocols/checkpoints and 20-percent comparison are untouched.
Cases resume only after execution/metric/trace checksum validation.

Commands (PowerShell, repository root):

    git pull origin experiment/paironly-paid-replay
    .\AMCAL\run_amcal_selected_beta_budgets.ps1 -Dataset IAT -Beta 26
    .\AMCAL\run_amcal_selected_beta_budgets.ps1 -Dataset PS -Beta 55

To include both the original CNN and the fine-tuned CNN:

    .\AMCAL\run_amcal_selected_beta_budgets.ps1 -Dataset PS -Beta 55 -Variants Current,CNN13

Use -TestRows 0 to evaluate all eligible TEST rows per level. The default 300
matches the preceding pilot and avoids changing the comparison sample size.

Push JSON/CSV results (existing helper; model binaries excluded):

    .\AMCAL\push_amcal_post_selector_cnn13_results.ps1 -Dataset Both

The wrapper runs the accompanying small mocked budget-contract tests before
evaluation. Tests, training and inference were not executed by the authoring assistant.

