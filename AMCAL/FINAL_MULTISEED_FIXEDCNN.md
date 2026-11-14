# Final AMCAL: independent adaptive training with a fixed CNN

This is the selected two-action, fixed Context -> Frozen, learned-beta DDQN
family. IAT uses the original CNN; PS uses Light CNN. Budget is20% per TEST
intensity5/7/10/12/15/17/20; all eligible rows are evaluated, never a300-row TEST.

Default training seeds:42,48,54,62,72. Each independently refits Context
development, the compact paper-reward warmup, fixed-beta DDQN warm start and
meta-learned beta. The classifier/scaler AND pretrained Context/router starting
checkpoint are shared from the selected initialization. No fitted adaptive
checkpoint from seed42 is copied to the other training seeds. This measures
adaptive-training variation conditional on common pretraining, not independent
CNN pretraining or a new all-stage-from-scratch algorithm.

TEST IDs, original observer, source replay pool and shuffle/sampler seed42 are
fixed across the newly trained models. The old seed42 setup is loaded only to
reconstruct that data protocol; its scores/policy never select new models.
Q and beta are frozen in TEST. Context/Adam/memory/budget reset per intensity.

Run from the implementation repository:
    .\AMCAL\run_amcal_final_multiseed.ps1 -Dataset IAT -Action All
    .\AMCAL\run_amcal_final_multiseed.ps1 -Dataset PS -Action All

Train and Evaluate can be invoked separately with the same seeds/output root.
All stages log dataset, training seed and stage. Interrupted nonresumable
Context/cache stages are renamed .partial_TIMESTAMP, preserving their files.
Resumable selector/DDQN/meta stages use their existing checkpoint logic.

Outputs: protocol_runs/final_multiseed_fixedcnn_v1 in each dataset.
Results/summary_by_level.csv has mean and sample SD across training seeds.
Results/summary_overall.csv first averages seven levels within each training
seed, then computes between-training-seed mean/SD; it never pools35 cases as
independent repeats. Actual labels and signed original-domain AC forgetting
are reported; negative forgetting means an improvement.

    .\AMCAL\push_amcal_final_multiseed_results.ps1 -Dataset IAT
    .\AMCAL\push_amcal_final_multiseed_results.ps1 -Dataset PS

The wrapper runs only the pure-stdlib coordinator contract tests before model
work. No experiment, model test, Python import, training or inference was run
by the code-editing agent; execution is delegated to the user.
