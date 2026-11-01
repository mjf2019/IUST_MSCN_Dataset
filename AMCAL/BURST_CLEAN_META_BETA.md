# Clean-only learned-beta DDQN: Burst IAT and PS

This is a separate experiment. Existing Burst DDQN/beta checkpoints were trained
using levels 0/1/3 and **must not be relabeled as clean-only**.

- Both IAT and PS use the approved **Light CNN**.
- Import only the existing source preprocessing's official partition IDs, class
  encoder and feature schema. Do not import any CNN, Context, Actor, Q or beta
  weights. Refit scaling using raw level-0 TRAIN features only.
- Train a fresh Light CNN (up to 60 epochs) and fresh Context (up to 20 epochs),
  both using clean level 0 and early stopping on clean VAL only. The initial
  Context learning rate is 0.00025. The CNN retains the existing Light CNN
  supervised optimizer/regularization. Historical weights remain untouched.
- Then use the same clean-only warmup (3 epochs) and learned-beta DDQN training
  (3 epochs) as `amcal_clean_meta_beta.py`. This preserves its signed gain,
  meta objective, competitive Context update, batch5 feedback replay, source
  replay weight 0.1 and Context learning rate 0.00025.
- Gain/meta/reference reserves use only clean TRAIN. All remaining clean TRAIN
  rows are used, with no main-TRAIN pilot cap. Probe independence is relative to
  current adaptive updates, not the preceding supervised source pretraining.
- Q/beta checkpoint selection uses clean VAL only. No perturbed CSV is read
  during a Train-only invocation. Listing target filenames does not load them.
- At TEST, use every supplied intensity file, including **0,1,3,5,7,10,12,15,17,20**
  and any additional matching official intensity files. No 300-row TEST cap.
- Each level starts with the clean-only initial Context, a fresh Adam optimizer,
  fresh feedback memory and its own 20% budget. CNN, Q and beta remain fixed.
  Truth is observed only after the current prediction and request decision.

The old `light_iat_paironly_batch5_trainseed42/source` and
`light_ps_paironly_batch5_trainseed42/source` runs are **partition templates
only**. Their attack-trained weights are not loaded. Set `-SourceTemplate` if
the partition template has a different local path.

Default output per dataset:
`protocol_runs/clean0_light_meta_beta_ddqn_seed42_v1`.
The source stage resumes completed CNN training, and repeats an interrupted
Context fit from its deterministic clean initialization. DDQN warmup/meta stages
resume the last completed epoch. Previously completed evaluation cases are
verified and skipped. Nothing is deleted.

`-Protocol Both` is the default and produces distinct reports:

1. **Online**: all original TEST rows, pre-feedback scoring as in prior methods.
2. **Independent**: approximately 20% held-out TEST groups; adapt on the remaining
   approximately 80%, with a 20% budget **of that adaptive stream**. Final target
   scores use only the independent holdout. Both labeling denominators are saved.

Online and independent reports must not be averaged together. Original-domain
forgetting is before minus after on clean VAL, matching the other supplied-data
experiments. All means are reported separately for all levels and perturbed
levels only. The console reports AC/PR/RC/F1, N, labels/cap, budget use and
forgetting; independent final accuracy is printed only for that protocol.

## User-run commands

```powershell
git pull origin experiment/paironly-paid-replay
.\venv\Scripts\python.exe -m unittest discover -s AMCAL/tests -p test_burst_clean_meta_beta.py -v
```

Separate terminals (each command trains and tests one dataset):

```powershell
.\AMCAL\run_amcal_burst_clean_meta_beta.ps1 -Dataset IAT -Action All
```

```powershell
.\AMCAL\run_amcal_burst_clean_meta_beta.ps1 -Dataset PS -Action All
```

To separate stages, use `-Action Train`, then the same command with
`-Action Evaluate`. Keep epoch parameters identical when resuming/evaluating.
To run just the comparable full online protocol, add `-Protocol Online`.

```powershell
.\AMCAL\push_burst_clean_meta_beta_results.ps1 -Dataset Both
```

Only this experiment's JSON/CSV results are staged; model binaries are excluded.
