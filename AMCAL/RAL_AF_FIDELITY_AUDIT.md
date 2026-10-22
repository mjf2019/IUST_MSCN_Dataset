# RAL / Adaptive Fingerprinting fidelity audit

Audit target: `amcal_burst_baselines.py`, branch `AMCAL_final_enhancement`.
The pre-audit implementation is at `6b64df865bf68503ac20a288f86044738d48ed12`.
No old result is deleted, relabelled in place, or used to tune the corrections.
Strong baseline accuracy is not evidence of an implementation error. Neither
method is deliberately weakened to improve AMCAL's relative ranking.

## Primary references

- AMCAL's original `elsarticle/Ref.bib` cites RAL as Wassermann et al.,
  *Adaptive and Reinforcement Learning Approaches for Online Network Monitoring
  and Analysis*, TNSM 2021, DOI `10.1109/TNSM.2020.3037486`.
  The author's [publication page](https://tcuvelier.be/publications) links the
  manuscript. Its HAL download was access-blocked during this audit; this is
  not a claim of independently checking every line of that journal PDF.
- RAL's openly available precursor, [IAL 2019 paper](https://ceur-ws.org/Vol-2444/ialatecml_paper3.pdf),
  Algorithm 1 and sections 5.1–5.2; and [author Python implementation](https://github.com/SAWassermann/RAL/blob/2a34a4b1b1e11054a718233e19de0c2134fd68e8/ral.py).
- AF: Wang et al., *Adaptive Fingerprinting: Website Fingerprinting over Few
  Encrypted Traffic*, CODASPY 2021, DOI `10.1145/3422337.3447835`.
  [Author repository, pinned commit](https://github.com/SmartHomePrivacyProject/AdaptiveFingerprinting/tree/f9d33289f24447fe5edd738c0394d269d4648fb1).
  Compared `AF/one_classifier.py`, `AF/model.py`, `AF/test.py`,
  `AF/data.py`, `AF/optimizer.py`, and `tools/utility.py`.
  AF implementation details below are verified against those author files;
  a complete publisher PDF was not available in this audit.

## RAL findings

All **14 common RAL methods**, including initialization, fit, query decision,
reward, EXP4, threshold update, budget gate, acquisition and full committee
refitting, have identical abstract syntax to the author implementation after
removing docstrings and source locations. `ral_author_fingerprint.json` pins
the author method hashes; the regression test checks against this external
reference. The local unused reporting helper differs from the author's
`evaluate_performance`; missing sklearn metric imports were repaired.

| Component | Current RAL | Audit decision |
|---|---|---|
| Committee | kNN k=5, decision tree, RF with 10 trees | Matches author/MAWI example; keep |
| Reward | wrong prediction +1; correct prediction −1 | Matches; keep |
| Query/EXP4/threshold | weighted query votes, epsilon exploration, author update rules | Matches; keep |
| Acquired labels | append once; refit all members and soft-voting classifier | Matches; no extra neural model or replay added |
| Prefix gate | acquired/seen < budget, plus common strict total cap | Author gate plus disclosed benchmark cap |
| Initialization | all TRAIN rows at 0/1/3 | Matched-source benchmark adaptation, not original small bootstrap |
| Evaluation | shuffled independent levels, predict before feedback, budget .2 | AMCAL benchmark adaptation, not original experiment chronology/budget |

The IAL paper varies chronological initialization from .5% to 15%, reserves
the last 30% for validation, uses budget .05, and reports final validation
accuracy after streaming; it does **not** establish a universal requirement
that RAL must train on only .5%. Shrinking only RAL's training set while giving
AMCAL full source data would introduce a new confound. We retain matched source
supervision and explicitly name it `RAL-author-core-matched-source`. Source
label count and initialization are now saved in metrics. Historical RF100
checkpoints remain rejected by the existing RF10 guard.

## AF findings and correction

AF means **Adaptive Fingerprinting**, not Adaptive Random Forest (ARF).

| Component | Previous Burst AF extension | Author reference / new separate runner |
|---|---|---|
| Extractor | tabular MLP | 4 two-convolution blocks, 32/64/128/256 channels, ELU, BN, SAME pooling, global average, 512-D embedding |
| Source support | all labelled source TRAIN rows retained in final kNN | source M=25/class used for adaptation, excluded from final kNN |
| Domain adaptation | one GRL joint optimizer; clean source and training shifts | two alternating Adam steps and flipped domain labels, 1000 iterations, author convolutional heads and loss weights 4/4 |
| Target supervision | no author fine-tuning stage; arbitrary paid stream labels | N labelled target calibration examples/class; supervised target fine-tuning, up to 30 epochs, training-accuracy patience 10 |
| kNN | all source + paid target support; uniform vote over normalized embeddings | target calibration support ONLY; distance weights, cosine metric, k=N |
| k | selected from 1/3/5/7 using validation | fixed by N, no validation search |
| Online updates | DANN and complete support re-embedding after every query | no additional online updates; frozen evaluation after calibration |
| Target label schedule | uniform 20% of evaluation stream | few-shot calibration protocol; no active-learning query selector |

These are substantive method differences, not just hyperparameters. Some
changes could improve accuracy, others could harm it; their net effect must
be measured. The old extension's strong result is not the original AF result.
It is now excluded from default matched-method lists. Explicit `--methods af`
fails before creating output unless `--allow-af-extension` is supplied, and
the historical implementation is labelled `AF-Burst-online-extension-NOT-paper`.
Its core/checkpoints remain reproducible; evaluation fingerprints distinguish
explicit extension runs from previous output filenames.

`amcal_af_paper_burst.py` implements the separate author-structure protocol.
It never accepts test features/labels in its fitting function. All calibrated
support IDs, test IDs, hashes, source/calibration label counts, parameters,
training logs, checkpoint and predictions are saved. Insufficient per-class
data fail explicitly before starting a run; N is never reduced automatically.

### Exactness limits (do not label the Burst port an exact WF reproduction)

The new backend is a **PyTorch port of the author structure**, not a numerical
parity-validated execution of the original Python3.6 / TensorFlow2.3 / Keras
code. It follows the unusual published two-step flipped-label convention
rather than silently replacing it with GRL. BN buffers of the frozen side are
restored between adversarial phases, as in the author weight-restoration
implementation. Target tuning starts with the freshly initialized FeaturesVec
projection used by author `test.copy_weights` (the last three extractor layers
are excluded from pretrained weight copying). Training-accuracy early stopping
retains last-epoch weights, as the author code does; no heldout score selects
weights or epochs.

Explicit data/backend adaptations:

- Continuous Burst IAT/PS, AMCAL's existing TRAIN-fitted preprocessing and
  partitions replace 5000 packet-direction WF features and original datasets.
  Features are not converted to fake packet directions or padded to 5000.
- Source uses level-0 TRAIN, 25/class. Target calibration uses a 20/class pool
  from remaining TRAIN IDs at each target intensity. Original flow IDs used
  as source are excluded from target calibration as well as from TEST.
- Default evaluation uses all common heldout TEST rows. `-TestPerClass 70`
  selects the author's per-class test count and refuses undersized classes.
- Default N=5; `-Shots @(1,5,10,15,20)` runs the author N-shot sweep. These are
  separate N-shot fits, not repeated evaluation of one checkpoint.
- Seed42, one repetition per user instruction, rather than author's ten.
- If an N-shot Burst task has fewer than 64 total calibration rows, domain
  minibatches are reduced to that size and recorded. The original generator
  samples 64 without replacement and would fail on such a small task.
- PyTorch optimizer/BN/random-number numerics can differ from Keras. No claim
  of original accuracy reproduction or expected accuracy decrease is made.

**Comparison rule:** AF here has target-intensity calibration *before* scoring.
Its batch accuracy must not be placed in AMCAL's online-accuracy column as if
it used the same 20% streaming label protocol. Calibration labels are reported
separately; class-balanced few-shot selection is an offline dataset protocol,
not a label-blind active selection rule. RAL's matched online comparison remains
valid under the declared common source/stream rules.

## Run (Windows PowerShell)

```powershell
git pull --rebase --autostash origin AMCAL_final_enhancement
if ($LASTEXITCODE -eq 0) {
    .\AMCAL\run_amcal_ral_af_audit.ps1 -Dataset IAT -Seed 42
}
```

This retrains/evaluates unchanged author-core RAL under the matched-source
protocol and then runs AF N=5 with the author structure. New timestamped output
folders avoid collisions and preserve previous results. For PS use `-Dataset PS`.
For AF only, use `-Method AF`; for RAL only use `-Method RAL`. Use `-Shots
@(1,5,10,15,20)` only when the full N-shot sweep is wanted. The default is one
seed, one N and one evaluation per target level. SourceRun/BurstDir can be
overridden if the existing structured source folder has a different name.

```powershell
$run = Get-ChildItem 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs' -Directory -Filter 'ral_af_fidelity_seed42_*' |
    Sort-Object Name -Descending | Select-Object -First 1
git add -f -- "$($run.FullName)/ral_matched_source/training_manifest.json" "$($run.FullName)/ral_matched_source/Results" "$($run.FullName)/af_fewshot/execution.json" "$($run.FullName)/af_fewshot/combined_af_fewshot.csv" "$($run.FullName)/af_fewshot/*/metrics.json" "$($run.FullName)/af_fewshot/*/predictions.csv"
if ($LASTEXITCODE -eq 0) { git commit -m "Add audited RAL and AF few-shot results" }
if ($LASTEXITCODE -eq 0) { git pull --rebase --autostash origin AMCAL_final_enhancement }
if ($LASTEXITCODE -eq 0) { git push origin AMCAL_final_enhancement }
```

For AF-only/RAL-only runs, add only the directories/files that were produced.
Large model files are intentionally excluded from this results-push command.

## Verification

`test_ral_af_fidelity.py` checks the 14 RAL methods against pinned author AST
hashes; explicit opt-in for the legacy AF extension; target-only distance/cosine
k=N voting; preservation of extractor weights/BN in the discriminator phase;
and actual CSV loading, disjoint flow IDs, N=1/N=5 adaptation/tuning, checkpoint
save/load and frozen inference. The existing matched-baseline suite verifies
predict-before-feedback, budget enforcement and RAL reward/exploration rules.
Full dataset/GPU accuracy must be obtained by the new runs; this audit does
not invent replacement results or assume that corrected AF will be weaker.
