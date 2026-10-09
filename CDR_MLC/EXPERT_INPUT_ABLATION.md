# MF-CDR-MLC expert-input ablation (Reviewer 2, comment D)

This experiment directly compares separated expert inputs with complete expert
inputs in the full MF-CDR-MLC architecture. The nominal 32/35 schema does not
imply that every column is usable in each development partition: the production
selector rejects constant or otherwise unusable fields. Actual counts such as
24/27 are recorded without changing this existing selection behavior. The only
intervention is restoring `AckDat`, `TcpRtt`, and `SynAck` to the RF expert
inputs. Both variants retain those measurements in the congestion-context
path, utility estimation, meta-fusion, level balancing, and confidence fallback.

The production implementation is reused. Expert input selection applies to
both the preliminary fit and the full-development refit. Each variant trains
its own utility and fusion models because its expert outputs change. Both use
the same candidate sets and development-only selection procedure; selected
fusion variants or confidence thresholds may consequently differ. Test scores
are never used to select either model.

## Run from the repository root

Use the existing Python environment or install the small dependency set:

```powershell
python -m pip install -r CDR_MLC/requirements-expert-ablation.txt
python CDR_MLC/expert_input_ablation.py --seeds 42
```

The default runs all seven evaluations with the paper ablation settings:
base window 3, congestion window 50, 20 expert trees, 10 utility trees, 20
meta trees, and final expert refit enabled. Scenarios 1--3 use the complete
source level and complete untouched target level; the four mixed protocols
are LM-H, LH-M, MH-L, and ALL-80-20. ALL-80-20 uses chronological 80/20
development/test splits within captures. This matches the existing component
ablation protocol; it differs from the confirmatory experiment's default
20% ordered target tails. Set `--target-test-fraction 0.20` if comparing with
that confirmatory protocol, and use a separate output folder.

For the paper's ten paired seeds:

```powershell
python CDR_MLC/expert_input_ablation.py --seeds 42 52 62 72 82 92 102 112 122 132 --output CDR_MLC/outputs/expert_input_ablation_10seeds
```

For a quicker first check on Scenario 1 only:

```powershell
python CDR_MLC/expert_input_ablation.py --scenarios 1 --protocols --seeds 42 --output CDR_MLC/outputs/expert_input_ablation_quick
```

An interrupted run can be continued with the **same full command** plus
`--resume`. Completed pairs are reused; an interrupted pair is rerun. Input
capture hashes, code hashes, and configuration must match. Use a new output
folder for a different seed list or protocol. Reports are checkpointed after
every completed pair, so early results are available before the entire run ends.

## Outputs and interpretation

All generated reports are under the selected `outputs/` directory, already
ignored by Git:

- `per_seed_metrics.csv`: both models' accuracy, balanced accuracy, macro-F1,
  weighted-F1, and evaluated row count for every protocol/seed.
- `summary.csv`: per-protocol mean and sample standard deviation over seeds.
  Standard deviation is undefined with one seed.
- `paired_deltas.csv`: **all inputs minus separated inputs** for each score.
  Positive favors including the three timing fields; negative favors separation.
- `<protocol>/seed_<seed>/predictions.csv`: paired predictions on identical rows.
- `<protocol>/seed_<seed>/pair.json`: selected thresholds/variants, selection
  trials, actual expert columns and encoded dimensions, and invariant audits.
- `manifest.json` and `input_audit.csv`: run configuration and input provenance.

The runner verifies identical routing geometry, context values, partition
sizes, and eligible rows, and rejects development/test overlap. It requires
that the complete expert input set differs by exactly the three timing fields,
with all shared inputs preserved. Metric rows record the actual raw and encoded
feature counts; category encoding can increase the RF's internal dimension.
The method names describe the intervention rather than assume fixed counts.

Report all seven protocols and all requested seeds, regardless of which model
wins. A one-seed check is exploratory evidence, not a significance claim.
No performance result from the real dataset was generated during implementation.

## Local integration checks

```powershell
python -m unittest discover -s CDR_MLC/tests -p test_expert_input_ablation.py -v
```

Synthetic flows test the paired invariant checks with and without final expert
refit, default-model equivalence, prediction invariance to hidden test labels,
and the sign of the paired differences. These checks are not paper results.
