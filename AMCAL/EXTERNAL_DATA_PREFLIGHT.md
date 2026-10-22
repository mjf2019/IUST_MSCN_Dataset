# External CSV preflight for PairOnly

The pushed schema report is on experiment/paironly-paid-replay. It describes only
the first 256 rows of each CSV, not full class counts. DReLAB has 59 CSVs including
one raw_labelled_flow file; the task CSVs use separate source/target subdirectories.
NPBOv2 uses a ternary column named label, whereas TUN uses label_3. Both contain
many label metadata columns that must never become model inputs. DReLAB attack
Label=adversarial maps to malicious=1, not a third prediction class. n_actions
is generation metadata and must not be an input. Original column spelling is
preserved (DReLAB has distinct direction columns containing whitespace).

Run from repository root:

```powershell
python AMCAL/amcal_external_data_audit.py
```

This reads all CSV rows in chunks without PyTorch or GPU, validates binary/role
labels, excludes raw flow export, and saves AMCAL/external_datasets_audit.json.
It reports excluded identifiers/labels, candidate features, nonnumeric/missing/
infinite counts, file hashes and full class/role counts. Raw CSVs are unchanged.
Errors are retained in the report and result in exit code 2. Candidate features
are only a schema audit, not a learned feature selection. The source/attack row
indices do not yet establish identity correspondence or split independence.

This is the data readiness stage, not an AMCAL training/evaluation runner.
Before training, create independent source and target development/test partitions,
resolve derivative identity overlap for DReLAB, and inspect attack group sizes for
ASNM. Any imputation, feature selection and scaling must fit TRAIN only.
The current Burst runner cannot consume these CSV layouts directly.
Retain selected PairOnly settings: one Context, three actions, Context LR .0005,
20% feedback cap; no extra paid-replay controls enabled by default.
A source CNN/Context and selector must be trained for the new feature schemas.

Validation: four pandas/numpy CPU fixture tests pass, including ternary/binary
consistency, DReLAB path/label validation, complete chunked reading, semicolon CSV,
missing inputs, metadata exclusion, nonfinite reporting and raw-byte preservation.
No user full dataset, PyTorch training or GPU evaluation was executed here.

```powershell
python -m unittest discover -s AMCAL/tests -p test_external_data_audit.py -v
```
