# CDR-MLC reproducible experiments

This directory has one authoritative implementation of CDR-MLC:
`cdr_mlc/model.py`. Both the command-line runner and the notebook import this
module; model logic is not duplicated in notebooks.

## Leakage-safe contract

1. The router is fitted only on causal rolling statistics of `SynAck`,
   `AckDat`, and `TcpRtt`.
2. The fitted router assigns every training and inference sample with the same
   `predict` rule.
3. One multiclass Random Forest expert is fitted per routed cluster.
4. `label`, encoded labels, level labels, class indices, and unnamed CSV index
   columns are never classifier inputs.
5. An inference label is neither required nor inspected.

## Setup on Windows PowerShell

From this `CDR_MLC` directory:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-validation.txt
python -m pytest tests -q
```

Place the full dataset locally at:

```text
DATASETS/CDR-MLC/scale_0.001/
├── Short/
│   ├── level_1.csv
│   ├── level_2.csv
│   ├── level_3.csv
│   └── CDR-MLC-Shuffle.csv
└── Long/
    └── CDR-MLC-Shuffle.csv
```

The full dataset and generated run directory are ignored by Git.

## Reproduce Table 5

First run one deterministic seed:

```powershell
python run_table5.py --seeds 42
```

For repeated experiments:

```powershell
python run_table5.py --seeds 10 20 30 40 42
```

To run only one scenario:

```powershell
python run_table5.py --scenarios scenario_2 --seeds 42
```

Generated JSON and CSV files are written to `results/runs/table5/` by default.
Use `CDR-MLC.ipynb` to launch runs and inspect the resulting table and plots.

The loader uses 32-bit numeric columns and releases the training DataFrame
before loading the test set. This reduces peak memory for the 1.43 GB dataset;
the scenarios and seeds are intentionally executed sequentially.
