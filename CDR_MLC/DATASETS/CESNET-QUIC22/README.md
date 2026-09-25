# CESNET-QUIC22 (XS)

This directory contains the reproducible download entry point for the official
CESNET-QUIC22 dataset in its smallest published size (`XS`). Downloaded data are
kept outside Git through `.gitignore`.

The download step is deliberately separate from preprocessing and evaluation.
No train/test period, feature subset, label mapping, or scaler is selected here.

## Windows PowerShell

Run from the repository root on branch `revision/quic-evaluation`:

```powershell
python -m pip install -r CDR_MLC/DATASETS/CESNET-QUIC22/requirements-download.txt

python CDR_MLC/DATASETS/CESNET-QUIC22/download_cesnet_quic22_xs.py
```

By default, DataZoo stores the files below:

```text
CDR_MLC/DATASETS/CESNET-QUIC22/raw/XS/
```

To store the dataset on another drive while keeping the script in the repository:

```powershell
python CDR_MLC/DATASETS/CESNET-QUIC22/download_cesnet_quic22_xs.py `
  --data-root "D:\\Datasets\\CESNET-QUIC22"
```

The script is restart-safe through DataZoo: existing completed files are reused.
After successful verification it writes `download_manifest.json` containing the
package version, database path, file size, completion time, and available periods.

## Official sources

- Dataset: CESNET-QUIC22
- Size: XS
- Collection periods: W-2022-44, W-2022-45, W-2022-46, and W-2022-47
- Data access library: `cesnet-datazoo`
- Dataset DOI: <https://doi.org/10.1016/j.dib.2023.108888>
- Dataset record: <https://zenodo.org/records/7963302>
- DataZoo: <https://github.com/CESNET/cesnet-datazoo>
