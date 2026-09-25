# FT-Transformer benchmark

Independent numerical FT-Transformer implementation for the seven IUST_MSCN
protocols. Numerical features are converted to feature tokens, a learnable CLS
token is processed by Transformer encoder blocks, and the CLS representation is
classified. Preprocessing and early stopping use development data only.

```powershell
pip install -r CDR_MLC/benchmarks/FT_Transformer/requirements.txt
python CDR_MLC/benchmarks/FT_Transformer/ft_transformer_iust_mscn.py `
  --scenarios 1 2 3 4 5 6 7 `
  --seed 42 `
  --device cuda `
  --output CDR_MLC/outputs/ft_transformer_seed42
```

Omit `--device cuda` for automatic device selection. Complete metrics,
configuration and split audits are written to the output directory.
