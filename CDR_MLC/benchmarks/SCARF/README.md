# SCARF benchmark

Independent SCARF-style tabular self-supervised benchmark. The encoder is
pretrained by contrasting clean records with feature-corrupted views generated
only from the training partition, then fine-tuned for application
classification with chronological validation.

```powershell
pip install -r CDR_MLC/benchmarks/SCARF/requirements.txt
python CDR_MLC/benchmarks/SCARF/scarf_iust_mscn.py `
  --scenarios 1 2 3 4 5 6 7 `
  --seed 42 `
  --device cuda `
  --output CDR_MLC/outputs/scarf_seed42
```

The test partition is not used during self-supervised pretraining, supervised
fine-tuning or early stopping.
