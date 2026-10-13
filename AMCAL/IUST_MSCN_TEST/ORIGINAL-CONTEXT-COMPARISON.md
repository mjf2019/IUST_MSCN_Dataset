# Original Context architecture comparison

amcal_single_token.py reproduces the original single-token Context architecture
using the corrected v2 training/evaluation runner from commit 593fde25. It is an
architecture control, NOT an exact reproduction of the original notebook.

Use the same normal-data split, seed, epochs, learning rate and online budget as
the feature-token v3 pilot. Predictions are scored before label acquisition and
updates. The original notebook remains available as AMCAL.ipynb, unchanged.
Its post-update scoring, refitted scaler, query gates and selector exploration
must be distinguished from an architecture-only comparison.

Outputs default to protocol_runs/seed42_single_token_v2 and do not overwrite v3.
No training or evaluation was run by the assistant.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/IUST_MSCN_TEST/amcal_single_token.py train --base-epochs 5 --context-epochs 5 --pretrain-epochs 2
python AMCAL/IUST_MSCN_TEST/amcal_single_token.py evaluate --budget-fraction 0
python AMCAL/IUST_MSCN_TEST/amcal_single_token.py evaluate --budget-fraction 0.20 --online-selector-batch-size 32
```

These pilots diagnose architecture and adaptation differences; they do not
establish convergence or explain the original published results on attack levels.
