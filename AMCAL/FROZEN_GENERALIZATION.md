# Burst Frozen CNN generalization check

This is an isolated CNN experiment. The existing CNN architecture, scaler,
source-ID partitions and training intensities are retained. No Context, route
scorer, query policy or reward code is changed. No data are downloaded.

The train command warm-starts the source CNN and refines it using the same
allowed TRAIN domains. Defaults: AdamW, LR 0.0002, weight decay 0.001, label
smoothing 0.05, 30 epochs and patience 8. Dropout rates stay unchanged.
Validation selects the lowest **worst-domain ordinary cross-entropy**; mean
domain cross-entropy breaks ties. The initial source CNN is also a candidate,
so epoch 0 is retained when refinement does not improve this criterion.
This does not guarantee improvement on unseen intensities.

Each epoch prints TRAIN and validation accuracy and ordinary unsmoothed CE per
domain. Both measurements use evaluation mode and the natural row distribution.
They are comparable, unlike a stochastic dropout training objective versus an
evaluation-mode validation loss. The CSV also includes balanced accuracy and
macro F1 (over all encoder classes). A positive Train%-Val% gap is a
generalization gap within that domain; good validation on seen domains and poor
heldout-domain results indicate a different transfer problem.

PowerShell, from the repository root, for the existing PS run trained with 0/1/3:

```powershell
git pull --ff-only origin experiment/amcal-protocol
$burst = "AMCAL/ANT_Burst_PS_TEST/Dataset_official_v1/AdvBurst_FTSC-PS"
$source = "AMCAL/ANT_Burst_PS_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42"
$frozen = "AMCAL/ANT_Burst_PS_TEST/protocol_runs/frozen_generalization_levels1_3_seed42"
```

Optional audit of the existing source CNN, without training:

```powershell
python AMCAL/amcal_burst_frozen_generalization.py audit --source-run $source --burst-dir $burst --levels 0 1 3 --seed 42 --device cuda
```

CNN-only training (a source audit is also printed automatically):

```powershell
python AMCAL/amcal_burst_frozen_generalization.py train --source-run $source --burst-dir $burst --output $frozen --levels 0 1 3 --epochs 30 --patience 8 --lr 0.0002 --weight-decay 0.001 --label-smoothing 0.05 --selection worst-loss --seed 42 --device cuda
```

Test separately, first on the two diagnostic levels:

```powershell
python AMCAL/amcal_burst_frozen_generalization.py evaluate --burst-dir $burst --output $frozen --levels 7 10 --seed 42 --device cuda
```

Or score all seven heldout intensities in one comparison:

```powershell
python AMCAL/amcal_burst_frozen_generalization.py evaluate --burst-dir $burst --output $frozen --levels 5 7 10 12 15 17 20 --seed 42 --device cuda
```

The evaluation table compares source and selected CNN on identical TEST source
IDs with zero online updates and zero acquired labels. Training never scores
TEST IDs. Training levels are taken from checkpoint provenance; an existing
warm start cannot be reclassified as having seen fewer intensities. Heldout
evaluation refuses every previously seen training intensity.

Output files:

- `Models/initial_cnn_model.pth`: selected CNN and its validation selection data.
- `Models/protocol_preprocessing.joblib`: unchanged source preprocessing.
- `frozen_training.json`: settings, source hashes, progress and best epoch.
- `Results/frozen_training_history.csv`: baseline, epoch and selected TRAIN/validation metrics.
- `Results/frozen_heldout_comparison.csv`: per-intensity source/selected TEST comparison.

Existing runs are preserved. Use a fresh output directory for each training.
The best checkpoint and progress are saved during training. The output contains
no copied Context or selector weights: those must be trained against a changed
CNN before using the full seven-action system. Keep heldout comparisons separate
from checkpoint selection; this tool selects solely on the configured seen
domains' validation rows.
