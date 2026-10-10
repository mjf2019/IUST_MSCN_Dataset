# Original Burst notebook reproduction

amcal_burst_legacy.py combines source cells 2 (training definitions), 7 (online
test definitions), and 9 (214-label test configuration) from AMCAL.ipynb.
Training and evaluation must be invoked separately, reproducing notebook order.

Legacy behavior is intentionally preserved: gamma 0.2, terminal replay,
label-dependent skip reward, training loss clamp, inverse online loss ratio,
post-update scoring, online epsilon 1, training replay loading, refitted scaler
on the PER=0 evaluation CSV, original shallow checkpoint snapshots, and original
validation mode behavior. This is diagnostic reproduction, not scientific
validation of those choices. The recorded old outputs cannot certify which
kernel state/checkpoint bytes produced them; numerical identity is not guaranteed.
The original notebook sets no random seed.

Only operational edits: disambiguated function names to avoid notebook cell
redefinition collisions; absolute data paths and separate output folder; explicit
torch.load(weights_only=False) for locally generated legacy replay checkpoints
under PyTorch 2.7; plot windows closed after saving to avoid blocking.
No experiments or training were run by the assistant.

Train: base max200/patience30; Context max300/patience30; pretraining20.
Evaluate: all ten PER levels, each from the initial saved models, lr0.25,
threshold0.005, max1073 rows, budget214, minibatch128, replay capacity500.
Output: protocol_runs/notebook_original, separate from corrected runs.

```powershell
git pull --ff-only origin experiment/amcal-protocol
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py train
python AMCAL/ANT_Burst_TEST/amcal_burst_legacy.py evaluate
```
