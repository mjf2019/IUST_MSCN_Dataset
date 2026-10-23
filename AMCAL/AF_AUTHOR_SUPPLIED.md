# Corrected author-structure AF on the four supplied datasets

Use amcal_af_clean_supplied.py, NOT --methods AF in amcal_clean_supplied.py.
The latter is the historical online adapter. No old output is replaced.

The existing baselines/af_author_protocol.py is called unchanged: author-style
convolutional extractor and heads, two optimizers/flipped-domain-label procedure,
feature-vector reinitialization before supervised target tuning, and target-only
cosine distance-weighted kNN with k=N. This is the corrected documented PyTorch
port, not a bit-identical historical Keras reproduction.

Source data: 25 examples/class from clean original TRAIN only, as in our corrected
AF runner. Same clean TRAIN-fitted scaler and group-separated source partitions.
No AMCAL CNN, Context or RL is trained. Same 12 attack/strength files as the clean
AMCAL experiment: CW/DeepFool .1,1,5; FGSM/PGD .01,.1,.5. Every task constructs a
fresh AF model and optimizers with the same seed, with no cross-level adaptation.
AF's core jointly adapts clean source and target calibration; it does NOT load a
source-only checkpoint and then run an online label acquisition policy.

Default N=5 labels/class, 1000 adversarial iterations, up to 30 tuning epochs,
exactly the corrected author runner defaults. Support labels must fit within
floor(.2 * target_pool_rows) independently for each task. This is a ceiling,
not a requirement to spend all 20%. Smaller --shots must be explicitly requested
if the task cannot fit the cap; there is no silent parameter change.

The target pool is the original held-out source TEST partition. Class-balanced
N-shot support selection uses target labels, as an OFFLINE few-shot benchmark,
not a label-blind policy. Support rows are removed from the final query set.
Query features/labels never enter AF fit/tuning/kNN support. Consequently its
offline post-calibration accuracy is not directly interchangeable with AMCAL's
pre-feedback online accuracy over the full stream. Forgetting is not claimed:
this author structure has no separately fitted clean-only checkpoint baseline.
The historical 80/20 attack-file source-order assumption is explicitly recorded;
use --assume-historical-pool-order only for those historical supplied files when
no explicit *_rows.csv mapping exists.

```powershell
python AMCAL/amcal_af_clean_supplied.py --dataset SDNCampus --shots 5 --seed 42 --device cuda --assume-historical-pool-order
```

Other dataset keys: ISCX-TOR, IUST, IoT. Default output per dataset is
protocol_runs/clean0_AF_author_shot5_attacks_seed42/. Results/af_author_comparison.csv
contains the 12 tasks. AF_author/<attack>/level<id>_shot5/ contains metrics,
predictions, checkpoint and completion hashes. Complete tasks can be skipped on
repeat with identical inputs. Partial tasks are preserved and require a fresh
--output. No numerical tests were run; Python syntax/static routing reviewed.
