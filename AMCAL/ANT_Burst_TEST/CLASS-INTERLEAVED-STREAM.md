# Class-interleaved ordering with within-class FIFO

Default --stream-order class-interleaved --stream-seed 42 applies when reading the existing oversampled train/validation/test CSVs and the selected attack evaluation prefix. Original CSV files are not rewritten. Existing counts, duplicate oversampled rows, feature/label pairing and train/validation/test membership are preserved.

A local seeded RNG shuffles a multiset of class slots. Each slot consumes the earliest remaining original row of its class. This is proportional random interleaving, not equal class quotas, and does not guarantee balanced label queries. Labels are used only for offline benchmark ordering and scoring; they are not passed into the online selection state or gates.

Training visits each existing oversampled row once per epoch using ClassInterleavedSampler (seed + epoch). It replaces WeightedRandomSampler in interleaved mode: the latter samples with replacement and cannot preserve within-class order. CE class weights are unchanged. Validation/test ordering is fixed for a seed. This changes the training sampling protocol and requires fresh training for that comparison.

Attack evaluation selects the same first max_samples rows BEFORE interleaving. Both learned and no-RL random runners use the same helper. Identical class-label sequences and seed yield identical permutations across perturbation levels. If label sequences differ, permutations may differ; no unverified sample correspondence is claimed.

Results append _class_interleaved_seed42 (or the chosen seed). Traces include 1-based source_row, and diagnostics include the permutation hash. Model checkpoints record training stream settings. --stream-order input restores original input ordering and legacy weighted training sampling. Label-budget checks remain and reject exceeding the configured cap.

Validation: syntax, empty/single-class/mixed streams across seeds, permutation completeness, per-class FIFO, exact counts, reproducible epoch variation, independent RNG state, and the actual uploaded 1073-row label sequence. No PyTorch training/evaluation was run in this environment.
