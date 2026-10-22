# Modified RAL with five recent feedback rows

```powershell
.\AMCAL\run_amcal_ral_recent5.ps1 -Dataset IAT -Seed 42
.\AMCAL\run_amcal_ral_recent5.ps1 -Dataset PS -Seed 42
```

Reuse the previously trained paper-core RF10/kNN5/DT committee checkpoints.
New result roots ral_recent5_seed42, independent fresh model per level,
levels 5/7/10/12/15/17/20, original partitions and prefix ratio + strict 20% cap.
Original RAL, its checkpoints and results remain unchanged (default window=0).

Online acquire/update keeps ONLY the five latest acquired feedback rows including
the current row, no initial source rows in refits. Until five feedback rows have
arrived, retain the source-trained models, but run the ordinary alpha/threshold
feedback updates. This avoids invalid kNN5 predictions with fewer than 5 rows.
Thereafter refit the committee on this exact sliding window after every query.
Metrics record refit counts, maximum fit rows, retained rows and variant identity;
config fingerprints differ from original RAL.

This is RAL-recent5-modified, not a reproduction of the original RAL algorithm.
sklearn fit rebuilds RF/DT and replaces kNN support: it is NOT an incremental
parameter step on five rows. Earlier source knowledge is therefore not retained
after the first window refit. A five-row window may also omit classes. These are
intrinsic consequences of the requested variant, not a guarantee of a fairer or
stronger comparison. AMCAL additionally retains pretrained CNN knowledge and
protected replay; batch sizes alone do not equalize the methods or compute.
Report original RAL and recent5 separately, with time/labels/accuracy/retention.
Only syntax checks were run; numerical tests are left to the user as requested.
