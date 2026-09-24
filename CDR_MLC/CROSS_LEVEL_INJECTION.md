# Leakage-safe cross-level injection pools

`build_cross_level_injection_pools.py` creates three independent training
pools without modifying Clean-Valid:

| Pool | Primary development data | Injected donor prefixes |
|---|---|---|
| Low | Low | 20% Medium + 20% High |
| Medium | Medium | 20% Low + 20% High |
| High | High | 20% Low + 20% Medium |

The default final 20% of every original capture is exported as an immutable
fixed test. Injection always comes from the chronological prefix and cannot
overlap any fixed-test row. Windows must continue to be grouped by
`sequence_id`, so they never cross application, capture or physical level.

The script does **not** relabel physical congestion. `congestion_level` and
`original_congestion_level` retain the donor level; `pool_level` identifies the
training pool receiving that row. This distinction prevents injected Low/High
records from being falsely presented as physically Medium, for example.

## Build

```powershell
python CDR_MLC/build_cross_level_injection_pools.py --injection-fraction 0.20 --test-fraction 0.20 --output CDR_MLC/DATASETS/CDR-MLC/Cross_Level_Injection_20
```

Outputs:

- `training_pools/low_training_pool.csv`
- `training_pools/medium_training_pool.csv`
- `training_pools/high_training_pool.csv`
- `fixed_tests/low_fixed_test.csv`
- `fixed_tests/medium_fixed_test.csv`
- `fixed_tests/high_fixed_test.csv`
- composition, partition and overlap audits plus `manifest.json`
