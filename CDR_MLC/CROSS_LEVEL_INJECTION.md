# Integrated 20% cross-level injection

The 15 existing files under `DATASETS/CDR-MLC/Clean_Valid` are materialized in
place. Their names and directory are unchanged, so every existing runner keeps
the same command.

For each application:

| Existing destination file | Integrated content |
|---|---|
| `<App>_Low.flow` | all Low + 20% Medium + 20% High |
| `<App>_Medium.flow` | all Medium + 20% Low + 20% High |
| `<App>_High.flow` | all High + 20% Low + 20% Medium |

The first `floor(0.20 × donor rows)` rows are selected from each pristine donor
capture. All 15 pristine files are loaded before any file is replaced, so an
already injected destination can never become a donor during the same build.

Two non-model metadata columns are added:

- `InjectionOriginLevel`: physical source level of the row;
- `InjectionRole`: `primary` or `injected`.

Both are explicitly forbidden as model features. Existing loaders continue to
derive the nominal level from the unchanged destination filename. Exact
pre/post hashes and row counts are stored in
`cross_level_injection_manifest.json`.

The script refuses to run if the injection manifest already exists. To rebuild,
first restore the pre-injection Git commit.

```powershell
python CDR_MLC/inject_cross_level_in_place.py --fraction 0.20
```
