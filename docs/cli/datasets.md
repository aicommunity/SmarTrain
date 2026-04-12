> Russian version: [../ru/cli/datasets.md](../ru/cli/datasets.md)

# CLI: datasets

## `scan`

Updates the dataset index and synchronizes sources in the workspace.

- Output files: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Supports sources from `raw_data/`, `--dataset`, `--datasets-list`.
- Useful modes: `--mode refresh`, `--purge-processed-raw`.

## `normalize-data-yaml`

Rewrites every `datasets/**/data.yaml`: drops `path`, makes split paths relative. Foreign absolute paths from another machine are mapped to `train/images`, `val/images`, etc. when those folders exist under the same dataset root.

Example: `smartrain normalize-data-yaml --workspace /path/to/workspace` or `--datasets-dir ... --dry-run`.

## `fusion`

Collects a new dataset from several sources:

- selection of inputs: `--dataset` (repeatable) or `--datasets` (CSV);
- class management: `--classes`, `--merge-classes`, `--common-classes-only`;
- crash: `--fusion-split train,val,test`.

## `augment`, `balance`, `orient`, `roi`

- `augment` — autonomous augmentations with recording of a new dataset;
- `balance` — class balancing; after balancing, `--eval-coverage` (default) can rebalance items across `train`/`val`/`test` so eval splits are non-empty when possible and rare classes appear in `val`/`test`; `--no-eval-coverage` turns this off;
- `orient` — frame rotation correction;
- `roi` — crop according to the ROI-model.

All of the above commands form `dataset_passport.json` in the new dataset directory.

## `hash`

Checking and calculating the hash of the dataset structure:

```bash
smartrain hash --dataset my_dataset
smartrain hash /abs/path/to/dataset --validate a1b2c3d4
```

`--validate` exit codes: `0` match, `1` mismatch, `2` error.

## `stats`

Subcommand:

- `classes`
- `datasets`
- `compare`
