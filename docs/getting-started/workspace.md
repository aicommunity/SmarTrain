> Russian version: [../ru/getting-started/workspace.md](../ru/getting-started/workspace.md)

# Workspace

`smartrain` uses a single workspace root:

- via the `SMART_TRAIN_WORKSPACE` environment variable;
- or via the global flag `smartrain --workspace /path/to/ws ...`.

`--workspace` takes precedence over the environment variable.

## Directory structure

- `raw_data/` — external sources of datasets;
- `datasets/` — processed datasets and indexes (`datasets_info.json`, `class_names.json`);
- `runs/` — training outputs;
- `analytics/` — analytics artifacts (`analyze export-table`, etc.);
- `models/` — promoted models (`registry models-add`);
- `tmp/` — system files, including `tmp/status.txt`.

The default queue file is `queue.txt` in the workspace root.

## Initialization

```bash
export SMART_TRAIN_WORKSPACE=/path/to/workspace
smartrain deploy
```
