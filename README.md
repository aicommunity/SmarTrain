> Russian version: [docs/ru/README.md](docs/ru/README.md)

# Smart Train (`smartrain`)

A CLI toolkit for preparing YOLO datasets, training models, running queues, and analyzing runs.

## Quick start

Requirements: Python `3.10+`.

```bash
git clone <repo-url>
cd smart-train
pip install -e .
```

Working with the workspace:

`SMART_TRAIN_WORKSPACE` is optional. If it is not set, `smartrain` uses the current directory as the workspace root.

```bash
smartrain deploy
smartrain scan
smartrain fusion --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data 2026-01-01_12-00-00-merged -y
```

Optional explicit workspace root:

```bash
export SMART_TRAIN_WORKSPACE=/path/to/workspace
smartrain deploy
```

## What's included

- Single entry point: `smartrain` (module `smartrain.cli`).
- Single-workspace model: `raw_data/`, `datasets/`, `runs/`, `analytics/`, `models/`, `tmp/`.
- Pipeline support: `scan -> fusion -> train -> analyze`.
- Additional tools: `queue`, `registry`, `cvat`, `sahi`, `heatmap`, `orient`.

## How it works

`smartrain` uses a single workspace root and builds a process around file contracts:

- `scan` synchronizes sources and updates the dataset catalog;
- `fusion` generates the final dataset for training;
- `train` creates a run directory with metrics and metadata;
- `analyze` and `registry` work on artifacts in `runs/`.

## Key commands

| Command | Purpose |
|---|---|
| `smartrain deploy` | Initialize the workspace structure |
| `smartrain scan` | Synchronize sources and update the dataset catalog |
| `smartrain fusion` | Build the final training dataset |
| `smartrain train` | Train and validate YOLO models |
| `smartrain queue` / `smartrain queue-run` | Manage and run the command queue |
| `smartrain analyze` | Summaries, run comparison, PR curves, and inference benchmarks |
| `smartrain registry` | Catalog run artifacts and promoted models |

## Documentation

Current documentation is organized into sections in `docs/`:

- [Documentation navigation](docs/index.md)
- [Getting started and core workflows](docs/getting-started/quickstart.md)
- [CLI guide](docs/cli/overview.md)
- [API and format reference](docs/reference/api.md)
- [Architecture and diagrams](docs/development/architecture.md)

## Important details

- Interactive mode starts only when a command is launched with zero arguments (TTY required).
- Interactive dataset commands: `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`; plus `train`.
- Dataset cleanup command: `prune` (`prune empty` for empty pairs, `prune dedup` for duplicate images by content).
- If any arguments are provided but required ones are missing, commands return a clear "incomplete arguments" error instead of interactive prompts.
- Command help now includes practical `Examples` / `Quick examples` blocks for common workflows.
- `smartrain balance` presets:
  - `--preset weights-safe` for conservative balancing
  - `--preset rfs-aggressive` for stronger tail upsampling
  - `--preset hybrid-default` as a general default
- `smartrain balance` eval splits: `--eval-coverage` is on by default (keeps `val`/`test` non-empty when possible and improves class coverage there); use `--no-eval-coverage` to disable. The interactive wizard asks for this option.
- For `hash --validate`: `0` for a match, `1` for a mismatch, `2` for an error.
- By default, the workspace queue uses `queue.txt` and `tmp/status.txt`.
- Dependency extras:
  - `pip install -e ".[dev]"` for development and testing
  - `pip install -e ".[clearml]"` for ClearML
  - `pip install -e ".[sahi]"` for SAHI

## Common workflows

Scanning with an explicit source list:

```bash
smartrain scan --datasets-list /path/to/workspace/raw_data/datasets_list.txt
```

Check dataset hash:

```bash
smartrain hash --dataset my_dataset
smartrain hash /path/to/dataset --validate a1b2c3d4
```

Starting a queue without opening a GUI terminal:

```bash
smartrain queue run --no-gui
```

Quick run overview:

```bash
smartrain analyze scan
smartrain analyze export-table -o runs_summary.csv
```
