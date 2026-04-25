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

Work from the project root (current directory):

```bash
smartrain deploy
smartrain scan
smartrain fusion --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data 2026-01-01_12-00-00-merged --device 0 -y
```

## What's included

- Single entry point: `smartrain` (module `smartrain.cli`).
- Single-workspace model: `raw_data/`, `datasets/`, `runs/`, `analytics/`, `models/`, `inference/`, `tmp/`.
- Pipeline support: `scan -> fusion -> train -> analyze`.
- Additional tools: `queue`, `registry`, `report`, `model`, `normalize-data-yaml`, `migrate-models`, `clearml-upload`, `plot`, `cvat`, `sahi`, `heatmap`, `orient`.

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
| `smartrain inference` | Run inference on folder or dataset split and save JSON report |
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

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Important details

- Interactive mode starts only when a command is launched with zero arguments (TTY required).
- Interactive dataset commands: `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `inference`; plus `train`.
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
- Device selection in `train` and `inference`:
  - `--device 0` to force GPU 0
  - `--device cpu` to force CPU
  - If `--device` is omitted, default is `GPU 0` when CUDA is available, otherwise `cpu`
- PyTorch CUDA policy:
  - default target is CUDA 12.8 wheels (`cu128`)
  - if current environment already has `torch` with CUDA `13.x`, `smartrain` keeps it and does not downgrade
  - to apply policy in the current environment: `smartrain deps sync-torch`
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

Train and inference with explicit device:

```bash
smartrain train --data my_dataset --model yolo11n.pt --device 0
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --device cpu
```

## Running long jobs over SSH (tmux)

For long training runs on a remote server, use `tmux` so the job survives SSH disconnects.

Install `tmux` once (Ubuntu/Debian example):

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

Minimal workflow:

```bash
tmux new -s smartrain-train
smartrain train --data my_dataset --model yolo11n.pt --device 0
```

- Detach without stopping the training: `Ctrl+B`, then `D`
- Re-attach after reconnecting: `tmux attach -t smartrain-train`
- Stop training gracefully from attached session: `Ctrl+C`
- Close an unused session: `tmux kill-session -t smartrain-train`

You can also use helper scripts from `scripts/`:

```bash
./scripts/tmux_train_start.sh --session smartrain-train -- smartrain train --data my_dataset --model yolo11n.pt --device 0
./scripts/tmux_train_attach.sh --session smartrain-train
./scripts/tmux_train_stop.sh --session smartrain-train
```

Optional: keep a file log while preserving live console output:

```bash
./scripts/tmux_train_start.sh --session smartrain-train -- bash -lc 'smartrain train --data my_dataset --model yolo11n.pt --device 0 2>&1 | tee -a runs/train.log'
```

### Operations quick recipes

Check active tmux sessions:

```bash
tmux ls
```

See whether training process is still alive in session:

```bash
tmux list-panes -t smartrain-train -F '#{pane_current_command} #{pane_pid}'
```

Recover live console output after reconnect:

```bash
tmux attach -t smartrain-train
```

If already attached elsewhere, force re-attach:

```bash
tmux attach -d -t smartrain-train
```

Graceful stop and cleanup:

```bash
./scripts/tmux_train_stop.sh --session smartrain-train
tmux kill-session -t smartrain-train
```

### FAQ (tmux over SSH)

**Session exists, but no new output appears. What to check first?**
- Re-attach with force detach: `tmux attach -d -t smartrain-train`
- Check current pane command: `tmux list-panes -t smartrain-train -F '#{pane_current_command} #{pane_pid}'`
- If your training wrote logs via `tee`, inspect the log file (for example `runs/train.log`).

**I accidentally closed SSH. Did training stop?**
- Usually no, if it was started inside `tmux`.
- Reconnect and run: `tmux ls`, then `tmux attach -t smartrain-train`.

**Ctrl+C does not stop the run from my current shell.**
- Ensure you are attached to the right `tmux` session/window first.
- Or send interrupt explicitly: `./scripts/tmux_train_stop.sh --session smartrain-train`.

**How to quickly find the latest training logs?**
- Example pattern:
  - `ls -lt runs | head`
  - `tail -n 200 runs/train.log` (if you used `tee -a runs/train.log`)

**How to clean up stale tmux sessions?**
- List sessions: `tmux ls`
- Remove one: `tmux kill-session -t <session>`
- Remove all server sessions (careful): `tmux kill-server`

## Developers

- [@palexab](https://github.com/palexab)
- [@greisersem](https://github.com/greisersem)
