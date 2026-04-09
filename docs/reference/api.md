> Russian version: [../ru/reference/api.md](../ru/reference/api.md)

# Reference: APIs and modules

## Entry point

- `smartrain/cli.py` — Typer-router command.
- `smartrain/cli_argparse.py` is a basic argparse parser with default values ​​in `--help`.

## Basic modules

- `smartrain/datasets_json_former.py` — `scan`.
- `smartrain/dataset_former.py` — `fusion`.
- `smartrain/model_training_module.py` — `train`.
- `smartrain/training_queue.py` and `smartrain/training_queue_cli.py` are the queue.
- `smartrain/results_analyzer.py` — `analyze`.
- `smartrain/registry_cli.py` — `registry`.
- `smartrain/dataset_hash.py` — `hash`.

## CLI mapping -> module

| CLI command | Module |
|---|---|
| `smartrain scan` | `smartrain/datasets_json_former.py` |
| `smartrain fusion` | `smartrain/dataset_former.py` |
| `smartrain train` | `smartrain/model_training_module.py` |
| `smartrain analyze` | `smartrain/results_analyzer.py` |
| `smartrain queue` / `queue-run` | `smartrain/training_queue_cli.py` / `smartrain/training_queue.py` |
| `smartrain registry` | `smartrain/registry_cli.py` |

## Actual behavior notes

- `hash --validate`: `0` (match), `1` (mismatch), `2` (error).
- Extended subcommands are available in `analyze`: `pr-curves`, `inference-benchmark`, `inference-plot`.
- The queue in the workspace uses `queue.txt` by default.

For detailed command examples, see [CLI section](../cli/overview.md).
