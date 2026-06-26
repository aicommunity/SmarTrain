> Russian version: [../ru/reference/api.md](../ru/reference/api.md)

# Reference: APIs and modules

## Entry point

- `smartrain/cli.py` — Typer-router command.
- `smartrain/cli_entrypoints/support/cli_argparse.py` — shared argparse helpers (e.g. `CliArgumentParser`).

## Basic modules

- `smartrain/workflows/datasets/datasets_json_former.py` — `scan`.
- `smartrain/workflows/datasets/dataset_former.py` — `fusion`.
- `smartrain/workflows/training/train_entry.py` — `train`.
- `smartrain/workflows/queue/training_queue_cli.py` and `smartrain/workflows/queue/training_queue.py` — queue (`queue` subcommands and `queue-run`).
- `smartrain/workflows/analyze/results_analyzer.py` — `analyze`.
- `smartrain/workflows/registry/registry_cli.py` — `registry`.
- `smartrain/workflows/datasets/dataset_hash.py` — `hash`.
- `smartrain/workflows/inference/inference_cli.py` — `inference`.
- `smartrain/workflows/datasets/dataset_report.py` — `report dataset`.
- `smartrain/workflows/models/model_convert_cli.py`, `model_release_cli.py`, and `model_rename_cli.py` — `model`.
- `smartrain/workflows/datasets/data_yaml_normalize.py` — `normalize-data-yaml`.
- `smartrain/workflows/migration/migrate_models_to_smartrain.py` — `migrate-models`.
- `smartrain/workflows/analyze/clearml_upload.py` — `clearml-upload`.

## CLI mapping -> module

| CLI command | Module |
|---|---|
| `smartrain scan` | `smartrain/workflows/datasets/datasets_json_former.py` |
| `smartrain normalize-data-yaml` | `smartrain/workflows/datasets/data_yaml_normalize.py` |
| `smartrain fusion` | `smartrain/workflows/datasets/dataset_former.py` |
| `smartrain augment` | `smartrain/workflows/datasets/dataset_augment.py` |
| `smartrain balance` | `smartrain/workflows/datasets/dataset_balance.py` |
| `smartrain prune` | `smartrain/workflows/datasets/dataset_prune.py` |
| `smartrain roi` | `smartrain/workflows/datasets/dataset_roi_yolo.py` |
| `smartrain orient` | `smartrain/workflows/datasets/dataset_orient.py` |
| `smartrain rotate` | `smartrain/workflows/datasets/dataset_rotate.py` |
| `smartrain stats` | `smartrain/workflows/datasets/dataset_stats.py` |
| `smartrain hash` | `smartrain/workflows/datasets/dataset_hash.py` |
| `smartrain train` | `smartrain/workflows/training/train_entry.py` (`train_wiring.py`; CLI in `services/training/train_cli_main.py`) |
| `smartrain inference` | `smartrain/workflows/inference/inference_cli.py` |
| `smartrain report dataset` | `smartrain/workflows/datasets/dataset_report.py` |
| `smartrain analyze` | `smartrain/workflows/analyze/results_analyzer.py` |
| `smartrain plot` | `smartrain/workflows/analyze/plot_creator.py` |
| `smartrain queue` / `queue-run` | `smartrain/workflows/queue/training_queue_cli.py` / `smartrain/workflows/queue/training_queue.py` |
| `smartrain registry` | `smartrain/workflows/registry/registry_cli.py` |
| `smartrain model convert` / `model release` / `model rename` | `smartrain/workflows/models/model_convert_cli.py` / `model_release_cli.py` / `model_rename_cli.py` |
| `smartrain migrate-models` | `smartrain/workflows/migration/migrate_models_to_smartrain.py` |
| `smartrain clearml-upload` | `smartrain/workflows/analyze/clearml_upload.py` |
| `smartrain cvat` | `smartrain/workflows/datasets/cvat_cli.py` |
| `smartrain sahi` | `smartrain/workflows/inference/sahi_cli.py` |
| `smartrain heatmap` | `smartrain/workflows/inference/heatmap_cli.py` |

## Actual behavior notes

- `hash --validate`: `0` (match), `1` (mismatch), `2` (error).
- Extended subcommands are available in `analyze`: `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`.
- The queue in the workspace uses `queue.txt` by default.

## Module map diagram

```mermaid
flowchart LR
    cliRouter["smartrain/cli.py"]
    cliRouter --> datasetsBlock["workflows/datasets/*"]
    cliRouter --> trainBlock["train_entry.py"]
    cliRouter --> analyzeBlock["results_analyzer.py"]
    cliRouter --> queueBlock["workflows/queue/training_queue_cli.py + training_queue.py"]
    cliRouter --> registryBlock["workflows/registry/registry_cli.py"]
    cliRouter --> modelBlock["workflows/models/model_convert_cli.py + workflows/models/model_release_cli.py"]
    cliRouter --> ioBlock["workflows/datasets/cvat_cli.py + workflows/inference/sahi_cli.py + heatmap_cli.py"]
    cliRouter --> reportBlock["workflows/datasets/dataset_report.py"]
```

For detailed command examples, see [CLI section](../cli/overview.md).
