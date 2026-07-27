> Russian version: [../ru/reference/api.md](../ru/reference/api.md)

# Reference: APIs and modules

## Entry point

- `smartrain/cli.py` — Typer-router command.
- `smartrain/cli_entrypoints/support/cli_argparse.py` — shared argparse helpers (e.g. `CliArgumentParser`).

## Basic modules

- `smartrain/cli.py` — top-level Typer router and command groups.
- `smartrain/cli_entrypoints/cli_forwarding.py` — bridge from Typer wrappers to argparse parsers.
- `smartrain/workflows/queue/training_queue_cli.py` and `smartrain/workflows/queue/training_queue.py` — queue (`queue` and `queue-run`).
- `smartrain/workflows/registry/registry_cli.py` — `registry`.
- `smartrain/workflows/analyze/analyze_entry.py` and `smartrain/workflows/analyze/plot_creator.py` — `analyze` and legacy `plot`.
- `smartrain/workflows/models/model_convert_cli.py`, `model_release_cli.py`, `model_unrelease_cli.py`, `model_comment_cli.py`, `model_rename_cli.py` — `model` group.
- `smartrain/providers/cli.py` — `providers`.

## CLI mapping -> module

| CLI command | Module |
|---|---|
| `smartrain scan` | `smartrain/workflows/datasets/datasets_entry.py` |
| `smartrain normalize-data-yaml` | `smartrain/services/datasets/data_yaml_normalize.py` |
| `smartrain merge` / `smartrain fusion` | `smartrain/workflows/datasets/dataset_former.py` |
| `smartrain augment` | `smartrain/workflows/datasets/dataset_augment.py` |
| `smartrain balance` | `smartrain/workflows/datasets/dataset_balance.py` |
| `smartrain prune` | `smartrain/workflows/datasets/dataset_prune.py` |
| `smartrain filter` | `smartrain/workflows/datasets/dataset_filter.py` |
| `smartrain roi` | `smartrain/workflows/datasets/dataset_roi_yolo.py` |
| `smartrain orient` | `smartrain/workflows/datasets/dataset_orient.py` |
| `smartrain rotate` | `smartrain/workflows/datasets/dataset_rotate.py` |
| `smartrain stats` | `smartrain/services/datasets/dataset_stats.py` |
| `smartrain hash` | `smartrain/services/datasets/dataset_hash.py` |
| `smartrain train` | `smartrain/cli_entrypoints/train_app.py` |
| `smartrain test` | `smartrain/cli_entrypoints/test_app.py` |
| `smartrain inference` | `smartrain/cli_entrypoints/inference_app.py` |
| `smartrain vis` | `smartrain/workflows/visualization/vis_cli.py` |
| `smartrain dataset report` | `smartrain/services/datasets/dataset_report.py` |
| `smartrain dataset rename` | `smartrain/workflows/datasets/dataset_rename_cli.py` |
| `smartrain dataset convert` | `smartrain/workflows/datasets/dataset_convert_cli.py` |
| `smartrain analyze` | `smartrain/workflows/analyze/analyze_entry.py` |
| `smartrain plot` | `smartrain/workflows/analyze/plot_creator.py` |
| `smartrain queue` / `queue-run` | `smartrain/workflows/queue/training_queue_cli.py` / `smartrain/workflows/queue/training_queue.py` |
| `smartrain registry` | `smartrain/workflows/registry/registry_cli.py` |
| `smartrain model convert` / `model release` / `model unrelease` / `model comment` / `model rename` | `smartrain/workflows/models/model_convert_cli.py` / `model_release_cli.py` / `model_unrelease_cli.py` / `model_comment_cli.py` / `model_rename_cli.py` |
| `smartrain providers` | `smartrain/providers/cli.py` |
| `smartrain deps sync-torch` | `smartrain/external_providers/installer.py` |
| `smartrain deps doctor` / `deps install` | `smartrain/services/deps/optional_extras.py` |
| `smartrain deploy` / `quickstart` / `info` / `sync` / `update` | `smartrain/cli.py` (+ `workspace_sync_service.py` for `sync`; `workflows/update/update_cli.py` for `update`) |
| `smartrain migrate` | `smartrain/workflows/migration/cli_migration.py` |
| `smartrain migrate-models` | `smartrain/workflows/migration/migrate_models_to_smartrain.py` |
| `smartrain clearml-upload` | `smartrain/workflows/analyze/clearml_upload.py` |
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
    cliRouter --> ioBlock["workflows/datasets/dataset_convert_cli.py + workflows/inference/sahi_cli.py + heatmap_cli.py"]
    cliRouter --> reportBlock["workflows/datasets/dataset_report.py"]
```

For detailed command examples, see [CLI section](../cli/overview.md).
