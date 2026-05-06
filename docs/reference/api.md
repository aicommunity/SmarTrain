> Russian version: [../ru/reference/api.md](../ru/reference/api.md)

# Reference: APIs and modules

## Entry point

- `smartrain/cli.py` — Typer-router command.
- `smartrain/cli_argparse.py` is a basic argparse parser with default values ​​in `--help`.

## Basic modules

- `smartrain/workflows/datasets/datasets_json_former.py` — `scan`.
- `smartrain/dataset_former.py` — `fusion`.
- `smartrain/workflows/training/model_training_module.py` — `train`.
- `smartrain/training_queue.py` and `smartrain/training_queue_cli.py` are the queue.
- `smartrain/workflows/analyze/results_analyzer.py` — `analyze`.
- `smartrain/registry_cli.py` — `registry`.
- `smartrain/dataset_hash.py` — `hash`.
- `smartrain/workflows/inference/inference_cli.py` — `inference`.
- `smartrain/dataset_report.py` — `report dataset`.
- `smartrain/workflows/models/model_convert_cli.py` and `smartrain/workflows/models/model_release_cli.py` — `model`.
- `smartrain/data_yaml_normalize.py` — `normalize-data-yaml`.
- `smartrain/workflows/migration/migrate_models_to_smartrain.py` — `migrate-models`.
- `smartrain/workflows/analyze/clearml_upload.py` — `clearml-upload`.

## CLI mapping -> module

| CLI command | Module |
|---|---|
| `smartrain scan` | `smartrain/workflows/datasets/datasets_json_former.py` |
| `smartrain normalize-data-yaml` | `smartrain/data_yaml_normalize.py` |
| `smartrain fusion` | `smartrain/dataset_former.py` |
| `smartrain augment` | `smartrain/dataset_augment.py` |
| `smartrain balance` | `smartrain/dataset_balance.py` |
| `smartrain prune` | `smartrain/dataset_prune.py` |
| `smartrain roi` | `smartrain/dataset_roi_yolo.py` |
| `smartrain orient` | `smartrain/dataset_orient.py` |
| `smartrain stats` | `smartrain/dataset_stats.py` |
| `smartrain hash` | `smartrain/dataset_hash.py` |
| `smartrain train` | `smartrain/workflows/training/model_training_module.py` |
| `smartrain inference` | `smartrain/workflows/inference/inference_cli.py` |
| `smartrain report dataset` | `smartrain/dataset_report.py` |
| `smartrain analyze` | `smartrain/workflows/analyze/results_analyzer.py` |
| `smartrain plot` | `smartrain/workflows/analyze/plot_creator.py` |
| `smartrain queue` / `queue-run` | `smartrain/training_queue_cli.py` / `smartrain/training_queue.py` |
| `smartrain registry` | `smartrain/registry_cli.py` |
| `smartrain model convert` / `model release` | `smartrain/workflows/models/model_convert_cli.py` / `smartrain/workflows/models/model_release_cli.py` |
| `smartrain migrate-models` | `smartrain/workflows/migration/migrate_models_to_smartrain.py` |
| `smartrain clearml-upload` | `smartrain/workflows/analyze/clearml_upload.py` |
| `smartrain cvat` | `smartrain/cvat_cli.py` |
| `smartrain sahi` | `smartrain/sahi_cli.py` |
| `smartrain heatmap` | `smartrain/heatmap_cli.py` |

## Actual behavior notes

- `hash --validate`: `0` (match), `1` (mismatch), `2` (error).
- Extended subcommands are available in `analyze`: `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`.
- The queue in the workspace uses `queue.txt` by default.

## Module map diagram

```mermaid
flowchart LR
    cliRouter["smartrain/cli.py"]
    cliRouter --> datasetsBlock["dataset commands modules"]
    cliRouter --> trainBlock["model_training_module.py"]
    cliRouter --> analyzeBlock["results_analyzer.py"]
    cliRouter --> queueBlock["training_queue_cli.py + training_queue.py"]
    cliRouter --> registryBlock["registry_cli.py"]
    cliRouter --> modelBlock["workflows/models/model_convert_cli.py + workflows/models/model_release_cli.py"]
    cliRouter --> ioBlock["cvat_cli.py + sahi_cli.py + heatmap_cli.py"]
    cliRouter --> reportBlock["dataset_report.py"]
```

For detailed command examples, see [CLI section](../cli/overview.md).
