> English version: [../../reference/api.md](../../reference/api.md)

# Справочник: API и модули

## Точка входа

- `smartrain/cli.py` — Typer-роутер команд.
- `smartrain/cli_argparse.py` — базовый argparse-парсер со значениями по умолчанию в `--help`.

## Основные модули

- `smartrain/datasets_json_former.py` — `scan`.
- `smartrain/dataset_former.py` — `fusion`.
- `smartrain/model_training_module.py` — `train`.
- `smartrain/training_queue.py` и `smartrain/training_queue_cli.py` — очередь.
- `smartrain/results_analyzer.py` — `analyze`.
- `smartrain/registry_cli.py` — `registry`.
- `smartrain/dataset_hash.py` — `hash`.
- `smartrain/inference_cli.py` — `inference`.
- `smartrain/dataset_report.py` — `report dataset`.
- `smartrain/workflows/models/model_convert_cli.py` и `smartrain/workflows/models/model_release_cli.py` — `model`.
- `smartrain/data_yaml_normalize.py` — `normalize-data-yaml`.
- `smartrain/workflows/migration/migrate_models_to_smartrain.py` — `migrate-models`.
- `smartrain/workflows/analyze/clearml_upload.py` — `clearml-upload`.

## Соответствие CLI -> модуль

| CLI команда | Модуль |
|---|---|
| `smartrain scan` | `smartrain/datasets_json_former.py` |
| `smartrain normalize-data-yaml` | `smartrain/data_yaml_normalize.py` |
| `smartrain fusion` | `smartrain/dataset_former.py` |
| `smartrain augment` | `smartrain/dataset_augment.py` |
| `smartrain balance` | `smartrain/dataset_balance.py` |
| `smartrain prune` | `smartrain/dataset_prune.py` |
| `smartrain roi` | `smartrain/dataset_roi_yolo.py` |
| `smartrain orient` | `smartrain/dataset_orient.py` |
| `smartrain stats` | `smartrain/dataset_stats.py` |
| `smartrain hash` | `smartrain/dataset_hash.py` |
| `smartrain train` | `smartrain/model_training_module.py` |
| `smartrain inference` | `smartrain/inference_cli.py` |
| `smartrain report dataset` | `smartrain/dataset_report.py` |
| `smartrain analyze` | `smartrain/results_analyzer.py` |
| `smartrain plot` | `smartrain/workflows/analyze/plot_creator.py` |
| `smartrain queue` / `queue-run` | `smartrain/training_queue_cli.py` / `smartrain/training_queue.py` |
| `smartrain registry` | `smartrain/registry_cli.py` |
| `smartrain model convert` / `model release` | `smartrain/workflows/models/model_convert_cli.py` / `smartrain/workflows/models/model_release_cli.py` |
| `smartrain migrate-models` | `smartrain/workflows/migration/migrate_models_to_smartrain.py` |
| `smartrain clearml-upload` | `smartrain/workflows/analyze/clearml_upload.py` |
| `smartrain cvat` | `smartrain/cvat_cli.py` |
| `smartrain sahi` | `smartrain/sahi_cli.py` |
| `smartrain heatmap` | `smartrain/heatmap_cli.py` |

## Актуальные примечания по поведению

- `hash --validate`: `0` (совпадение), `1` (несовпадение), `2` (ошибка).
- В `analyze` доступны расширенные подкоманды: `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`.
- Очередь в рабочем каталоге по умолчанию использует `queue.txt`.

## Схема модулей

```mermaid
flowchart LR
    cliRouter["smartrain/cli.py"]
    cliRouter --> datasetsBlock["модули команд датасетов"]
    cliRouter --> trainBlock["model_training_module.py"]
    cliRouter --> analyzeBlock["results_analyzer.py"]
    cliRouter --> queueBlock["training_queue_cli.py + training_queue.py"]
    cliRouter --> registryBlock["registry_cli.py"]
    cliRouter --> modelBlock["workflows/models/model_convert_cli.py + workflows/models/model_release_cli.py"]
    cliRouter --> ioBlock["cvat_cli.py + sahi_cli.py + heatmap_cli.py"]
    cliRouter --> reportBlock["dataset_report.py"]
```

Подробные сценарии запуска см. в [CLI-разделе](../cli/overview.md).
