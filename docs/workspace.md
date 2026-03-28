# Единый workspace

Корень задаётся **`--workspace`** или переменной **`SMART_TRAIN_WORKSPACE`** (CLI перекрывает env). Если ни то ни другое не задано — скрипты, которым нужен workspace, завершатся с ошибкой.

## Каталоги

| Путь | Назначение |
|------|------------|
| `source_datasets/datasets_info.json` | Описание исходных датасетов |
| `source_datasets/class_names.json` | Нормализация имён классов |
| `work_datasets/` | Рабочие датасеты; `dataset_former` пишет пиксели в `work_datasets/<имя>/` и обновляет `work_datasets/datasets_info.json` |
| `runs/` | Прогоны обучения (`model_training_module`) |
| `analytics/` | Сессии экспорта (`results_analyzer export-table --analytics-session <id>`) |
| `models/` | Промотированные веса (`registry_cli.py models-add`) |

## Поле `data_path` в `datasets_info.json`

Строка: **абсолютный** путь или путь **относительно корня workspace**. Если ключа нет, корень данных = `source_datasets/<ключ_записи>` (или `work_datasets/<ключ>` для каталога work).

## Скрипты

- [workspace_paths.py](../workspace_paths.py) — `WorkspaceLayout`, `resolve_workspace_root`, `resolve_dataset_root`
- [registry_cli.py](../registry_cli.py) — подкоманды `runs-list`, `runs-info`, `runs-metrics`, `models-add`, `models-list`, `models-info`, `models-remove`
