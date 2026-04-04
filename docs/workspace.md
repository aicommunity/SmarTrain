# Единый workspace

Корень задаётся **`--workspace`** (глобальная опция Typer у `smartrain`) или переменной **`SMART_TRAIN_WORKSPACE`**. В коде модулей при вызове `resolve_workspace_root()` **непустой** аргумент CLI перекрывает env; если оба пусты — **ValueError** (кроме сценариев без workspace, например legacy-пути у `fusion`).

При запуске CLI, если переменная окружения не задана, Typer может выставить **`SMART_TRAIN_WORKSPACE`** в текущий каталог — см. [`smartrain/cli.py`](../smartrain/cli.py).

## Каталоги

| Путь | Назначение |
|------|------------|
| `source_datasets/datasets_info.json` | Описание исходных датасетов |
| `source_datasets/class_names.json` | Нормализация имён классов |
| `work_datasets/` | Рабочие датасеты; `fusion` пишет данные в `work_datasets/<имя>/` (по умолчанию `<YYYY-MM-DD_HH-MM-SS>-merged`, иначе `--output-name`) и обновляет `work_datasets/datasets_info.json` |
| `runs/` | Прогоны обучения (`train` / `model_training_module`) |
| `analytics/` | Сессии анализа (`analyze export-table` и др. с `--analytics-session`) |
| `models/` | Промотированные веса (`registry models-add`, …) |
| `tmp/` | В т.ч. `tmp/status.txt` для исполнителя очереди |
| `queue.txt` (в корне workspace) | Файл очереди обучения по умолчанию |

Создание каталогов и пустых JSON: **`smartrain deploy`** → [`deploy_workspace()`](../smartrain/workspace_paths.py).

## Поле `data_path` в `datasets_info.json`

Строка: **абсолютный** путь или путь **относительно корня workspace**. Если ключа нет, корень данных = `source_datasets/<ключ_записи>` или `work_datasets/<ключ>` для каталога work.

## Код

- [`smartrain/workspace_paths.py`](../smartrain/workspace_paths.py) — `WorkspaceLayout`, `resolve_workspace_root`, `resolve_dataset_root`, `deploy_workspace`, пути к `queue.txt` и `tmp/status.txt`
- [`smartrain/registry_cli.py`](../smartrain/registry_cli.py) — подкоманды `runs-list`, `runs-info`, `runs-metrics`, `models-add`, `models-list`, `models-info`, `models-remove`
