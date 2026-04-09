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

## Соответствие CLI -> модуль

| CLI команда | Модуль |
|---|---|
| `smartrain scan` | `smartrain/datasets_json_former.py` |
| `smartrain fusion` | `smartrain/dataset_former.py` |
| `smartrain train` | `smartrain/model_training_module.py` |
| `smartrain analyze` | `smartrain/results_analyzer.py` |
| `smartrain queue` / `queue-run` | `smartrain/training_queue_cli.py` / `smartrain/training_queue.py` |
| `smartrain registry` | `smartrain/registry_cli.py` |

## Актуальные примечания по поведению

- `hash --validate`: `0` (совпадение), `1` (несовпадение), `2` (ошибка).
- В `analyze` доступны расширенные подкоманды: `pr-curves`, `inference-benchmark`, `inference-plot`.
- Очередь в рабочем каталоге по умолчанию использует `queue.txt`.

Подробные сценарии запуска см. в [CLI-разделе](../cli/overview.md).
