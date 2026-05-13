> English version: [../../development/package-layout.md](../../development/package-layout.md)

# Карта каталогов пакета SmarTrain

Краткий обзор `smartrain/` для навигации **по функционалу**. Подробности и статус волн рефакторинга: [../../refactor/13-project-current-state.md](../../refactor/13-project-current-state.md).

## Точки входа

- `cli.py` — маршрутизация Typer; делегирование в argparse-модули или `cli_apps/*`.
- `__main__.py`, `__init__.py` — вход пакета.

## workflows/

Пользовательские CLI-сценарии: `main()` с argparse, связка с Typer, пайплайны датасетов / обучения / теста / inference / analyze.

Подпакеты:

- `training/` — обучение (`train_entry`, `model_training_module`, `train_*_service`).
- `datasets/` — scan, fusion, augment, balance, prune, orient, отчёты, CVAT; `dataset_access.py` для работы с файловой структурой; `dataset_cli_catalog.py` / `dataset_cli_common.py` для каталога и интерактивного выбора датасета.
- `testing/` — CLI и backend-и model test.
- `inference/` — inference CLI, backends, SAHI/heatmap.
- `analyze/` — подкоманды analyze, сервисы `analyze_*_service.py`, сборка парсера в `results_analyzer.py`.
- `queue/` — очередь обучения (`training_queue.py`).
- `registry/` — реестр.
- `migration/` — миграции.
- `models/` — convert / release.

**Правило:** новая или изменённая CLI-обвязка — сначала здесь (`build_*_arg_parser`, `main(argv)`).

## services/

Переиспользуемая бизнес-логика без привязки к одному скрипту. **Не** импортирует `smartrain.workflows.*` (доступ через `core/workflow_adapters/`).

## core/

Общая механика: `runtime/`, `training/`, `workflow_adapters/`, `inference/`.

## orchestrators/

- `canonical_gateway.py` — чтение canonical, метрики, контекст задачи.

## domain/ и adapters/

- `domain/canonical/` — модели данных и валидация.
- `adapters/canonical/` — чтение/запись и снимки.

## backends/

Контракты backend, реестр возможностей, адаптеры Ultralytics и внешних провайдеров.

## tasks/

Адаптеры метрик по типу задачи (детекция / классификация / сегментация).

## cli_support/

`cli_replay.py`, `cli_contracts.py`, общие промпты argparse.

## external_providers/

Лончеры subprocess и установка внешних провайдеров.

## providers/

CLI и индекс опциональных внешних провайдеров (`providers/cli.py`, `providers/core/global_index.py`).
