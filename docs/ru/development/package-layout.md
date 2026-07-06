> English version: [../../development/package-layout.md](../../development/package-layout.md)

# Карта каталогов пакета SmarTrain

Краткий обзор `smartrain/` для навигации **по функционалу**. Подробности и статус волн рефакторинга: [../../refactor/13-project-current-state.md](../../refactor/13-project-current-state.md).

## Точки входа

- `cli.py` — маршрутизация Typer; делегирование в argparse-модули или `cli_entrypoints/*`.
- `__main__.py`, `__init__.py` — вход пакета.

## workflows/

Тонкие CLI-фасады: argparse `main()`, связка с Typer, делегирование в `services/`. Бизнес-логика — в `services/`, не здесь.

Подпакеты (соответствие командам):

- `training/` — train CLI (`train_entry`, `train_wiring` для resume/calc-confidence; исполнение в `services/training/`).
- `datasets/` — фасады (`dataset_former.py`, `datasets_json_former.py`, …) → `services/datasets/`.
- `testing/` — `model_test_cli.py`, `model_test_backends.py` (фасад) → `services/testing/backends/`.
- `inference/` — inference CLI, SAHI/heatmap; runtime в `services/inference_service.py`.
- `analyze/` — `results_analyzer.py` (фасад) → `services/analyze/cli_commands.py`.
- `queue/` — очередь обучения (`training_queue.py`).
- `registry/` — registry CLI.
- `migration/` — migration CLI.
- `models/` — convert / release CLI.

**Правило:** новая CLI-обвязка — сначала здесь (`build_*_arg_parser`, `main(argv)`).

## services/

Use-case слой: `analyze/`, `datasets/`, `training/`, `testing/`, `inference_service.py`, `reporting/`. **Запрещены** прямые импорты `smartrain.workflows.*` (доступ через `core/workflow_adapters/`).

Важный helper analyze: `data_yaml_splits.py` (разрешение путей датасета и split для benchmark/PR).

## core/

Общая механика: `runtime/` (workspace, env), `training/` (профили, каталоги), `workflow_adapters/` (фасады к workflows для services), `inference/` (общие helpers inference).

## Контракт run/model (`smartrain/run_model_contract/`)

Чтение legacy-раскладок, schema v2, API через `gateway`, опциональные снимки в `.smartrain/unified/`.

| Путь | Роль |
|------|------|
| `run_model_contract/gateway.py` | `load_target`, `load_metrics`, `resolve_task_context`, predictions API |
| `run_model_contract/domain/` | DTO (`UnifiedPayload`, `UnifiedIdentity`) и валидация |
| `run_model_contract/io/` | Адаптеры чтения/записи, legacy mapper, snapshot hook |
| `run_model_contract/refs.py` | Путь к модели из run или каталога модели |
| `run_model_contract/schema.py` | Artifact schema v2 |
| `run_model_contract/env.py` | `SMARTTRAIN_UNIFIED_WRITE` и режим dual-write |

На диске: `{run|model}/.smartrain/unified/` (fallback чтения legacy: `.smartrain/canonical/`).

## Глоссарий: backend vs алиас модели

| Термин | Где | Смысл |
|--------|-----|--------|
| **Execution backend** | `backends/contracts.py` | Движок runtime (ultralytics, onnxruntime, tensorrt) |
| **Ultralytics model alias** | `core/training/ultralytics_model_alias_registry.py` | Алиасы YAML YOLO (`yolo11n`, …), не ONNX/TRT |

## backends/

Контракты backend, реестр возможностей, адаптеры Ultralytics и внешних провайдеров.

## tasks/

Адаптеры метрик по типу задачи (детекция / классификация / сегментация).

## cli_entrypoints/

Тонкие Typer-приложения (`train_app.py`, `test_app.py`, …) и `support/` (`cli_replay.py`, `cli_contracts.py`, argparse, `--nit`).

## external_providers/

Лончеры subprocess и установка внешних провайдеров.

## providers/

CLI и индекс опциональных внешних провайдеров (`providers/cli.py`, `providers/core/global_index.py`).
