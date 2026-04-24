> English version: [../../providers/overview.md](../../providers/overview.md)

# Обзор провайдеров

Внешние провайдеры — это fork-специфичные подсистемы обучения/инференса, подключённые через общий слой адаптеров и раннеров.

## Поддерживаемые провайдеры

- `dr-yolo`
- `leaf-yolo`
- `mfel-yolo`
- `mp-yolo`
- `ssdm-yolo`
- `enhanced-yolov8`

## Слои интеграции

- Registry: `smartrain/external_providers/registry.py`
  - описание id провайдера, URL/branch репозитория, номинальных entrypoint-скриптов.
- Installer: `smartrain/external_providers/installer.py`
  - клонирование репозитория, создание provider-venv, установка зависимостей, обновление глобального индекса.
- Global index: `smartrain/provider_global_index.py`
  - хранение местоположения провайдера (`repo_path`, `venv_path`, state, диагностика).
- Adapters: `smartrain/external_providers/adapters.py`
  - отображение нормализованных аргументов Smart Train в аргументы launcher-скриптов.
- Runners: `smartrain/external_providers/runner.py`
  - запуск launcher-скриптов в `venv` провайдера.
- Launchers: `smartrain/external_providers/launchers/*.py`
  - провайдер-ориентированные обёртки над runtime API форков.

## Контракт runtime-артефактов

Для любого внешнего провайдера Smart Train приводит структуру run к контракту:

- `train/weights/best.pt`
- каталог `test/`
- `test_metrics.csv`
- `training_metadata.json`

Этот контракт нужен для совместимости `analyze`, `registry` и автоматизаций поверх run-артефактов.
