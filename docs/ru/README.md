> English version: [../../README.md](../../README.md)

# Smart Train (`smartrain`)

CLI-набор для подготовки YOLO-датасетов, обучения моделей, исполнения очередей и анализа прогонов.

## Быстрый старт

Требования: Python `3.10+`.

```bash
git clone <repo-url>
cd smart-train
pip install -e .
```

Работа с рабочим каталогом (workspace):

`SMART_TRAIN_WORKSPACE` — опционален. Если переменная не задана, `smartrain` использует текущий каталог как корень workspace.

```bash
smartrain deploy
smartrain scan
smartrain fusion --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data 2026-01-01_12-00-00-merged -y
```

Явное указание корня workspace (опционально):

```bash
export SMART_TRAIN_WORKSPACE=/path/to/workspace
smartrain deploy
```

## Что внутри

- Единая точка входа: `smartrain` (модуль `smartrain.cli`).
- Модель единого рабочего каталога: `raw_data/`, `datasets/`, `runs/`, `analytics/`, `models/`, `tmp/`.
- Поддержка конвейера: `scan -> fusion -> train -> analyze`.
- Отдельные инструменты: `queue`, `registry`, `cvat`, `sahi`, `heatmap`, `orient`.

## Принцип работы

`smartrain` использует единый корень workspace и строит процесс вокруг файловых контрактов:

- `scan` синхронизирует источники и обновляет каталог датасетов;
- `fusion` формирует итоговый датасет под обучение;
- `train` создаёт run-каталог с метриками и метаданными;
- `analyze` и `registry` работают по артефактам в `runs/`.

## Ключевые команды

| Команда | Назначение |
|---|---|
| `smartrain deploy` | Инициализация структуры workspace |
| `smartrain scan` | Синхронизация источников и обновление каталога датасетов |
| `smartrain fusion` | Сборка итогового датасета для обучения |
| `smartrain train` | Обучение/валидация модели YOLO |
| `smartrain queue` / `smartrain queue-run` | Управление и запуск очереди команд |
| `smartrain analyze` | Сводки, сравнение запусков, PR-кривые, бенчмарк инференса |
| `smartrain registry` | Каталогизация артефактов запусков и промо моделей |

## Документация

Актуальная документация организована по разделам в `docs/`:

- [Навигация по документации](index.md)
- [Старт и базовые сценарии](getting-started/quickstart.md)
- [CLI-руководство](cli/overview.md)
- [Справочник форматов и API](reference/api.md)
- [Архитектура и диаграммы](development/architecture.md)

## Важные детали

- Интерактивный режим включается только если команда запущена вообще без аргументов (нужен TTY).
- Интерактивные команды по датасетам: `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, а также `train`.
- Очистка датасетов: `prune` (`prune empty` для удаления пустых пар, `prune dedup` для удаления дублей изображений по содержимому).
- Если аргументы переданы частично и обязательных не хватает, команда выводит понятную ошибку о неполных аргументах и не уходит в prompt-режим.
- В справке команд есть практические блоки `Examples` / `Quick examples` для типовых сценариев.
- Пресеты `smartrain balance`:
  - `--preset weights-safe` для консервативной балансировки
  - `--preset rfs-aggressive` для более агрессивного усиления tail-классов
  - `--preset hybrid-default` как универсальный дефолт
- Для `hash --validate`: `0` при совпадении, `1` при несовпадении, `2` при ошибке.
- По умолчанию очередь workspace использует `queue.txt` и `tmp/status.txt`.
- Расширения зависимостей:
  - `pip install -e ".[dev]"` для разработки и тестов
  - `pip install -e ".[clearml]"` для ClearML
  - `pip install -e ".[sahi]"` для SAHI

## Частые сценарии

Сканирование с явным списком источников:

```bash
smartrain scan --datasets-list /path/to/workspace/raw_data/datasets_list.txt
```

Проверка хеша датасета:

```bash
smartrain hash --dataset my_dataset
smartrain hash /path/to/dataset --validate a1b2c3d4
```

Запуск очереди без открытия GUI-терминала:

```bash
smartrain queue run --no-gui
```

Быстрый просмотр запусков:

```bash
smartrain analyze scan
smartrain analyze export-table -o runs_summary.csv
```
