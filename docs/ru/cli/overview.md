> English version: [../../cli/overview.md](../../cli/overview.md)

# CLI: обзор

Точка входа: `smartrain` (Typer-роутер с единым поведением команд).

## Группы команд

- Датасеты: `scan`, `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`, `hash`, `stats`, `report dataset`
- Обучение: `train`, `clearml-upload`
- Очередь: `queue`, `queue-run`
- Аналитика: `analyze`, `plot` (устаревшая обёртка)
- Реестр: `registry`
- Инструменты форматов: `cvat`, `sahi`, `heatmap`

## Справка

```bash
smartrain --help
smartrain <команда> --help
```

Для вложенных команд:

```bash
smartrain queue list --help
smartrain analyze inference-benchmark --help
```

Единый контракт интерактива:

- интерактив включается только при запуске команды без аргументов (TTY обязателен);
- выбор датасета(ов): сразу нумерованный список; ввод по имени или по номеру (несколько датасетов — через CSV номеров или имён);
- для `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `report dataset` пустой вызов запускает интерактивный режим;
- если переданы любые аргументы, но их недостаточно, команда завершится понятной ошибкой о неполных аргументах (без prompt-режима).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.

Дополнения для балансировки и статистики:

- `smartrain balance` поддерживает стратегии `weights`, `rfs`, `hybrid` и параметры их настройки.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` применяет готовые настройки под типовые сценарии.
- `smartrain balance --eval-coverage` (по умолчанию включено) подстраивает пул train после балансировки: по возможности не оставлять пустыми `val`/`test` и донаполнять в eval отсутствующие классы из train, при этом один и тот же source-кадр не распределяется между разными сплитами; если уникальных кадров не хватает, `val/test` могут быть заполнены не полностью; отключение — `--no-eval-coverage`. В интерактивном `balance` тот же выбор задаётся вопросом.
- `smartrain stats --balance-ready` выводит метрики дисбаланса и рекомендации для балансировщика.
- `smartrain prune empty` удаляет пустые пары image/label в новый датасет `<dataset>_pruned`.
- `smartrain prune dedup` удаляет дубли изображений по содержимому в `<dataset>_deduped` (глобальный приоритет split: train > val > test).
- `smartrain report dataset` формирует многоязычный отчёт с примерами по классам (Markdown + PNG; по умолчанию `analytics/datasets-reports/<dataset>_<timestamp>/`). В базовые зависимости входят pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2` и `odfpy` для PDF/ODT. Для WeasyPrint на некоторых ОС могут понадобиться системные библиотеки (Cairo, Pango), если нет подходящего wheel.
