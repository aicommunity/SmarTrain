> English version: [../../cli/overview.md](../../cli/overview.md)

# CLI: обзор

Точка входа: `smartrain` (Typer-роутер с единым поведением команд).

## Группы команд

- Датасеты: `scan`, `fusion`, `augment`, `balance`, `orient`, `roi`, `hash`, `stats`
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
- для `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient` пустой вызов запускает интерактивный режим;
- если переданы любые аргументы, но их недостаточно, команда завершится понятной ошибкой о неполных аргументах (без prompt-режима).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.

Дополнения для балансировки и статистики:

- `smartrain balance` поддерживает стратегии `weights`, `rfs`, `hybrid` и параметры их настройки.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` применяет готовые настройки под типовые сценарии.
- `smartrain stats --balance-ready` выводит метрики дисбаланса и рекомендации для балансировщика.
