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

Вызов `smartrain <команда>` без обязательных аргументов обычно выводит справку этой команды.
Исключение: `smartrain train` без аргументов запускает интерактивную настройку (нужен TTY).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.
