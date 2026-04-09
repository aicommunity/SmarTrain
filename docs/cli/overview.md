# CLI: обзор

Точка входа: `smartrain` (Typer-роутер + делегирование в argparse-модули).

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
