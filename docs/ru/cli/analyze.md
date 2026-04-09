> English version: [../../cli/analyze.md](../../cli/analyze.md)

# CLI: анализ запусков

`smartrain analyze` работает по каталогам артефактов запусков.

Базовый корень поиска запусков: каталог `runs` внутри рабочего каталога (или путь из `--models-root`).

## Подкоманды

- `scan`
- `export-table`
- `compare`
- `interactive`
- `pr-curves`
- `inference-benchmark`
- `inference-plot`

## Примеры

```bash
smartrain analyze scan
smartrain analyze export-table -o runs_summary.csv
smartrain analyze compare --baseline /path/to/run_a --others /path/to/run_b /path/to/run_c --out-csv cmp.csv
smartrain analyze pr-curves --run /path/to/run
smartrain analyze inference-benchmark --model /path/to/best.pt --source /path/to/images
smartrain analyze inference-plot --csv benchmark.csv --out benchmark.png
```

## Артефакты

- `export-table` формирует сводный CSV по найденным прогонам.
- `compare` может создавать таблицу сравнения и PNG-графики.
- `inference-benchmark` формирует CSV с измерениями инференса.
- `inference-plot` строит визуализацию на основе CSV из `inference-benchmark`.

`smartrain plot` остаётся устаревшей обёрткой и делегирует в `analyze`.
