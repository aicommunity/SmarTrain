# SmarTrain Analyze: подкоманды, артефакты и зависимости

Этот документ описывает:
- какие артефакты создаёт каждая подкоманда `smartrain analyze`,
- как подкоманды связаны между собой,
- как их использует корневая команда `smartrain analyze`,
- практическую шпаргалку по запуску.

## Поведение корневой команды

- `smartrain analyze` (без подкоманды, в TTY) запускает интерактивный orchestrator, эквивалентный `smartrain analyze all`.
- `smartrain analyze` без подкоманды в неинтерактивном режиме завершится ошибкой.
- Основная orchestration-логика реализована в `all`.

## Подкоманды и создаваемые артефакты

### `smartrain analyze scan`

- Назначение: показать найденные run.
- Артефакты: не создаёт.

### `smartrain analyze export-table`

- Назначение: экспорт сводной таблицы по run.
- Артефакты:
  - `runs_summary.csv` (или путь из `-o/--output`).

### `smartrain analyze compare`

- Назначение: сравнение baseline run с остальными.
- Артефакты:
  - `compare_delta.csv`,
  - `compare_insights.txt`,
  - `compare_curves.png`,
  - `compare_curves_bars.png` (если есть достаточно данных).

### `smartrain analyze leaderboard`

- Назначение: ранжирование run по composite score.
- Артефакты:
  - `leaderboard.csv`.

### `smartrain analyze pr-curves`

- Назначение: PR-анализ по группе run.
- Артефакты:
  - per-run: `run/test/pr.csv`,
  - session-level:
    - `artifacts/pr/pr_all_classes.png`,
    - `artifacts/pr/per_class/pr_per_class.csv`,
    - `artifacts/pr/per_class/pr_class_*.png`.
- Run-level cache:
  - `.smartrain_cache/analyze/pr/aggregate/...`,
  - `.smartrain_cache/analyze/pr/per_class/...`.

### `smartrain analyze inference-benchmark`

- Назначение: замер скорости инференса.
- Артефакты:
  - `benchmark.csv`.
- Run-level cache:
  - `.smartrain_cache/analyze/inference/bench_<fingerprint>.csv`.

### `smartrain analyze inference-plot`

- Назначение: построение графика по CSV из `inference-benchmark`.
- Зависимость:
  - требует CSV (`--csv`), обычно созданный `inference-benchmark`.
- Артефакты:
  - `benchmark_bars.png` (или путь из `--out-png`).

### `smartrain analyze test-metrics-plot`

- Назначение: сравнение test-метрик по run.
- Артефакты:
  - `test_metrics_*_<metric>.png`,
  - `metric_sources.json` (если включён вывод),
  - `run/test_metrics_recomputed.csv` (при пересчёте отсутствующих метрик).
- Run-level cache:
  - `.smartrain_cache/analyze/metrics/recomputed_<fingerprint>.csv`,
  - `.smartrain_cache/analyze/metrics/recompute_status_<fingerprint>.json`.

### `smartrain analyze all`

- Назначение: полный сценарий анализа с отчётом.
- Session-level структура:
  - `analytics/analyze-reports/<session>/session.json`,
  - `ru/index.md`, `en/index.md`,
  - `report-ru.pdf|odt`, `report-en.pdf|odt` (если доступен движок),
  - `artifacts/compare/...`,
  - `artifacts/table/...`,
  - `artifacts/leaderboard/...`,
  - `artifacts/metrics/...`,
  - `artifacts/inference/...`,
  - `artifacts/pr/...`,
  - `artifacts/speed_quality/speed_quality.csv`,
  - `artifacts/speed_quality/speed_vs_map.png`.

## Зависимости подкоманд и flow

- `all` orchestrates:
  - `compare`
  - `export-table`
  - `leaderboard`
  - `test-metrics-plot` (quality/full)
  - `inference-benchmark` -> `inference-plot` (speed/full)
  - `pr-curves` (full)
  - затем: `session.json` + RU/EN отчёты

- `speed-vs-mAP` scatter строится из:
  - speed: output `inference-benchmark`,
  - quality: test metrics (`original/recomputed`).

## Single-run vs cross-run артефакты

### Single-run (кэшируемые в run)

- `test_metrics_recomputed.csv`,
- `test/pr.csv`,
- `.smartrain_cache/analyze/metrics/...`,
- `.smartrain_cache/analyze/pr/...`,
- `.smartrain_cache/analyze/inference/...`.

### Cross-run (сессионные)

- compare deltas/plots/insights,
- leaderboard,
- speed-vs-mAP,
- session manifest,
- итоговые RU/EN отчёты.

## Шпаргалка

### Быстрые сценарии

- Полный интерактивный анализ:
  - `smartrain analyze`

- Полный анализ с настройками:
  - `smartrain analyze all --report-languages ru,en --scatter-x avg_inference_ms_per_frame --scatter-y mAP50-95`

- Сводная таблица по всем run:
  - `smartrain analyze export-table -o runs_summary.csv`

- Сравнение двух/нескольких run:
  - `smartrain analyze compare --baseline /path/to/run_a --others /path/to/run_b /path/to/run_c`

- Лидерборд по quality/speed:
  - `smartrain analyze leaderboard --quality-metric mAP50-95 --speed-metric avg_inference_fps -o leaderboard.csv`

### PR и инференс

- PR curves (включая per-class):
  - `smartrain analyze pr-curves --runs-group-dir runs/<dataset> --data-yaml datasets/<dataset>/data.yaml --pr-per-class`

- Benchmark скорости:
  - `smartrain analyze inference-benchmark --runs-group-dir runs/<dataset> --data-yaml datasets/<dataset>/data.yaml --split test --frames 200`

- График скорости из benchmark CSV:
  - `smartrain analyze inference-plot --csv /path/to/benchmark.csv --out-png /path/to/benchmark_bars.png`

### Метрики теста и пересчёт

- Сравнение test-метрик:
  - `smartrain analyze test-metrics-plot --runs-group-dir runs/<dataset> --metrics mAP50-95 Box-F1`

- С пересчётом отсутствующих метрик:
  - `smartrain analyze test-metrics-plot --runs-group-dir runs/<dataset> --metrics mAP50-95 Box-F1 --recompute-missing-metrics`

### Память GPU в аналитике

- По умолчанию используется memory-safe ladder в `val()`:
  - очистка GPU между попытками,
  - снижение batch/imgsz при OOM,
  - FP16 (если включено),
  - политика GPU-only/CPU fallback управляется флагами.

- Полезные флаги:
  - `--val-batch`,
  - `--val-imgsz`,
  - `--val-half` / `--no-val-half`,
  - `--gpu-only-val` / `--allow-cpu-fallback`.
