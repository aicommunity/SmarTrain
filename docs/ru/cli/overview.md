> English version: [../../cli/overview.md](../../cli/overview.md)

# CLI: обзор

Точка входа: `smartrain` (Typer-роутер с единым поведением команд).

## Группы команд

- Датасеты: `scan`, `normalize-data-yaml`, `fusion`, `augment`, `balance`, `prune`, `orient`, `rotate`, `roi`, `inference`, `hash`, `stats`, `report dataset`
- Обучение: `train`, `clearml-upload`
- Провайдеры: `providers`
- Инфо: `info`
- Очередь: `queue`, `queue-run`
- Аналитика: `analyze`, `plot` (устаревшая обёртка)
- Реестр: `registry`
- Модели: `model convert`, `model release`, `model rename`
- Инструменты форматов: `cvat`, `sahi`, `heatmap`
- Миграция: `migrate-models`

## Справка

```bash
smartrain --help
smartrain <команда> --help
```

Для вложенных команд:

```bash
smartrain queue list --help
smartrain analyze inference-benchmark --help
smartrain model convert --help
```

Единый контракт интерактива:

- интерактив включается только при запуске команды без аргументов (TTY обязателен);
- выбор датасета(ов): сразу нумерованный список; ввод по имени или по номеру (несколько датасетов — через CSV номеров или имён);
- для `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `inference`, `orient`, `rotate`, `report dataset`, `model convert`, `model release`, `model rename` пустой вызов запускает интерактивный режим;
- если переданы любые аргументы, но их недостаточно, команда завершится понятной ошибкой о неполных аргументах (без prompt-режима).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.

Особенности `smartrain info`:

- Печатает секцию `Supported train models` с алиасами, которые можно копировать напрямую в `smartrain train --model ...`.
- Включает алиасы backend по умолчанию и провайдер-специфичные алиасы установленных внешних провайдеров.

Особенности `model convert`:

- `smartrain model convert` экспортирует `.pt` в `onnx`, `tensorrt-engine` и `tensorrt-trt`, а также поддерживает прямую конвертацию `.onnx -> tensorrt-trt`.
- По умолчанию: статический batch-режим, `--batch 1`, `--precision fp32`.
- ONNX-параметры настраиваются в `model convert` (`--opset`, `--simplify/--no-simplify`, `--half/--no-half`).
- В интерактивном режиме команда автоматически находит `.pt/.onnx` в `models/` и `runs/` workspace и даёт выбор источника по номеру или ручной ввод пути.
- Выходные модели выбираются отдельно (`onnx`, `engine`, `trt`) с мультивыбором (`1,2` или `onnx,trt`), недоступные варианты показываются с причиной.
- Для run-источников интерактивный выбор использует канонические артефакты (`<run_dir>/<run_dir_name>.<ext>`). Legacy-раскладка run автоматически канонизируется при первом обращении.

Особенности `model release`:

- `smartrain model release` публикует canonical run-модель `<run_dir_name>.pt` из выбранного run в `models/<dataset>/<task>_<model>_<train_datetime>.pt`.
- Рядом создаётся JSON с тем же basename (`.json`) c описанием источника, данных обучения, метрик, классов и `io_spec` модели.
- Повторный вызов для того же run и того же веса (совпадают источник и хеш) ничего не делает (`skip`).

Особенности `model rename`:

- `smartrain model rename` переименовывает release-модель в `models/<dataset>/`: меняется stem (`.pt`, sidecar `.json`, каталог артефактов release и конвертированные ONNX/engine/trt с тем же префиксом).
- Registry-бандлы (`model_manifest.json`) и модели в `runs/` не затрагиваются.
- В интерактивном режиме показывается список release-моделей, текущий stem подставляется в поле ввода для редактирования.

Особенности `train` (контроль модели):

- В интерактивном режиме итоговая модель для запуска печатается явно (`Final model for launch`).
- Перед `model.train()` выводятся `Requested model` и `Loaded model` для проверки фактически загруженных весов.
- Для YOLO-алиасов контролируется не только family, но и scale (`n/s/m/l/x`): тихая подмена вроде `yolo11x -> yolo11n` блокируется.
- В non-interactive режиме при таком расхождении запуск завершается ошибкой; в interactive режиме требуется явное подтверждение.
- В интерактивном режиме выбор модели выполняется из списка поддерживаемых алиасов с опцией `<manual>` для ручного ввода (например для форков/кастомных весов).

Дополнения для балансировки и статистики:

- `smartrain balance` поддерживает стратегии `weights`, `rfs`, `hybrid` и параметры их настройки.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default,hybrid-aug-tail-budget}` применяет готовые настройки под типовые сценарии.
- Для `--strategy hybrid-aug` по умолчанию включён режим контролируемого роста с приоритетом хвоста: `--aug-total-bbox-cap-mult 1.10`, `--aug-budget-tail-first`, `--aug-budget-tail-gamma 1.0`, `--train-head-bbox-undersample median-factor`, `--train-head-bbox-cap-mult 5.0`, а также консервативное прореживание head в eval-сплитах `--eval-head-bbox-undersample median-factor --eval-head-bbox-cap-mult 8.0 --eval-head-bbox-min-count 30 --eval-head-bbox-max-remove-frac 0.35` (явные CLI-флаги имеют приоритет).
- `smartrain balance --eval-coverage` (по умолчанию включено) подстраивает пул train после балансировки: по возможности не оставлять пустыми `val`/`test` и донаполнять в eval отсутствующие классы из train, при этом один и тот же source-кадр не распределяется между разными сплитами; если уникальных кадров не хватает, `val/test` могут быть заполнены не полностью; отключение — `--no-eval-coverage`. В интерактивном `balance` тот же выбор задаётся вопросом.
- `smartrain stats --balance-ready` выводит метрики дисбаланса и рекомендации для балансировщика.
- `smartrain prune empty` удаляет пустые пары image/label в новый датасет `<dataset>_pruned`.
- `smartrain prune dedup` удаляет дубли изображений по содержимому в `<dataset>_deduped` (глобальный приоритет split: train > val > test).
- `smartrain report dataset` формирует многоязычный отчёт с примерами по классам (Markdown + PNG; по умолчанию `analytics/datasets-reports/<dataset>_<timestamp>/`). В базовые зависимости входят pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2` и `odfpy` для PDF/ODT. Для WeasyPrint на некоторых ОС могут понадобиться системные библиотеки (Cairo, Pango), если нет подходящего wheel.
- `smartrain inference` запускает инференс по двум режимам источника данных: `folder` (произвольная папка с изображениями) и `dataset-split` (`train|val|test` подвыборка из датасета по `datasets_info` + `data.yaml`). Результат сохраняется в `inference/<model>/<timestamp>-<source>/inference_results.json`; в отчёте есть параметры запуска, источник (абсолютный/относительный путь), ROI (если включён pre-detect) и детекции с координатами в ROI/исходной системе координат, классом и confidence.
