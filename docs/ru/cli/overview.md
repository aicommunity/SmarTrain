> English version: [../../cli/overview.md](../../cli/overview.md)

# CLI: обзор

Точка входа: `smartrain` (Typer-роутер с единым поведением команд).

## Группы команд

- Датасеты: `scan`, `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`, `inference`, `hash`, `stats`, `report dataset`
- Обучение: `train`, `clearml-upload`
- Очередь: `queue`, `queue-run`
- Аналитика: `analyze`, `plot` (устаревшая обёртка)
- Реестр: `registry`
- Модели: `model convert`, `model release`
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
- для `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `inference`, `orient`, `report dataset`, `model convert`, `model release` пустой вызов запускает интерактивный режим;
- если переданы любые аргументы, но их недостаточно, команда завершится понятной ошибкой о неполных аргументах (без prompt-режима).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.

Особенности `model convert`:

- `smartrain model convert` экспортирует `.pt` в `onnx`, `tensorrt` или `both`, а также поддерживает прямую конвертацию `.onnx -> tensorrt`.
- По умолчанию: статический batch-режим, `--batch 1`, `--precision fp32`.
- В интерактивном режиме команда автоматически находит `.pt/.onnx` в `models/` и `runs/` workspace и даёт выбор по номеру или ручной ввод пути.

Особенности `model release`:

- `smartrain model release` публикует только `train/weights/best.pt` из выбранного run в `models/<dataset>/<task>_<model>_<train_datetime>.pt`.
- Рядом создаётся JSON с тем же basename (`.json`) c описанием источника, данных обучения, метрик, классов и `io_spec` модели.
- Повторный вызов для того же run и того же веса (совпадают источник и хеш) ничего не делает (`skip`).

Особенности `train` (контроль модели):

- В интерактивном режиме итоговая модель для запуска печатается явно (`Final model for launch`).
- Перед `model.train()` выводятся `Requested model` и `Loaded model` для проверки фактически загруженных весов.
- Для YOLO-алиасов контролируется не только family, но и scale (`n/s/m/l/x`): тихая подмена вроде `yolo11x -> yolo11n` блокируется.
- В non-interactive режиме при таком расхождении запуск завершается ошибкой; в interactive режиме требуется явное подтверждение.

Дополнения для балансировки и статистики:

- `smartrain balance` поддерживает стратегии `weights`, `rfs`, `hybrid` и параметры их настройки.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` применяет готовые настройки под типовые сценарии.
- `smartrain balance --eval-coverage` (по умолчанию включено) подстраивает пул train после балансировки: по возможности не оставлять пустыми `val`/`test` и донаполнять в eval отсутствующие классы из train, при этом один и тот же source-кадр не распределяется между разными сплитами; если уникальных кадров не хватает, `val/test` могут быть заполнены не полностью; отключение — `--no-eval-coverage`. В интерактивном `balance` тот же выбор задаётся вопросом.
- `smartrain stats --balance-ready` выводит метрики дисбаланса и рекомендации для балансировщика.
- `smartrain prune empty` удаляет пустые пары image/label в новый датасет `<dataset>_pruned`.
- `smartrain prune dedup` удаляет дубли изображений по содержимому в `<dataset>_deduped` (глобальный приоритет split: train > val > test).
- `smartrain report dataset` формирует многоязычный отчёт с примерами по классам (Markdown + PNG; по умолчанию `analytics/datasets-reports/<dataset>_<timestamp>/`). В базовые зависимости входят pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2` и `odfpy` для PDF/ODT. Для WeasyPrint на некоторых ОС могут понадобиться системные библиотеки (Cairo, Pango), если нет подходящего wheel.
- `smartrain inference` запускает инференс по двум режимам источника данных: `folder` (произвольная папка с изображениями) и `dataset-split` (`train|val|test` подвыборка из датасета по `datasets_info` + `data.yaml`). Результат сохраняется в `inference/<model>/<timestamp>-<source>/inference_results.json`; в отчёте есть параметры запуска, источник (абсолютный/относительный путь), ROI (если включён pre-detect) и детекции с координатами в ROI/исходной системе координат, классом и confidence.
