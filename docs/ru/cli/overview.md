> English version: [../../cli/overview.md](../../cli/overview.md)

# CLI: обзор

Точка входа: `smartrain` (Typer-роутер с единым поведением команд).

## Группы команд

- Датасеты: `scan`, `normalize-data-yaml`, `fusion`, `augment`, `balance`, `prune`, `filter`, `orient`, `rotate`, `roi`, `inference`, `hash`, `stats`, `dataset report`, `dataset rename`
- Обучение: `train`, `clearml-upload`
- Провайдеры: `providers`
- Workspace: `deploy`, `quickstart`, `info`, `sync`
- Очередь: `queue`, `queue-run`
- Аналитика: `analyze`, `plot` (устаревшая обёртка)
- Реестр: `registry`
- Модели: `model convert`, `model release`, `model comment`, `model rename`
- Каталог датасетов: `dataset report`, `dataset rename`
- Инструменты форматов: `dataset convert`, `sahi`, `heatmap`, `vis`
- Миграция: `migrate`, `migrate-models`
- Обслуживание: `deps sync-torch`

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
- для `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `inference`, `orient`, `rotate`, `dataset report`, `dataset rename`, `model convert`, `model release`, `model comment`, `model rename` пустой вызов запускает интерактивный режим;
- если переданы любые аргументы, но их недостаточно, команда завершится понятной ошибкой о неполных аргументах (без prompt-режима).
Для ключевых команд и групп в help также добавлены блоки `Examples` / `Quick examples`.

Автодополнение:

- авто: best-effort настройка выполняется при первом запуске `smartrain`;
- ручной fallback:
  - `smartrain --install-completion`
  - `smartrain --show-completion`

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
- Если экспорт ONNX пропущен (уже есть публичный `.onnx`), интерактивный ONNX+TRT и пропуск в `--format onnx` всё равно создают dedicated `*_trtprep.onnx` для `trtexec` при совпадении подписи ONNX.

Особенности `model release`:

- `smartrain model release` публикует canonical run-модель `<run_dir_name>.pt` из выбранного run в самодостаточную папку `models/<dataset>/<task>_<model>_<train_datetime>/` (веса, sidecar JSON и копия артефактов train).
- Общий каталог `models/releases_manifest.json` хранит однострочные комментарии ко всем release-моделям; тот же комментарий дублируется в sidecar JSON модели.
- В интерактивном режиме запрашивается необязательный однострочный комментарий (на любом языке); в non-interactive — флаг `--comment`.
- Повторный вызов для того же run и того же веса (совпадают источник и хеш) ничего не делает (`skip`).

Особенности `model comment`:

- `smartrain model comment` задаёт или обновляет однострочный комментарий release-модели в `releases_manifest.json` и sidecar JSON.
- В интерактивном режиме показывается список release-моделей (с текущими комментариями), поле ввода предзаполнено текущим комментарием.

Особенности `model rename`:

- `smartrain model rename` переименовывает release-модель в `models/<dataset>/`: меняется stem (`.pt`, sidecar `.json`, каталог release и конвертированные ONNX/engine/trt с тем же префиксом).
- Registry-бандлы (`model_manifest.json`) и модели в `runs/` не затрагиваются.
- В интерактивном режиме показывается список release-моделей, текущий stem подставляется в поле ввода для редактирования.

Особенности `analyze`:

- `smartrain analyze` без подкоманды (TTY) запускает интерактивный `analyze all` (compare, метрики, опционально speed/PR, отчёт).
- Блок quality использует метрики обучения; отдельный `smartrain test` для compare и графиков test-metrics не обязателен.
- При `profile=full` speed (inference benchmark) берёт кадры из split `test`, затем `val`, затем `train`; если изображений нет — speed/PR деградируют с предупреждениями, отчёт всё равно строится.
- Пути в runtime `_runtime_data_*.yaml` разрешаются через поле `path:` в data.yaml (а не относительно файла в `run/tmp/`).
- `--strict-diagnostics` включайте только если отсутствие PR/metric_sources должно прерывать сессию.

Особенности `dataset report`:

- `smartrain dataset report` формирует многоязычный отчёт с примерами по классам (Markdown + PNG; по умолчанию `analytics/datasets-reports/<dataset>_<timestamp>/`). Для PDF/ODT зависимости `pypandoc-binary` и `weasyprint` ставятся через optional extra: `pip install -e ".[export]"`; `fpdf2` и `odfpy` остаются базовыми зависимостями. Для WeasyPrint на некоторых ОС могут понадобиться системные библиотеки (Cairo, Pango), если нет подходящего wheel.

Особенности `dataset rename`:

- `smartrain dataset rename` переименовывает ключ датасета и каталог `datasets/<name>/`, а также связанные `runs/<name>/` и `models/<name>/`, если они есть.
- Обновляет `datasets_info.json`, ссылки в `dataset_passport.json`, метаданные run, `queue.txt` и артефакты в `analytics/`.
- `--dry-run` показывает план без изменений; `--move-data-path` нужен, если `data_path` указывает вне стандартного `datasets/<name>/`.

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
- `smartrain stats --after-augment` сравнивает per-class bbox после balance и после augment (читает `balance_manifest.json` у hybrid-aug).
- Отдельный `smartrain augment --preset augment-tail-safe` включает class-aware geo, cap 1.10× и tail-first budget (те же настройки augment, что у `balance --preset hybrid-aug-tail-budget`, без запуска balance).
- `smartrain prune empty` удаляет пустые пары image/label в новый датасет `<dataset>_pruned`.
- `smartrain prune dedup` удаляет дубли изображений по содержимому в `<dataset>_deduped` (глобальный приоритет split: train > val > test).
- `smartrain prune classes` удаляет неиспользуемые классы из метаданных в `<dataset>_classes_pruned` (файлы не удаляются, `class_id` перенумеровывается).
- `smartrain filter` удаляет edge-truncated bbox в `<dataset>_fltd` (baseline inset + пороги; аудит в `_filter_audit/`; `--stats-only`, `--drop-images`, интерактивный preview).
- `smartrain scan --strip-unused-classes` очищает неиспользуемые классы у **новых** датасетов при scan (по умолчанию **вкл.**; `--no-strip-unused-classes` для отключения).
- `smartrain inference` запускает инференс по двум режимам источника данных: `folder` (произвольная папка с изображениями) и `dataset-split` (`train|val|test` подвыборка из датасета по `datasets_info` + `data.yaml`). Результат сохраняется в `inference/<model>/<timestamp>-<source>/inference_results.json`; по умолчанию дополнительно создаётся YOLO-датасет `<basename>_autolabeled/` в виде независимых поддатасетов `part_XXX/` с `autolabel_manifest.json` и опционально `pred_overlays/`. Пустой экспорт (нет меток после фильтра confidence) не создаёт эти каталоги. Команда `vis` вызывает inference без экспорта датасета.
