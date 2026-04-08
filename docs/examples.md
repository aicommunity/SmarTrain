# Примеры вызовов CLI

Ниже — готовые команды для терминала. Полный список флагов у каждой подкоманды: **`smartrain <команда> --help`**. Форматы каталогов и полей JSON — в [data_formats.md](data_formats.md).

**Глобально для всех подкоманд Typer:** можно задать корень workspace один раз — **`smartrain --workspace /path/to/ws <команда> ...`** или переменная **`SMART_TRAIN_WORKSPACE`** (см. [workspace.md](workspace.md)).

Для любой структуры датасета (`flat`, `nested_split`, `darknet`, `cvat11`, …) типичная цепочка: **`scan`** → **`fusion`** → **`train`**; отличается только то, что попадёт в `datasets_info.json` после `scan`.

---

## Workspace: deploy, scan, fusion, train

```bash
export SMART_TRAIN_WORKSPACE=/path/to/workspace
# или: smartrain --workspace /path/to/workspace deploy

smartrain deploy
# развернуть структуру в указанном каталоге:
smartrain deploy /path/to/new_workspace

smartrain scan
smartrain fusion --output-name my_merge --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data my_merge --model yolov8n --epochs 50 --batch 16 -y
```

`--data my_merge` — имя каталога в `datasets/`, как в `datasets/datasets_info.json`.

---

## `smartrain scan`

Сканирование внутри workspace (синхронизирует источники из `raw_data/` в `datasets/`, индекс строит по `datasets/`):

```bash
smartrain scan
```

Очистить из `raw_data` источники, которые были обработаны и синхронизированы в `datasets/` в текущем запуске (с предупреждением и подтверждением, по умолчанию `Y`):

```bash
smartrain scan --purge-processed-raw
```

Явный источник (добавляется в `raw_data/datasets_list.txt`):

```bash
smartrain scan --dataset my_dataset
smartrain scan --dataset /abs/path/to/external_dataset
```

Указать корень с датасетами и куда положить `datasets_info.json` / `class_names.json`:

```bash
smartrain scan --datasets-path /data/datasets --output-path .
```

Список путей к датасетам в файле:

```bash
smartrain scan --datasets-list /path/to/datasets_list.txt --output-path .
```

Пересканировать **только** записи с полем `data_path` в уже существующем `datasets_info.json` (полезно после правок путей), в workspace:

```bash
smartrain scan --mode refresh
```

---

## `smartrain fusion` (workspace)

Имя выхода в `datasets/`:

```bash
smartrain fusion --output-name my_merge --dataset ds_a --dataset ds_b --classes "class_a,class_b"
```

Автоимя каталога (timestamp + `-merged`), если `--output-name` не задавать:

```bash
smartrain fusion --dataset ds_a --dataset ds_b --classes "class_a,class_b"
```

Если не передавать `--dataset/--datasets`, команда запускается в интерактивном режиме: показывает доступные датасеты и ждёт ввода списка через запятую с автодополнением.

**Без `--classes`:** в merge попадает **объединение всех классов** из всех записей `datasets_info.json` (кроме датасета с именем выходной папки), имена предварительно нормализуются через **`class_names.json`**. В логе будет строка вида «`--classes` не задан: используется объединение классов…». Маппинг в `class_names.json` при этом **всё равно** применяется; вы просто не сужаете список вручную.

```bash
smartrain fusion --output-name all_classes_auto --dataset ds_a --dataset ds_b
```

Слияние нескольких исходных имён в один класс из `--classes`:

```bash
smartrain fusion \
  --output-name unified \
  --classes "class_x" \
  --merge-classes "ClassX,class_x_alt,CLASS_X" class_x
```

Датасеты, где нет **всех** перечисленных в `--classes` классов, но есть часть:

```bash
smartrain fusion --output-name merged --dataset ds_a --dataset ds_b --classes "class_a,class_b,class_c" --include-partial-datasets
```

Не подмешивать исходные тестовые кадры:

```bash
smartrain fusion --output-name no_src_test --dataset ds_a --dataset ds_b --classes "class_a,class_b" --exclude-test
```

Доли **train / val / test** при случайном разбиении внутри каждой «корзины» кадров (сумма **1.0**; только **`fusion`**, не `train`):

```bash
smartrain fusion --output-name split_701020 --dataset ds_a --dataset ds_b --classes "class_a,class_b" --fusion-split 0.7,0.2,0.1
```

Только train + val на выходе:

```bash
smartrain fusion --output-name tv_only --dataset ds_a --dataset ds_b --classes "class_a,class_b" --fusion-split 0.9,0.1,0
```

После слияния убрать пары image+label с пустыми метками:

```bash
smartrain fusion --output-name cleaned --dataset ds_a --dataset ds_b --classes "class_a,class_b" --drop-empty-images
```

Сузить набор классов до **пересечения** по датасетам-кандидатам (`--common-classes-only`; часто вместе с осмысленным `--classes`):

```bash
smartrain fusion --output-name common_only --dataset ds_a --dataset ds_b --classes "class_a,class_b" --common-classes-only
```

Временные файлы (например для cvat11) в явный каталог:

```bash
smartrain fusion --output-name tmp_here --dataset ds_a --dataset ds_b --classes "class_a,class_b" --tmp-dir /path/to/tmp
```

Явный выход в файловую систему при заданном workspace (вместо `datasets/<name>`):

```bash
smartrain fusion --output-name my_merge --dataset ds_a --dataset ds_b --classes "class_a,class_b" --target-path /data/custom_out
```

---

## `smartrain fusion` (legacy, без workspace)

Нужны все три пути:

```bash
smartrain fusion \
  --source-path /data/datasets \
  --target-path /data/output/merged \
  --datasets-info-path . \
  --dataset ds_a \
  --dataset ds_b \
  --classes "class_a,class_b"
```

Относительные пути от текущей директории:

```bash
smartrain fusion \
  --source-path ./datasets \
  --target-path ./output/merged \
  --datasets-info-path . \
  --datasets ds_a,ds_b \
  --classes "class_a,class_b"
```

---

## `smartrain train`

Интерактивный режим (без аргументов): выбор датасета из `datasets_info.json`, затем опциональный выбор базового прогона из `runs/` (в начале списка идут прогоны выбранного датасета), после чего ввод остальных параметров с дефолтами по Enter.

```bash
smartrain train
```

Workspace и имя датасета из `datasets/`:

```bash
smartrain train --data my_merge --model yolov8n --epochs 50 --batch 16 -y
```

Без workspace: каталог с `data.yaml` и база для прогонов:

```bash
smartrain train \
  --data /abs/path/to/dataset \
  --target-path /abs/path/runs \
  --model yolov8s \
  --epochs 100 \
  -y
```

Базовый профиль smart-train (`--config`), часть полей можно не дублировать в CLI:

```bash
smartrain train --workspace . --data my_merge -c /path/to/train_profile.yaml --epochs 100
```

Внешний Ultralytics `args.yaml`:

```bash
smartrain train --workspace . --data my_merge --ultralytics_yaml /path/to/args.yaml --epochs 100
```

Приоритет параметров: `CLI > --ultralytics_yaml > --config > defaults`.  
`data` из `--ultralytics_yaml` игнорируется, используется выбранный `--data`.

Другая задача Ultralytics (`detect` по умолчанию):

```bash
smartrain train --data my_merge --task segment --model yolo11n-seg.pt -y
```

Размер входа и согласие без вопросов при существующей папке прогона:

```bash
smartrain train --data my_merge --img-size 1280 --epochs 100 -y
```

В workspace положить прогоны не в `runs/`, а в свой каталог:

```bash
smartrain train --data my_merge --target-path /data/my_runs --model yolov8n -y
```

Взвешенная выборка изображений (дисбаланс классов; патч ultralytics):

```bash
smartrain train --data my_merge --weighted-sampling -y
```

Экспорт ONNX после успешного обучения:

```bash
smartrain train --data my_merge --export-onnx -y
smartrain train --data my_merge --export-onnx --export-onnx-fp32 -y
```

Дополнительно см. справку: **`--config`**, **`--ultralytics_yaml`**, **`--val-imgsz`**, **`--val-batch`**, **`--val-conf`**, **`--val-iou`**.

Несколько разных моделей на одном датасете:

```bash
smartrain train --data my_merge --model yolov8n --epochs 30 --batch 32 -y
smartrain train --data my_merge --model yolov8s --epochs 50 --batch 16 -y
smartrain train --data my_merge --model yolov8l --epochs 100 --batch 8 -y
```

Только прогон валидации на test по уже обученному прогону:

```bash
smartrain train \
  --test-only \
  --model-dir /path/to/run_folder \
  --data /path/to/dataset_with_data_yaml \
  --val-imgsz 1280 --val-batch 1
```

С явными порогами для `val`:

```bash
smartrain train --data my_merge --val-imgsz 1280 --val-conf 0.35 --val-iou 0.6 -y
```

---

## ClearML (опционально)

Пакет: **`pip install 'smartrain[clearml]'`** (или `pip install clearml`). Учётка и сервер — как в документации ClearML (`clearml-init`, переменные окружения).

**Во время обучения** — гиперпараметры из словаря, который уходит в `YOLO.train`, попадают в задачу через `Task.connect`:

```bash
smartrain train --data my_merge --model yolov8n --epochs 50 -y --clearml
smartrain train --data my_merge --model yolov8n --epochs 50 -y --clearml --clearml-project MyProject
```

Имя проекта, если не задано флагом: переменная **`CLEARML_PROJECT`**, иначе **`smartrain`**. Имя задачи в ClearML — последний компонент пути прогона (каталог run).

В YAML-профиле обучения (`-c`): ключи **`clearml: true`** и опционально **`clearml_project: ...`** (они не передаются в Ultralytics).

**После прогона** — отдельная подкоманда: залить каталог run (`train/args.yaml`, `results.csv`, картинки, `best.pt`):

```bash
smartrain clearml-upload /path/to/run_folder
smartrain clearml-upload /path/to/run_folder --project MyProject --task-name custom_name
smartrain clearml-upload /path/to/run_folder --no-images
```

Справка: **`smartrain clearml-upload --help`**.

---

## `smartrain stats`

Статистика считается только по каталогу `datasets/` (не по `raw_data/`).

Быстрый режим (без подкоманды) показывает сводку по датасетам и классам:

```bash
smartrain stats
smartrain stats --dataset my_merge
```

Таблица по классам (train/val/test/total + итог по дисбалансу):

```bash
smartrain stats classes
smartrain stats classes --dataset my_merge
smartrain stats classes --dataset my_merge --classes "cat,dog" --sort total --desc
```

Сводка по датасетам в целом:

```bash
smartrain stats datasets
smartrain stats datasets --dataset my_merge --sort gini --desc
```

Проверки duplicate/leakage risk и экспорт списка проблем в `analytics/stats/...`:

```bash
smartrain stats datasets --check-duplicates --check-near-duplicates --export-issues
```

Сравнение двух датасетов:

```bash
smartrain stats compare --left 170325 --right 291124
smartrain stats compare --left 170325 --right 291124 --details all --top-n 30
smartrain stats compare --left 170325 --right 291124 --export-json --export-csv
```

Интерактивный compare (выбор left/right и опций):

```bash
smartrain stats compare
```

---

## `smartrain augment`

Создаёт новый датасет в `datasets/` (по умолчанию: `<dataset>_aug`, при конфликте — `_aug_2`, `_aug_3`, ...).

```bash
smartrain augment --dataset my_merge
smartrain augment --dataset my_merge --enable-flip --flip vertical --flip-prob 0.7 --enable-conveyor --splits train,val
smartrain augment --dataset my_merge --enable-center-rotate --center-rotate-deg 5 --rotate-copies 2
smartrain augment --dataset my_merge --enable-bbox-copy --bbox-copy-copies 2 --placement-mode detector --roi-model yolo11n.pt --copy-paste-count 2
smartrain augment --dataset my_merge --enable-bbox-copy --placement-mode bbox --copy-paste-count 2
smartrain augment --dataset my_merge --enable-bbox-copy --placement-mode none --copy-paste-count 2
smartrain augment --dataset my_merge --enable-bbox-copy --copy-paste-placement-style uniform-grid --copy-paste-min-center-dist 0.2
smartrain augment --dataset my_merge --imbalance-mode soft --imbalance-strength 1.0
smartrain augment --dataset my_merge --classes "cat,dog" --output-name my_merge_aug_custom
```

Имена аугментированных файлов имеют короткий тег: `<orig>__a-<mode><flip><v><idx>`, например `img_001__a-bhn1.jpg`.

Интерактивный режим:

```bash
smartrain augment
```

Для `bbox_copy` в интерактиве доступен выбор ROI-режима:
- `detector` (по умолчанию) — ROI через детектор;
- `bbox` — ROI по существующей разметке;
- `none` — без ROI-ограничений.

Для `bbox_copy` также доступен стиль размещения:
- `random` — полностью случайный выбор валидной позиции;
- `uniform-grid` — более равномерное покрытие ROI/кадра по сетке.

---

## `smartrain balance`

Создаёт новый датасет в `datasets/` (по умолчанию: `<dataset>_balanced`, при конфликте — `_balanced_2`, `_balanced_3`, ...).

```bash
smartrain balance --dataset my_merge --strategy oversample --target 1.5
smartrain balance --dataset my_merge --strategy undersample --target 0.8
smartrain balance --dataset my_merge --class cat
smartrain balance --dataset my_merge --classes "cat,dog" --emit-train-config
```

Интерактивный режим:

```bash
smartrain balance
```

---

## `smartrain hash`

Хеш датасета (как при именовании прогонов `train`). По пути к каталогу с `data.yaml`:

```bash
smartrain hash /path/to/dataset
```

В workspace по имени work-датасета:

```bash
smartrain hash --dataset my_merge
```

По имени датасета из каталога `datasets/`:

```bash
smartrain hash --raw-dataset dataset_key_from_json
```

Проверка, что хеш совпадает с ожидаемым:

```bash
smartrain hash /path/to/dataset --validate a1b2c3d4
```

Для архива `.zip` только метаданные архива (без распаковки):

```bash
smartrain hash --raw-dataset my_zip_dataset --hash-zip-metadata
```

---

## `smartrain roi`

Кроп датасета по ROI из модели YOLO (detect/segment). В workspace:

```bash
smartrain roi --dataset-name my_dataset
```

Явный выход и веса:

```bash
smartrain roi --dataset-name my_dataset --output-path /data/my_dataset_roi --weights /path/to/model.pt
```

Сегментационная модель и политика ROI:

```bash
smartrain roi --dataset-name my_dataset --mode yolo_segment --roi-policy union --conf 0.35
```

Legacy (родительский каталог датасетов):

```bash
smartrain roi --dataset-name my_dataset --source-path /data/parent --datasets-info-path .
```

---

## Очередь: `queue.txt` и `queue-run`

Содержимое `queue.txt` в корне workspace (по одной команде на строку, `#` — комментарий):

```text
# fusion затем три обучения
smartrain fusion --output-name merged --dataset ds_a --dataset ds_b --classes "class_a,class_b"
smartrain train --data merged --model yolov8n --epochs 50 -y
smartrain train --data merged --model yolov8s --epochs 50 -y
smartrain train --data merged --model yolov8m --epochs 50 -y
```

Запуск исполнителя:

```bash
smartrain queue-run
```

Добавить строку в очередь из терминала:

```bash
smartrain queue add -- smartrain train --data merged --model yolov8n -y --epochs 10
smartrain queue list
```

Удалить строку по номеру из `queue list` или по подстроке:

```bash
smartrain queue remove --index 3
smartrain queue remove --substring "yolov8s"
smartrain queue remove --substring "obsolete" --all
```

Очистить очередь:

```bash
smartrain queue clear
```

Запуск очереди **через подкоманду** `queue run` (как `queue-run`, опционально без GUI):

```bash
smartrain queue run --no-gui
smartrain queue run --cwd /path/to/workspace
```

Исполнитель **`queue-run`**: свой файл очереди/статуса и рабочий каталог:

```bash
smartrain queue-run --no-gui
smartrain queue-run --queue-file /path/to/queue.txt --status-file /path/to/status.txt
smartrain queue-run --cwd /path/to/workspace
```

---

## `smartrain analyze`

Корень поиска прогонов по умолчанию — `workspace/runs`, если задан workspace; иначе текущий каталог. Явно:

```bash
smartrain analyze scan
smartrain analyze scan --models-root /path/to/runs
```

Сводная таблица по всем прогонам с `training_metadata.json`:

```bash
smartrain analyze export-table -o runs_summary.csv
smartrain analyze export-table --models-root /path/to/runs -o out/summary.csv
```

Сравнение (как в примере ниже) + опционально сессия артефактов в `workspace/analytics/<name>/`:

```bash
smartrain analyze compare --baseline /path/to/run1 --others /path/to/run2 /path/to/run3 \
  --out-csv cmp.csv --out-png cmp.png --metric-column "metrics/mAP50-95(B)"
```

Интерактивный выбор номеров прогонов в терминале:

```bash
smartrain analyze interactive --output-dir ./analytics_out
```

---

## `smartrain plot`

Устаревшая обёртка над модулем анализа; предпочтительно **`smartrain analyze`**. Справка: **`smartrain plot --help`**.

---

## `smartrain registry`

Работает от workspace (`runs/`, `models/`). Список прогонов и детали:

```bash
smartrain registry runs-list
smartrain registry runs-info /path/to/specific/run
smartrain registry runs-info 3
smartrain registry runs-metrics 3
```

Промо модели в `models/<имя>/`:

```bash
smartrain registry models-add 3
smartrain registry models-list
smartrain registry models-info my_model_name
smartrain registry models-remove my_model_name
```

---

## `smartrain cvat`

Импорт архива CVAT 1.1 → YOLO-папка:

```bash
smartrain cvat import --cvat-zip /path/to/export.zip --output-dir /path/to/yolo_dataset --force
```

Экспорт YOLO → CVAT zip:

```bash
smartrain cvat export --dataset-dir /path/to/yolo_dataset --zip-path /path/to/out.cvat11.zip
smartrain cvat export --dataset-dir /path/to/yolo_dataset --names "a,b,c"
```

---

## `smartrain sahi` и `smartrain heatmap`

Нужны extras: **`pip install 'smartrain[sahi]'`**; heatmap использует только `ultralytics`.

Тайловый инференс (изображение или каталог):

```bash
smartrain sahi --model /path/to/best.pt --source /path/to/image_or_dir --output sahi_out \
  --slice-h 640 --slice-w 640 --overlap-h 0.2 --overlap-w 0.2 --conf 0.25 --device cuda
```

Heatmap по одному изображению (без `--output` откроется окно просмотра, если доступно):

```bash
smartrain heatmap --model /path/to/best.pt --source /path/to/image.jpg --output heat.png
```

---

## Типичные ошибки (как выглядит вывод)

Датасет для `scan` не найден:

```bash
$ smartrain scan --datasets-path /wrong/path
# [ERROR] Папка '/wrong/path' не найдена.
```

Классы не подходят под строгий режим `fusion`:

```bash
$ smartrain fusion --output-name x --dataset ds_a --dataset ds_b --classes "missing_class"
# [ERROR] Ни один датасет не содержит все выбранные классы.
```

Попробовать с частичным пересечением:

```bash
smartrain fusion --output-name x --dataset ds_a --dataset ds_b --classes "class_a,class_b" --include-partial-datasets
```

Нет `data.yaml` у `train`:

```bash
$ smartrain train --data /path/to/bad
# [ERROR] Не найден yaml файл: .../data.yaml
```

---

## Скрипт bash: scan → fusion → train

```bash
#!/bin/bash
set -e
export SMART_TRAIN_WORKSPACE="/path/to/ws"
smartrain scan
smartrain fusion --output-name nightly --dataset ds_a --dataset ds_b --classes "class_a,class_b" --fusion-split 0.8,0.1,0.1
smartrain train --data nightly --model yolov8n --epochs 50 -y
```

```bash
chmod +x run_nightly.sh
./run_nightly.sh
```

---

## Заметка про `nested_split` и `fusion`

Если после `scan` у датасета несколько пар каталогов `images`/`labels` (например исходные `train`, `val`, `test`), **`fusion` обрабатывает каждую пару отдельно**: кадры перемешиваются и снова делятся пропорциями **`--fusion-split`** на выходные `train`, `valid`, `test`. Это не копирование исходного разбиения один в один.
