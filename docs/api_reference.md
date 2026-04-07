# Справочник API

## cli.py (команда `smartrain`)

Единая точка входа: **Typer**-приложение `app`, callback синхронизирует `SMART_TRAIN_WORKSPACE` с текущим каталогом, если переменная пуста.

Подкоманды с префиксом `context_settings` (`allow_extra_args`, `ignore_unknown_options`) и **`add_help_option=False`**: при наличии `--help` / `-h` в хвосте аргументов вызывается **`_dispatch_argparse_help`** с фабрикой `build_*_arg_parser` целевого модуля; иначе **`_call(module, "main", ctx)`** → `main(list(ctx.args))`.

Соответствие имён подкоманд и модулей см. исходный файл [`smartrain/cli.py`](../smartrain/cli.py).

Добавлены подкоманды:
- `smartrain augment` (модуль `smartrain.dataset_augment`)
- `smartrain balance` (модуль `smartrain.dataset_balance`)
- `smartrain stats` (модуль `smartrain.dataset_stats`)

Для `stats` режимы:
- `smartrain stats classes`
- `smartrain stats datasets`

## cvat_cli.py (команда `smartrain cvat`)

CLI-обёртка для конвертации **CVAT 1.1 (Images + bbox)**:

- `smartrain cvat import --cvat-zip <file.zip> --output-dir <dir> [--task-name <name>] [--force]`
  - распаковывает zip, читает `annotations.xml`, копирует `images/*`, пишет YOLO `labels/*.txt` и `data.yaml`.
- `smartrain cvat export --dataset-dir <dir> [--zip-path <out.zip>] [--task-name <name>] [--names a,b,c] [--force]`
  - экспортирует плоский YOLO-датасет (`images/` + `labels/`) обратно в CVAT 1.1 zip.

Примечание: `smartrain fusion` умеет работать с `structure="cvat11"` **нативно** (временные `.txt` метки генерируются на лету из `annotations.xml`), поэтому отдельный `smartrain cvat import` не обязателен для merge.

## cli_argparse.py

**`CliArgumentParser`** — подкласс `argparse.ArgumentParser` с `formatter_class=ArgumentDefaultsHelpFormatter`, чтобы в справке отображались значения по умолчанию опций.

---

## workspace_paths.py

Модуль единого корня workspace.

### `WORKSPACE_ENV_VAR`
Строка `"SMART_TRAIN_WORKSPACE"`.

### `resolve_workspace_root(cli_workspace: str | None) -> str`
Корень: непустой `--workspace` перекрывает переменную окружения; иначе `ValueError`.

### `class WorkspaceLayout`
В `__init__(root)` задаются поля: `root`, `raw_data`, `datasets`, `runs`, `analytics`, `models`, а также alias-поля совместимости `source_datasets` и `work_datasets`.

### `resolve_path_under_workspace(workspace_root, relative_or_absolute) -> str`
### `resolve_dataset_root(workspace_root, entry_key, entry_dict, catalog_dir) -> str`
Если в `entry_dict` есть `data_path` — резолв от workspace или абсолют; иначе `os.path.join(catalog_dir, entry_key)`.

---

## registry_cli.py

Через **`smartrain registry`**: `--workspace` (или env), подкоманды `runs-list`, `runs-info`, `runs-metrics`, `models-add`, `models-list`, `models-info`, `models-remove`. Парсер: **`build_registry_arg_parser()`**. Веса копируются в `models/<friendly_name>/<friendly_name>.pt` с `model_manifest.json`.

---

## datasets_json_former.py

**CLI**: `smartrain scan`.

### Функции

#### `find_yaml_file(folder_path: str) -> str | None`
Ищет файл `data.yaml` или `data.yml` в директории и поддиректориях.

**Параметры**:
- `folder_path` - путь к директории для поиска

**Возвращает**: Путь к найденному YAML файлу или `None`

---

#### `find_obj_names_file(folder_path: str) -> str | None`
Ищет файл `obj.names` в директории и поддиректориях (формат Darknet).

**Параметры**:
- `folder_path` - путь к директории для поиска

**Возвращает**: Путь к найденному файлу или `None`

---

#### `find_obj_data_file(folder_path: str) -> str | None`
Ищет файл `obj.data` в директории и поддиректориях (формат Darknet).

**Параметры**:
- `folder_path` - путь к директории для поиска

**Возвращает**: Путь к найденному файлу или `None`

---

#### `load_obj_names(file_path: str) -> list[str] | None`
Загружает список классов из файла `obj.names` (по одному классу на строку).

**Параметры**:
- `file_path` - путь к файлу `obj.names`

**Возвращает**: Список имен классов или `None` при ошибке

---

#### `load_obj_data(file_path: str) -> int | None`
Парсит файл `obj.data` и извлекает количество классов.

**Параметры**:
- `file_path` - путь к файлу `obj.data`

**Возвращает**: Количество классов или `None` при ошибке

---

#### `detect_structure(folder_path: str) -> str`
Определяет структуру организации датасета.

**Параметры**:
- `folder_path` - путь к директории датасета

**Возвращает**: Один из типов структуры:
- `"split"` - разделение на train/val/test
- `"flat"` - плоская структура images/labels (файлы в корне этих папок)
- `"subset_flat"` - плоская структура с подпапками одинакового имени в `images/` и `labels/` (экспорт CVAT Ultralytics YOLO Detection 1.0)
- `"nested_split"` - вложенное разделение
- `"darknet"` - формат Darknet YOLO
- `"unknown"` - неизвестная структура

---

#### `load_yaml(file_path: str) -> dict | None`
Загружает и парсит YAML файл.

**Параметры**:
- `file_path` - путь к YAML файлу

**Возвращает**: Словарь с данными или `None` при ошибке

---

#### `count_elements(folder_path: str, structure: str) -> int | tuple | None`
Подсчитывает количество элементов (изображений/аннотаций) в датасете.

**Параметры**:
- `folder_path` - путь к директории датасета
- `structure` - тип структуры датасета

**Возвращает**:
- `int` - количество элементов (если изображения и аннотации совпадают)
- `tuple` - кортеж (количество изображений, количество аннотаций) при несовпадении
- `None` - при ошибке или неизвестной структуре

---

#### `process_dataset(folder_path: str, folder_name: str) -> dict | None`
Обрабатывает один датасет и извлекает информацию о нем.

**Параметры**:
- `folder_path` - путь к директории датасета
- `folder_name` - имя датасета

**Возвращает**: Словарь с информацией:
```python
{
    "classes": {class_name: index},
    "structure": "split|flat|subset_flat|nested_split|darknet|cvat11",
    "elements_count": int_or_list
}
```
или `None` при ошибке

---

#### Сохранение `datasets_info.json` и поля `roi_auto` / `tags` / `data_path`

Если выходной `datasets_info.json` уже существует, перед записью он читается; для каждого имени датасета, снова присутствующего в новом скане, в запись переносятся из старого файла необязательные ключи **`roi_auto`**, **`tags`** и **`data_path`** (остальное берётся из свежего `process_dataset`).

`scan` использует `raw_data` как источник и индексирует текущее состояние `datasets`.

Служебные поля scan в записи датасета:
- `dataset_hash`
- `source_hash`
- `source_ref`
- `source_signature`
- `modified`

**CLI**: `--workspace` (результат в `datasets/`) или пара `--datasets-path` + опционально `--output-path`; `--mode scan|refresh` (refresh только с workspace); `--dataset` (повторяемый).

---

## dataset_roi_yolo.py

CLI: кроп датасета по ROI модели Ultralytics (detect/segment), пересчёт нормализованных меток. Подробности и формат `roi_auto` — в [data_formats.md](data_formats.md#опциональные-поля-не-перезаписываются-сканером).

**Основные аргументы (workspace, предпочтительно)**: `--dataset-name` (ключ в `datasets/datasets_info.json`), `--workspace` / `SMART_TRAIN_WORKSPACE`, опционально `--output-path` (по умолчанию `datasets/<dataset-name>_roi`), `--tmp-dir`, `--datasets-info-path` (файл или каталог с JSON), переопределения `--weights`, `--conf`, `--pad-px`, `--roi-policy`, `--mode`, `--on-empty`, `--require-roi-auto`.

**Legacy**: `--source-path`, обязательный `--output-path`, прежняя проверка `{source-path}/{dataset-name}/`.

---

## dataset_former.py

**CLI**: `smartrain fusion`.

### Параметры CLI (дополнительно)

- **`--dataset`** — входной датасет для объединения (повторяемый флаг).
- **`--datasets`** — CSV-список входных датасетов.
- Если `--dataset/--datasets` не переданы, `fusion` запускает интерактивный выбор (prompt_toolkit, ввод списка через запятую с автодополнением).
- **`--output-name`** — подкаталог в `datasets/`; если не указан, используется **`YYYY-MM-DD_HH-MM-SS-merged`** (локальное время на момент запуска).
- **`--classes`** — если не задан, список классов строится как **объединение** всех классов из всех записей `datasets_info.json` (кроме датасета с именем выходной папки), с нормализацией имён через `class_names.json`; порядок в итоговом списке — по возрастанию нормализованного имени.
- **`--include-partial-datasets`** — по умолчанию в merge попадают только датасеты, в которых объявлены **все** выбранные классы; с этим флагом берутся датасеты с **любым непустым пересечением** с выбранным набором (удобно при авто-объединении классов из разных источников).
- **`--drop-empty-images`** — после записи выходного каталога удалить пары `images/*` + `labels/*.txt`, где в метке нет ни одной валидной строки YOLO (пустые или битые файлы).
- **`--common-classes-only`** — среди датасетов, у которых есть пересечение с запрошенным набором классов (`--classes` или авто-объединение) и источниками `--merge-classes`, оставить в итоге только те классы, которые есть **в каждом** таком датасете; остальные отбрасываются с предупреждением в лог.
- **`--fusion-split`** — три доли `train,val,test` через запятую (сумма 1.0): как делить кадры **внутри каждого bucket** при случайном переразбиении на выходе `fusion`. По умолчанию `0.8,0.1,0.1`. На `scan`, `train`, `roi` не влияет.

### Функции

#### `safe_mkdir(path: str) -> None`
Создает директорию, если она не существует.

**Параметры**:
- `path` - путь к директории

---

#### `find_dataset_paths` / `iter_image_label_buckets` — модуль `dataset_access.py`

`find_dataset_paths(dataset_path, structure, arg=False)` — пары `(images, labels)` для YOLO-раскладок (без `cvat11`).

`iter_image_label_buckets(..., dataset_name, temp_root, exclude_test)` — то же для всех `structure`, включая `cvat11` (временные `.txt` в `temp_root`).

Реэкспорт `find_dataset_paths` из `dataset_former` сохранён для обратной совместимости.

---

#### `filter_label_file(src_label_path: str, dst_label_path: str, class_map: dict, class_names_map: dict, selected_classes: list[str]) -> bool`
Фильтрует файл аннотаций, оставляя только выбранные классы и переиндексируя их.

**Параметры**:
- `src_label_path` - путь к исходному файлу аннотаций
- `dst_label_path` - путь к выходному файлу аннотаций
- `class_map` - словарь соответствия имен классов их индексам в исходном датасете
- `class_names_map` - словарь нормализации имен классов
- `selected_classes` - список выбранных классов

**Возвращает**: `True` если файл содержит выбранные классы, иначе `False`

---

## model_training_module.py

### Функции

#### `train_yolo(..., workspace_root=None) -> tuple`
Обучает модель YOLO. Возвращает `(model_dir, training_start_time, training_end_time, dataset_hash, workspace_root)`.

**Параметры**: `dataset_path`, `model_version`, `epochs`, `batch`, `img_size`, `target_dir`, `non_interactive`, опционально `workspace_root` для метаданных.

**CLI**: `--workspace` и `--data` (каталог с `data.yaml` или имя из `datasets/datasets_info.json`), либо без workspace — обязательны `--data` и `--target-path`. Прогоны по умолчанию в `workspace/runs`.

YAML-источники:
- `--config` — базовый профиль smart-train;
- `--ultralytics_yaml` — внешний Ultralytics `args.yaml`;
- приоритеты: `CLI > --ultralytics_yaml > --config > defaults`;
- `data` из `--ultralytics_yaml` игнорируется (используется выбранный `--data`).

**Исключения**:
- `FileNotFoundError` - если датасет или `data.yaml` не найдены

---

#### `test_yolo(model_dir: str, dataset_path: str) -> None`
Тестирует обученную модель на тестовом наборе данных.

**Параметры**:
- `model_dir` - путь к директории с обученной моделью
- `dataset_path` - путь к датасету (должен содержать `data.yaml`)

**Исключения**:
- Может выбросить исключение при ошибке тестирования

---

#### `save_metrics_csv(test_result, model_dir: str) -> str`
Сохраняет метрики тестирования в CSV файл.

**Параметры**:
- `test_result` - результат тестирования от Ultralytics
- `model_dir` - директория для сохранения CSV файла

**Возвращает**: Путь к созданному CSV файлу

---

## training_queue.py

Исполнитель очереди: **`smartrain queue-run`**. Резолв путей: **`resolve_queue_status_paths()`** — при успешном `resolve_workspace_root` очередь = `queue.txt` в корне workspace, статусы = `workspace/tmp/status.txt`; иначе fallback на `training_queue.txt` и `tmp/status.txt` рядом с пакетом (см. константы в коде).

### Функции

#### `main_window() -> None`
Открывает окно терминала с автоматическим обновлением статуса задач.

---

#### `update_status(index: int, status: str) -> None`
Обновляет статус задачи по индексу в файле статуса.

**Параметры**:
- `index` - индекс строки в файле статуса
- `status` - новый статус задачи

---

#### `start_new_process(cmd: str) -> int`
Запускает процесс выполнения команды.

**Параметры**:
- `cmd` - команда для выполнения

**Возвращает**: Код возврата процесса (0 - успех, иначе ошибка)

---

#### `read_txt(txt_file: str) -> list[str]`
Читает текстовый файл построчно.

**Параметры**:
- `txt_file` - путь к текстовому файлу

**Возвращает**: Список строк файла

---

#### `process_line(line: str) -> str | None`
Обрабатывает строку команды из очереди.

**Параметры**:
- `line` - строка команды

**Возвращает**: Обработанную команду или `None` если строка пустая/комментарий

**Обработка** (актуальная логика):
- Строки, начинающиеся с `smartrain` или с пути, оканчивающегося на `/smartrain`, возвращаются без изменений
- Строки с префиксом `python3` / `python` — без изменений
- Иначе: вставка `python3` в начало и дополнение `.py` ко второму токену при необходимости (legacy)
- Комментарии `#` и пустые строки → `None`

---

#### `load_statuses() -> dict[str, str]`
Загружает статусы задач из файла.

**Возвращает**: Словарь `{задача: статус}`

---

#### `save_statuses(tasks: list[str], statuses: dict[str, str], status_file: str | None) -> None`
Сохраняет статусы в порядке строк очереди.

**Параметры**:
- `tasks` - список строк задач (без `\\n`)
- `statuses` - словарь `{задача: статус}`
- `status_file` - путь к `status.txt` (по умолчанию `tmp/status.txt`)

---

#### `get_queue_tasks(queue_path: str | None) -> list[str]`
Возвращает непустые строки очереди без комментариев.

---

#### `run_queue(no_terminal: bool, cwd: str | None, queue_path: str | None, status_file: str | None) -> None`
Цикл исполнения очереди. При `no_terminal=True` не вызывается `gnome-terminal`.

---

## dataset_hash.py

### CLI
- Ровно один из: позиционный путь к каталогу датасета, `--dataset`, `--raw-dataset`
- `--workspace` / `SMART_TRAIN_WORKSPACE` для `--dataset` и `--raw-dataset`
- `--hash-zip-metadata` — для `.zip` в `data_path`: хеш по пути/размеру/mtime без распаковки
- `--validate <hash>` — код выхода `0` при совпадении, `1` при расхождении, `2` при ошибке

### `calculate_dataset_hash(dataset_path: str) -> str`
Первые 8 символов MD5 по структуре, именам и размерам файлов.

### `calculate_zip_metadata_hash(zip_path: str) -> str`
Хеш метаданных архива (без обхода содержимого).

---

## dataset_stats.py

### CLI
- Команда: `smartrain stats`
- Подкоманды:
  - `classes` — статистика по классам в `datasets/` (train/val/test/total)
  - `datasets` — сводка по датасетам в `datasets/` (объем, quality flags, imbalance)
- Источник данных: только файловая система `datasets/` (без `raw_data/`)
- Интерактивный режим: для `classes`/`datasets` без фильтров в TTY

### Основные метрики
- `classes`: per-class counts по split и total, `images_with_class`, `avg_instances_per_image`, итог по дисбалансу (`ratio`, `cv`, `gini`, `mean/median`)
- `datasets`: `num_classes`, `images_total`, `labeled_images`, `empty_images`, `instances_total`, `orphan_images/labels`, `broken_label_lines`, `unknown_class_ids`
- Опционально: `--check-duplicates`, `--check-near-duplicates`, `--export-issues`

---

## dataset_augment.py

### CLI
- Команда: `smartrain augment`
- Источник: датасет из `datasets/datasets_info.json`
- Результат: новый датасет в `datasets/<name>` (по умолчанию `<dataset>_aug`, авто-нумерация при конфликте)
- Интерактивный режим: запуск без аргументов в TTY
- Паспорт изменений: `dataset_passport.json` в корне выходного датасета

---

## dataset_balance.py

### CLI
- Команда: `smartrain balance`
- Стратегии: `oversample`, `undersample`, `class-aware`, `weights`
- Фильтр классов: `--class` или `--classes`
- Результат: новый датасет в `datasets/<name>` (по умолчанию `<dataset>_balanced`, авто-нумерация при конфликте)
- Интерактивный режим: запуск без аргументов в TTY
- Паспорт изменений: `dataset_passport.json` в корне выходного датасета

---

## dataset_passport.py

Утилиты воспроизводимости для команд, создающих новые датасеты:
- `next_dataset_name(...)` — авто-нумерация имени при конфликте
- `write_dataset_passport(...)` — запись `dataset_passport.json` с параметрами, источником, хешами и метриками

---

## training_queue_cli.py

Команда **`smartrain queue`**. Парсер: **`build_queue_cli_arg_parser()`**. Подкоманды: `list`, `add`, `remove`, `clear`, `run`. Общие опции: `--workspace`, `--queue-file`, `--status-file`.

---

## results_analyzer.py

Общие флаги: `--workspace` (или `SMART_TRAIN_WORKSPACE`), `--models-root` (явный корень поиска прогонов), `--analytics-session` (только `export-table`: `workspace/analytics/<имя>/` + `session.json`).

Подкоманды:
- `scan` — список прогонов
- `export-table -o` — сводный CSV
- `compare --baseline --others … -o --out-png` — дельты и графики
- `interactive` — выбор прогонов в терминале

---

## Константы

### datasets_json_former.py
- `OUTPUT_FILE` / `OUTPUT_CLASS_NAMES_FILE` — алиасы имён из `workspace_paths` (`datasets_info.json`, `class_names.json`)

### dataset_former.py
- `FUSION_DEFAULT_DIR_SUFFIX` — суффикс имени каталога по умолчанию (`"merged"`); полное имя без `--output-name`: `YYYY-MM-DD_HH-MM-SS-merged`
- `DATASETS_INFO_FILE`, `CLASS_NAMES_FILE` — из `workspace_paths`
- `TRAIN_PART`, `VAL_PART`, `TEST_PART` — доли по умолчанию для переразбиения кадров в `fusion` (0.8 / 0.1 / 0.1); на практике чаще задают **`--fusion-split`**
- `parse_fusion_split_arg()` — разбор строки `train,val,test` для `--fusion-split`
- `RANDOM_SEED` - seed для генератора случайных чисел (12345)

### model_training_module.py
- `MODEL_VERSION` - версия модели по умолчанию (`"yolov8n"`)
- `EPOCHS` - количество эпох по умолчанию (50)
- `BATCH` - размер batch по умолчанию (16)
- `IMG_SIZE` - размер изображения по умолчанию (640)

### training_queue.py
- `BASE_DIR` — каталог пакета (рядом с модулем)
- `QUEUE_TXT` — запасной файл очереди, если workspace не задан (`training_queue.txt` в `BASE_DIR`)
- `STATUS_FILE` — запасной путь к статусам (`tmp/status.txt` в `BASE_DIR`)
- В режиме workspace пути задают `workspace_queue_path` / `workspace_queue_status_path` (`queue.txt`, `tmp/status.txt` в корне workspace)

