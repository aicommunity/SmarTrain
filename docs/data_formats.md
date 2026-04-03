# Форматы данных

## Структура датасетов YOLO

### 1. Split (разделенный)

Самый распространенный формат для датасетов YOLO. Изображения и аннотации разделены по папкам train/val/test.

```
dataset_name/
├── train/
│   ├── images/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── labels/
│       ├── image001.txt
│       ├── image002.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Определение**: Наличие папок `train`, `val`, `test` на верхнем уровне.

---

### 2. Flat (плоский)

Все изображения и аннотации находятся в общих папках без разделения на train/val/test.

```
dataset_name/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

**Определение**: Наличие папок `images` и `labels` на верхнем уровне; файлы лежат непосредственно в них; внутри `images/` нет подкаталогов `train` / `val` / `test` и нет общих пар подпапок с `labels/` (см. ниже).

---

### 2а. Subset flat (CVAT Ultralytics YOLO Detection 1.0)

Экспорт из [CVAT](https://docs.cvat.ai/docs/dataset_management/formats/) в формате **Ultralytics YOLO Detection 1.0** часто даёт одну или несколько **произвольно именованных** подпапок внутри `images/` и с теми же именами в `labels/` (например, имя задачи или набора: `Pads`, `Batch1` и т.д.).

```
dataset_name/
├── data.yaml
├── images/
│   └── Pads/                    # имя может быть любым
│       ├── shot001.jpg
│       └── ...
└── labels/
    └── Pads/
        ├── shot001.txt
        └── ...
```

**Определение**: есть `images/` и `labels/` на верхнем уровне; внутри `images/` **нет** подкаталогов с именами `train` / `val` / `test`; при этом существует хотя бы одна пара вложенных каталогов с **одинаковым именем** в `images/<name>/` и `labels/<name>/`. Допускается сочетание с файлами в корне `images/` и `labels/` (тогда учитываются и корень, и все такие пары).

В `datasets_info.json` поле `structure` для такого датасета: **`subset_flat`**.

---

### 3. Nested Split (вложенное разделение)

Разделение train/val/test находится внутри папок images и labels.

```
dataset_name/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── image001.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

**Определение**: Наличие папок `images` и `labels` на верхнем уровне, внутри которых находятся папки train/val/test.

---

### 4. Darknet (формат Darknet YOLO)

Формат, используемый оригинальной реализацией YOLO от Darknet. Все файлы находятся в одной папке.

```
dataset_name/
├── obj_train_data/
│   ├── image001.jpg
│   ├── image001.txt
│   ├── image002.jpg
│   ├── image002.txt
│   └── ...
├── obj.names
└── obj.data
```

**Определение**: Наличие папки `obj_train_data` и файла `obj.names` или `obj.data`.

**Формат obj.names**:
```
person
helmet
gloves
vest
```

**Формат obj.data**:
```
classes = 4
train = train.txt
valid = valid.txt
names = obj.names
backup = backup/
```

---

## Формат аннотаций YOLO

Файлы аннотаций имеют расширение `.txt` и содержат по одной строке на объект.

### Формат строки

```
class_id center_x center_y width height
```

**Параметры**:
- `class_id` - числовой идентификатор класса (начинается с 0)
- `center_x` - координата X центра bounding box (нормализованная, 0.0 - 1.0)
- `center_y` - координата Y центра bounding box (нормализованная, 0.0 - 1.0)
- `width` - ширина bounding box (нормализованная, 0.0 - 1.0)
- `height` - высота bounding box (нормализованная, 0.0 - 1.0)

### Пример файла аннотации

```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.2
0 0.8 0.2 0.1 0.15
```

**Интерпретация**:
- Первая строка: класс 0, центр в (0.5, 0.5), размер 0.2×0.3
- Вторая строка: класс 1, центр в (0.3, 0.7), размер 0.15×0.2
- Третья строка: класс 0, центр в (0.8, 0.2), размер 0.1×0.15

### Формат сегментации (YOLO segment)

Строка: `class_id x1 y1 x2 y2 ...` — нормализованные координаты вершин полигона (парные `x`, `y` в долях ширины и высоты изображения). Минимум три вершины (шесть чисел после класса). Скрипт `dataset_roi_yolo.py` при кропе пересчитывает эти координаты в систему нового изображения и прижимает точки к границам кропа.

### Формула нормализации

Если изображение имеет размеры `img_width × img_height`, а bounding box имеет координаты:
- Левый верхний угол: `(x_min, y_min)`
- Правый нижний угол: `(x_max, y_max)`

То нормализованные координаты вычисляются как:
```
center_x = (x_min + x_max) / 2 / img_width
center_y = (y_min + y_max) / 2 / img_height
width = (x_max - x_min) / img_width
height = (y_max - y_min) / img_height
```

---

## Формат data.yaml

Файл конфигурации датасета для YOLO моделей (YOLOv8, YOLOv11).

### Базовая структура

```yaml
train: ./train/images
val: ./valid/images
test: ./test/images

nc: 3
names: ['helmet', 'gloves', 'vest']
```

### Параметры

- `train` - путь к папке с обучающими изображениями (относительно файла yaml)
- `val` - путь к папке с валидационными изображениями
- `test` - путь к папке с тестовыми изображениями (опционально)
- `nc` - количество классов (number of classes)
- `names` - список имен классов в порядке их индексов

### Альтернативный формат с абсолютными путями

```yaml
train: /absolute/path/to/train/images
val: /absolute/path/to/valid/images
test: /absolute/path/to/test/images

nc: 3
names: ['helmet', 'gloves', 'vest']
```

### Расширенный формат

```yaml
path: /path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: helmet
  1: gloves
  2: vest
```

---

## Формат datasets_info.json

JSON файл с метаданными о всех доступных датасетах. Создаётся командой **`smartrain datasets-json`** (модуль `datasets_json_former`).

### Источники датасетов для `datasets-json`

По умолчанию команда сканирует подкаталоги `source_datasets/` (или путь из `--datasets-path`).

Дополнительно можно передать файл списка:

```bash
smartrain datasets-json --datasets-list /abs/path/to/datasets_list.txt
```

Правила для `datasets_list.txt`:
- одна строка = один путь к датасету;
- поддерживаются пути к директории датасета и к архиву `.zip`;
- пустые строки и строки-комментарии (`# ...`) игнорируются;
- относительные пути интерпретируются относительно каталога, где лежит `datasets_list.txt`.

В workspace-режиме (`smartrain datasets-json` без `--datasets-path`) файл `source_datasets/datasets_list.txt` используется автоматически, если он есть.

### Структура

```json
{
    "dataset_name_1": {
        "classes": {
            "class_name": class_index
        },
        "structure": "split|flat|subset_flat|nested_split|darknet",
        "elements_count": number_or_array
    },
    "dataset_name_2": {
        ...
    }
}
```

### Поля

#### `classes`
Словарь соответствия имен классов их числовым индексам в датасете.

**Пример**:
```json
"classes": {
    "helmet": 0,
    "gloves": 1,
    "vest": 2
}
```

#### `structure`
Тип структуры организации датасета:
- `"split"` - разделение на train/val/test
- `"flat"` - плоская структура
- `"subset_flat"` - `images/<подпапка>/` и `labels/<подпапка>/` (см. выше)
- `"nested_split"` - вложенное разделение
- `"darknet"` - формат Darknet
- `"cvat11"` - распакованный экспорт CVAT 1.1: `annotations.xml` рядом с папкой `images/` (формат Images + bbox). Поддерживается нативно в `datasets-json` и `dataset-former` (в merge используются временные YOLO `.txt`, генерируемые из XML).

#### `elements_count`
Количество элементов (изображений/аннотаций) в датасете:
- Для `split` и `flat`: число (int)
- Для `nested_split`: массив чисел `[train_count, val_count, test_count]`
- Для `darknet`: число (int)

#### Опциональные поля (не перезаписываются сканером)

При повторном запуске `datasets_json_former.py` существующий `datasets_info.json` читается и для каждого датасета, который снова найден на диске, в новую запись **переносятся** поля `roi_auto`, `tags` и **`data_path`** из старого файла. Остальные поля берутся из свежего скана.

##### `data_path`

Строка: **абсолютный** путь к корню датасета на диске или путь **относительно корня workspace** (`SMART_TRAIN_WORKSPACE`). Если ключа нет, корень данных считается равным `<каталог_каталога>/<ключ_записи>` (например `source_datasets/MyDataset` в режиме workspace).

##### `tags`

Список строк-пометок, например `["roi_yolo"]`. Можно использовать для своих соглашений; обязательным для `dataset_roi_yolo.py` является наличие параметров модели (см. `roi_auto` или CLI).

##### `roi_auto`

Параметры автоматического кропа по ROI (скрипт `dataset_roi_yolo.py`):

```json
"roi_auto": {
    "mode": "yolo_detect",
    "weights": "/abs/path/best.pt",
    "conf": 0.25,
    "pad_px": 32,
    "class_ids": null,
    "roi_policy": "union"
}
```

- `mode`: `yolo_detect` или `yolo_segment` (для сегмент-модели ROI строится по box из Ultralytics).
- `weights`: путь к весам `.pt`.
- `conf`: порог уверенности инференса.
- `pad_px`: расширение прямоугольника ROI на столько пикселей с каждой стороны (перед зажатием в границы кадра).
- `class_ids`: `null` — учитываются все классы модели; иначе массив целых id классов модели, по которым строится ROI.
- `roi_policy`:
  - `union` — один кроп по объединённому AABB всех отфильтрованных боксов;
  - `largest` — кроп по боксу максимальной площади;
  - `best_conf` — кроп по боксу с максимальным `conf`;
  - `per_box` — отдельное изображение и файл меток на каждый бокс; имена: `{stem}_split_1`, `{stem}_split_2`, …

Путь к `datasets_info.json`: каталог **рядом с папкой датасета** (тот же родитель, что и `{dataset_name}`). Запуск:

```bash
smartrain roi \
  --dataset-name MyDataset \
  --source-path /data/datasets_parent \
  --output-path /data/MyDataset_cropped \
  [--datasets-info-path /data/datasets_parent]
```

Если задан `--datasets-info-path`, он должен быть тем же родителем, что и `--source-path`, в котором лежит `{dataset_name}` (проверяется совпадение каталогов). Файл метаданных: `{parent}/datasets_info.json`.

Поведение при отсутствии детекций: `--on-empty full_image` (по умолчанию — копия всего кадра и пересчёт меток), `skip` — не писать эту пару в выход, `fail` — завершение с ошибкой.

### Пример полного файла

```json
{
    "archive": {
        "classes": {
            "person": 0,
            "helmet": 1,
            "gloves": 2
        },
        "structure": "flat",
        "elements_count": 8099
    },
    "construction-ppe": {
        "classes": {
            "helmet": 0,
            "gloves": 1,
            "vest": 2
        },
        "structure": "nested_split",
        "elements_count": [1416, 1426, 0]
    }
}
```

---

## Формат class_names.json

JSON файл для нормализации имен классов между различными датасетами.

### Структура

```json
{
    "original_name_1": "normalized_name",
    "original_name_2": "normalized_name",
    "normalized_name": "normalized_name"
}
```

### Принцип работы

Ключ - исходное имя класса из датасета, значение - нормализованное имя, к которому оно приводится.

**Пример**:
```json
{
    "Helmet": "helmet",
    "helmet": "helmet",
    "Hard Hat": "helmet",
    "hardhat": "helmet",
    "Gloves": "gloves",
    "gloves": "gloves",
    "Glove": "gloves"
}
```

Все варианты написания (`Helmet`, `helmet`, `Hard Hat`, `hardhat`) приводятся к единому виду `helmet`.

### Использование

Файл используется модулем `dataset_former` / командой **`smartrain dataset-former`** при объединении датасетов для приведения имён классов к единому виду перед фильтрацией.

---

## Формат queue.txt (файл очереди)

Текстовый файл со списком задач для **`smartrain queue-run`** (или `smartrain queue run`). По умолчанию путь: **`queue.txt`** в корне workspace (см. `workspace_queue_path` в [`smartrain/workspace_paths.py`](../smartrain/workspace_paths.py)); иначе см. `--queue-file` у исполнителя.

### Формат строки

Рекомендуемый вид — полная shell-команда:

```
smartrain <подкоманда> [аргументы...]
```

### Правила (как в [`process_line()`](../smartrain/training_queue.py))

1. Одна задача на строку.
2. Если строка начинается с **`smartrain`** (или пути к бинарнику `smartrain`) — выполняется как есть.
3. Если начинается с **`python3`** / **`python`** — выполняется как есть.
4. Иначе для обратной совместимости строка нормализуется к виду `python3 script.py ...` (к первому токену дописывается `.py`, если нужно). Предпочтительно писать явно **`smartrain ...`**.
5. Строки с `#` в начале и пустые строки игнорируются.

### Пример

```
# Создание объединённого датасета
smartrain dataset-former --target-path /path/to/output --classes "helmet,vest"

# Обучение
smartrain train --data /path/to/dataset --model yolov8n --epochs 50 -y
smartrain train --data /path/to/dataset --model yolov8s --epochs 100 -y
```

---

## Формат tmp/status.txt

Текстовый файл с текущим статусом выполнения задач. Создаётся и обновляется исполнителем очереди ([`smartrain/training_queue.py`](../smartrain/training_queue.py)); путь по умолчанию — **`tmp/status.txt`** внутри workspace.

### Формат строки

```
задача | статус
```

### Статусы

- `Ждет выполнения` - задача в очереди, но еще не запущена
- `Выполняется` - задача выполняется в данный момент
- `Выполнено` - задача успешно завершена (код возврата 0)
- `Ошибка` - задача завершилась с ошибкой (код возврата != 0)

### Пример

```
smartrain dataset-former --target-path /path/to/output --classes "helmet,vest" | Выполнено
smartrain train --data /path/to/dataset --model yolov8n --epochs 50 | Выполняется
smartrain train --data /path/to/dataset --model yolov8s --epochs 100 | Ждет выполнения
```

---

## Поддерживаемые форматы изображений

Проект поддерживает следующие форматы изображений:
- `.jpg` / `.jpeg`
- `.png`

При поиске соответствия между изображениями и аннотациями скрипты проверяют все эти расширения.

---

## Кодировка файлов

Все текстовые файлы используют кодировку **UTF-8** для корректной работы с:
- Русскими именами файлов
- Русскими именами классов
- Специальными символами в путях

Скрипты автоматически настраивают кодировку вывода:
```python
sys.stdout.reconfigure(encoding='utf-8')
```

