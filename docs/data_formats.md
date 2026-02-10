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

**Определение**: Наличие папок `images` и `labels` на верхнем уровне, отсутствие папок train/val/test.

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

JSON файл с метаданными о всех доступных датасетах. Создается скриптом `datasets_json_former.py`.

### Структура

```json
{
    "dataset_name_1": {
        "classes": {
            "class_name": class_index
        },
        "structure": "split|flat|nested_split|darknet",
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
- `"nested_split"` - вложенное разделение
- `"darknet"` - формат Darknet

#### `elements_count`
Количество элементов (изображений/аннотаций) в датасете:
- Для `split` и `flat`: число (int)
- Для `nested_split`: массив чисел `[train_count, val_count, test_count]`
- Для `darknet`: число (int)

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

Файл используется скриптом `dataset_former.py` при объединении датасетов для приведения имен классов к единому виду перед фильтрацией.

---

## Формат training_queue.txt

Текстовый файл со списком задач для выполнения системой очереди.

### Формат строки

```
[python3] script_name.py [аргументы]
```

### Правила

1. Одна задача на строку
2. Команда может начинаться с `python3` или без него (добавляется автоматически)
3. Расширение `.py` добавляется автоматически, если отсутствует
4. Строки, начинающиеся с `#`, игнорируются (комментарии)
5. Пустые строки игнорируются

### Пример

```
# Создание объединенного датасета
dataset_former.py --target-path /path/to/output --classes "helmet,vest"

# Обучение модели yolov8n
model_training_module.py --data /path/to/dataset --model yolov8n --epochs 50

# Обучение модели yolov8s
python3 model_training_module.py --data /path/to/dataset --model yolov8s --epochs 100
```

---

## Формат tmp/status.txt

Текстовый файл с текущим статусом выполнения задач. Создается и обновляется автоматически скриптом `training_queue.py`.

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
dataset_former.py --target-path /path/to/output --classes "helmet,vest" | Выполнено
model_training_module.py --data /path/to/dataset --model yolov8n --epochs 50 | Выполняется
model_training_module.py --data /path/to/dataset --model yolov8s --epochs 100 | Ждет выполнения
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

