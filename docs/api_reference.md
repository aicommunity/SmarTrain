# Справочник API

## datasets_json_former.py

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
- `"flat"` - плоская структура images/labels
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
    "structure": "split|flat|nested_split|darknet",
    "elements_count": int_or_list
}
```
или `None` при ошибке

---

## dataset_former.py

### Функции

#### `safe_mkdir(path: str) -> None`
Создает директорию, если она не существует.

**Параметры**:
- `path` - путь к директории

---

#### `find_dataset_paths(dataset_path: str, structure: str, arg: bool = False) -> list[tuple[str, str]]`
Находит пути к изображениям и аннотациям в зависимости от структуры датасета.

**Параметры**:
- `dataset_path` - путь к датасету
- `structure` - тип структуры датасета
- `arg` - если `True`, исключает test из поиска

**Возвращает**: Список кортежей `(путь_к_изображениям, путь_к_аннотациям)`

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

#### `train_yolo(dataset_path: str, model_version: str, epochs: int, batch: int, img_size: int, target_dir: str) -> str`
Обучает модель YOLO на указанном датасете.

**Параметры**:
- `dataset_path` - путь к датасету (должен содержать `data.yaml`)
- `model_version` - версия модели (например, `"yolov8n"` или `"yolov8n.pt"`)
- `epochs` - количество эпох обучения
- `batch` - размер batch
- `img_size` - размер изображения
- `target_dir` - директория для сохранения результатов

**Возвращает**: Путь к директории с обученной моделью

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

**Обработка**:
- Добавляет `python3` если отсутствует
- Добавляет расширение `.py` если отсутствует
- Игнорирует строки, начинающиеся с `#`

---

#### `load_statuses() -> dict[str, str]`
Загружает статусы задач из файла.

**Возвращает**: Словарь `{задача: статус}`

---

#### `save_statuses(statuses: dict[str, str]) -> None`
Сохраняет статусы задач в файл.

**Параметры**:
- `statuses` - словарь `{задача: статус}`

---

## Константы

### datasets_json_former.py
- `BASE_DIR` - базовая директория с датасетами (по умолчанию: `/media/user/Data/IndustrialSafety/Datasets`)
- `OUTPUT_FILE` - имя выходного файла с информацией о датасетах (`"datasets_info.json"`)
- `OUTPUT_CLASS_NAMES_FILE` - имя выходного файла с именами классов (`"class_names.json"`)

### dataset_former.py
- `BASE_DIR` - базовая директория с датасетами
- `JSON_FILE` - имя файла с информацией о датасетах (`"datasets_info.json"`)
- `CLASS_NAMES_FILE` - имя файла с именами классов (`"class_names.json"`)
- `OUTPUT_DATASET_NAME` - имя выходного датасета (`"merged_dataset"`)
- `SELECTED_CLASSES` - список классов по умолчанию (`["hardhat", "no_hardhat"]`)
- `TRAIN_PART` - доля обучающей выборки (0.8)
- `VAL_PART` - доля валидационной выборки (0.1)
- `TEST_PART` - доля тестовой выборки (0.1)
- `RANDOM_SEED` - seed для генератора случайных чисел (12345)

### model_training_module.py
- `DATASET_PATH` - путь к датасету по умолчанию (`"/media/user/Data/IndustrialSafety/Datasets/HardHatSkz"`)
- `MODELS_BASE_DIR` - базовая директория для сохранения моделей (`"/media/user/Data/IndustrialSafety/Models"`)
- `MODEL_VERSION` - версия модели по умолчанию (`"yolov8n"`)
- `EPOCHS` - количество эпох по умолчанию (50)
- `BATCH` - размер batch по умолчанию (16)
- `IMG_SIZE` - размер изображения по умолчанию (640)

### training_queue.py
- `BASE_DIR` - директория скрипта
- `QUEUE_TXT` - путь к файлу очереди (`"training_queue.txt"`)
- `TMP_DIR` - временная директория (`"tmp"`)
- `STATUS_FILE` - путь к файлу статуса (`"tmp/status.txt"`)

