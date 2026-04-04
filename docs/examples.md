# Примеры использования

Во всех примерах предполагается установленная команда **`smartrain`** (`pip install -e .` из корня репозитория). Это соответствует коду в [`smartrain/cli.py`](../smartrain/cli.py). Прямой запуск `python smartrain/datasets_json_former.py` устарел; при необходимости вызывайте функции `main` модулей из Python или `python -m smartrain` с нужной подкомандой.

## Пример 1: Полный цикл работы с датасетами

### Шаг 1: Анализ существующих датасетов

Предположим, у вас есть несколько датасетов в директории `/data/datasets`:
- `archive/`
- `construction-ppe/`
- `PPE Dataset for Workplace Safety.v2i.yolov11/`

Запустите анализ:
```bash
smartrain scan --datasets-path /data/datasets --output-path .
```

Результат: созданы файлы `datasets_info.json` и `class_names.json` в текущей директории.

### Шаг 2: Просмотр информации о датасетах

Откройте `datasets_info.json` для просмотра доступных классов:
```json
{
    "archive": {
        "classes": {"person": 0, "helmet": 1, "gloves": 2},
        "structure": "flat",
        "elements_count": 8099
    },
    "construction-ppe": {
        "classes": {"helmet": 0, "gloves": 1, "vest": 2},
        "structure": "nested_split",
        "elements_count": [1416, 1426]
    }
}
```

### Шаг 3: Создание объединенного датасета

Создайте датасет только с классами `helmet` и `gloves`:
```bash
smartrain fusion \
    --source-path /data/datasets \
    --target-path /data/merged_helmet_gloves \
    --classes "helmet,gloves" \
    --datasets-info-path .
```

Результат: создан новый датасет `/data/merged_helmet_gloves` с:
- Структурой train/valid/test
- Только классами helmet и gloves
- Переиндексированными классами (0=helmet, 1=gloves)
- Файлом `data.yaml`

### Шаг 4: Обучение модели

Обучите модель YOLOv8n на созданном датасете:
```bash
smartrain train \
    --data /data/merged_helmet_gloves \
    --model yolov8n \
    --epochs 50 \
    --batch 16 \
    --target-path /data/models
```

Результат: обученная модель сохранена в `/data/models/merged_helmet_gloves/yolov8n_50epochs/`

---

## Пример 2: Работа с несколькими классами СИЗ

### Создание датасета с полным набором СИЗ

```bash
smartrain fusion \
    --classes "helmet,gloves,vest,boots,goggles,mask" \
    --target-path /data/full_ppe_dataset \
    --exclude-test
```

**Примечание**: Флаг `--exclude-test` исключает тестовые данные из исходных датасетов, чтобы использовать их только для финального тестирования.

### Обучение разных моделей

```bash
# Маленькая модель для быстрого прототипирования
smartrain train \
    --data /data/full_ppe_dataset \
    --model yolov8n \
    --epochs 30 \
    --batch 32

# Средняя модель для баланса скорости и точности
smartrain train \
    --data /data/full_ppe_dataset \
    --model yolov8s \
    --epochs 50 \
    --batch 16

# Большая модель для максимальной точности
smartrain train \
    --data /data/full_ppe_dataset \
    --model yolov8l \
    --epochs 100 \
    --batch 8
```

---

## Пример 3: Использование системы очереди

### Подготовка файла очереди

По умолчанию исполнитель читает **`queue.txt`** в корне workspace (см. [`workspace_paths.workspace_queue_path`](../smartrain/workspace_paths.py)). Создайте или отредактируйте этот файл:

```
# Шаг 1: Создание датасета
smartrain fusion --classes "helmet,vest" --target-path /data/helmet_vest_dataset

# Шаг 2: Обучение yolov8n
smartrain train --data /data/helmet_vest_dataset --model yolov8n --epochs 50

# Шаг 3: Обучение yolov8s
smartrain train --data /data/helmet_vest_dataset --model yolov8s --epochs 50

# Шаг 4: Обучение yolov8m
smartrain train --data /data/helmet_vest_dataset --model yolov8m --epochs 50
```

### Запуск очереди

```bash
smartrain queue-run
```

Система автоматически:
1. Откроет окно терминала с обновлением статуса
2. Выполнит задачи последовательно
3. Обновит статус каждой задачи

---

## Пример 4: Работа с форматом Darknet

Если у вас есть датасет в формате Darknet:

```
darknet_dataset/
├── obj_train_data/
│   ├── img001.jpg
│   ├── img001.txt
│   └── ...
├── obj.names
└── obj.data
```

Команда **`smartrain scan`** (модуль `datasets_json_former`) автоматически определит структуру:

```bash
smartrain scan --datasets-path /path/to/darknet_dataset
```

Результат в `datasets_info.json`:
```json
{
    "darknet_dataset": {
        "classes": {
            "person": 0,
            "helmet": 1
        },
        "structure": "darknet",
        "elements_count": 1500
    }
}
```

---

## Пример 5: Тестирование обученной модели

После обучения модели можно протестировать ее на тестовом наборе:

```bash
smartrain train \
    --test-only \
    --model-dir /data/models/merged_helmet_gloves/yolov8n_50epochs \
    --data /data/merged_helmet_gloves
```

Результат: создан файл `test_metrics.csv` с метриками производительности модели.

---

## Пример 6: Работа с вложенной структурой (nested_split)

Для датасета со структурой:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Скрипты автоматически определяют структуру и обрабатывают ее корректно:

```bash
smartrain scan --datasets-path /path/to/dataset
```

Результат:
```json
{
    "dataset": {
        "classes": {...},
        "structure": "nested_split",
        "elements_count": [1416, 1426, 500]
    }
}
```

---

## Пример 7: Фильтрация с нормализацией классов

Предположим, в разных датасетах класс "каска" называется по-разному:
- `Helmet` в одном датасете
- `helmet` в другом
- `Hard Hat` в третьем

Файл `class_names.json` содержит маппинг:
```json
{
    "Helmet": "helmet",
    "helmet": "helmet",
    "Hard Hat": "helmet"
}
```

При создании объединенного датасета все варианты будут нормализованы:

```bash
smartrain fusion --classes "helmet" --target-path /data/helmet_only
```

Все три варианта будут объединены в один класс `helmet`.

---

## Пример 8: Настройка пропорций разделения

Для изменения пропорций train/val/test отредактируйте константы в модуле `smartrain/dataset_former.py`:

```python
TRAIN_PART = 0.7  # 70%
VAL_PART = 0.2    # 20%
TEST_PART = 0.1   # 10%
```

Или используйте только train и val (без test), установив `TEST_PART = 0.0`.

---

## Пример 9: Пакетное обучение с разными конфигурациями

Создайте `queue.txt` в корне workspace для экспериментов:

```
# Эксперимент 1: Маленькая модель, мало эпох
smartrain train --data /data/dataset --model yolov8n --epochs 10 --batch 32

# Эксперимент 2: Маленькая модель, много эпох
smartrain train --data /data/dataset --model yolov8n --epochs 100 --batch 16

# Эксперимент 3: Средняя модель
smartrain train --data /data/dataset --model yolov8s --epochs 50 --batch 16

# Эксперимент 4: Большая модель
smartrain train --data /data/dataset --model yolov8l --epochs 50 --batch 8

# Эксперимент 5: YOLOv11
smartrain train --data /data/dataset --model yolov11n --epochs 50 --batch 16
```

Запустите очередь и оставьте работать на ночь.

---

## Пример 10: Работа с относительными путями

Все пути можно указывать относительно текущей директории:

```bash
# Если вы находитесь в /home/user/project
smartrain fusion \
    --source-path ./datasets \
    --target-path ./output/merged \
    --classes "helmet,gloves"
```

Или использовать абсолютные пути для большей надежности:

```bash
smartrain fusion \
    --source-path /home/user/project/datasets \
    --target-path /home/user/project/output/merged \
    --classes "helmet,gloves"
```

---

## Пример 11: Обработка ошибок

### Ошибка: Датасет не найден

```bash
$ smartrain scan --datasets-path /wrong/path
[ERROR] Папка '/wrong/path' не найдена.
```

**Решение**: Проверьте путь к датасетам.

### Ошибка: Классы не найдены

```bash
$ smartrain fusion --classes "nonexistent_class"
[ERROR] Ни один датасет не содержит все выбранные классы.
```

**Решение**: 
1. Проверьте доступные классы в `datasets_info.json`
2. Проверьте нормализацию в `class_names.json`
3. Убедитесь, что используете правильные имена классов

### Ошибка: YAML файл не найден

```bash
$ smartrain train --data /path/to/dataset
[ERROR] Не найден yaml файл: /path/to/dataset/data.yaml
```

**Решение**: Убедитесь, что датасет содержит файл `data.yaml` или создайте его вручную.

---

## Пример 12: Интеграция с другими инструментами

### Экспорт метрик в CSV

После тестирования модели метрики сохраняются в CSV:

```python
import pandas as pd

# Загрузка метрик
metrics = pd.read_csv('/path/to/model/test_metrics.csv')
print(metrics.head())
```

### Использование обученной модели

```python
from ultralytics import YOLO

# Загрузка обученной модели
model = YOLO('/path/to/model/train/weights/best.pt')

# Предсказание на изображении
results = model('/path/to/image.jpg')

# Визуализация результатов
results[0].show()
```

---

## Пример 14: Слияние классов при сборке датасета

Итоговые имена классов в `--classes`; `--merge-classes` задаёт соответствие «несколько исходных → одно имя из --classes»:

```bash
smartrain fusion \
    --source-path /data/datasets \
    --target-path /data/merged_head \
    --classes "head_ppe,vest" \
    --merge-classes "helmet, hard_hat" head_ppe \
    --datasets-info-path /data/datasets
```

## Пример 15: Очередь через CLI и анализ прогонов

```bash
smartrain queue add -- smartrain train --data /data/m --model yolov8n -y --epochs 2
smartrain queue list
smartrain analyze scan --models-root /path/to/Models
smartrain analyze compare --baseline /path/to/run1 --others /path/to/run2 -o cmp.csv --out-png cmp.png
```

## Пример 13: Автоматизация с помощью скриптов

Создайте bash скрипт для автоматизации:

```bash
#!/bin/bash
# auto_train.sh

DATASETS_PATH="/data/datasets"
OUTPUT_PATH="/data/output"
CLASSES="helmet,gloves,vest"

# Шаг 1: Анализ датасетов
echo "Анализ датасетов..."
smartrain scan --datasets-path $DATASETS_PATH

# Шаг 2: Создание объединенного датасета
echo "Создание объединенного датасета..."
smartrain fusion \
    --source-path $DATASETS_PATH \
    --target-path $OUTPUT_PATH/merged \
    --classes $CLASSES

# Шаг 3: Обучение модели
echo "Обучение модели..."
smartrain train \
    --data $OUTPUT_PATH/merged \
    --model yolov8n \
    --epochs 50

echo "Готово!"
```

Сделайте скрипт исполняемым и запустите:
```bash
chmod +x auto_train.sh
./auto_train.sh
```

