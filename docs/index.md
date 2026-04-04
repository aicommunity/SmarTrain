# Навигация по документации

## Основные документы

1. **[datasets.md](datasets.md)** — **Описание датасетов и каталога**
   - Где находится описание конкретных наборов (`datasets_info.json` после `scan`)
   - Ссылки на структуру YOLO, аннотации, `data.yaml`, workspace
   - Как обновить метаданные датасетов

2. **[implementation.md](implementation.md)** - Детали реализации для разработчиков
   - Архитектура системы
   - Алгоритмы и структуры данных
   - Обработка ошибок
   - Рекомендации по расширению

3. **[data_formats.md](data_formats.md)** - Подробное описание форматов данных
   - Структура датасетов YOLO
   - Формат аннотаций
   - Формат конфигурационных файлов
   - Формат JSON файлов
   - Кодировка и стандарты

4. **[api_reference.md](api_reference.md)** - Справочник API
   - В начале файла: разделы про [`cli.py` и `cli_argparse`](api_reference.md) (точка входа `smartrain` и форматтер справки)
   - Описание функций модулей, параметры, возвращаемые значения
   - Константы и настройки

5. **[examples.md](examples.md)** - Примеры использования
   - Полный цикл работы
   - Работа с разными форматами
   - Автоматизация процессов
   - Обработка ошибок

6. **[training_metadata.md](training_metadata.md)** - Метаданные обучения
   - Формат файла training_metadata.json
   - Структура метаданных
   - Отслеживание статуса и ошибок
   - Именование директорий с результатами

7. **[workspace.md](workspace.md)** - Единый корень `SMART_TRAIN_WORKSPACE`, каталоги `source_datasets` / `work_datasets` / `runs` / `analytics` / `models`, поле `data_path`

## Быстрая навигация по темам

### Начало работы
- [Быстрый старт](../README.md#быстрый-старт) — корневой README
- [Команды CLI](../README.md#команды-cli) — таблица подкоманд `smartrain`
- [Workspace и переменные](../README.md#workspace) — `SMART_TRAIN_WORKSPACE`, `deploy`

### Точка входа
- Единая команда **`smartrain`** ([`smartrain/cli.py`](../smartrain/cli.py)): Typer + делегирование в модули `datasets_json_former`, `dataset_former`, `model_training_module`, и т.д.
- Справка: `smartrain --help`, `smartrain <команда> --help` (полный текст argparse, в т.ч. для `queue list --help`).

### Модули (вызываются через CLI)
- [datasets_json_former.py](api_reference.md#datasets_json_formerpy) - Анализ датасетов
- [dataset_former.py](api_reference.md#dataset_formerpy) - Объединение датасетов
- [model_training_module.py](api_reference.md#model_training_modulepy) - Обучение моделей
- [dataset_hash.py](api_reference.md#dataset_hashpy) - Хеш датасета
- [training_queue.py](api_reference.md#training_queuepy) - Система очереди
- [training_queue_cli.py](api_reference.md#training_queue_clipy) - CLI очереди (`list`, `add`, `remove`, `run`, …)
- [results_analyzer.py](api_reference.md#results_analyzerpy) - Сводки и сравнение прогонов

### Датасеты и форматы
- **[О датасетах: каталог и навигация](datasets.md)** — с чего начать: `datasets_info.json`, ссылки на детали
- [Каталог в JSON (`datasets_info.json`)](data_formats.md#формат-datasets_infojson) — поля по каждому набору
- [Структура датасетов YOLO](data_formats.md#структура-датасетов-yolo)
- [Формат аннотаций](data_formats.md#формат-аннотаций-yolo)
- [Формат data.yaml](data_formats.md#формат-datayaml)
- [Другие JSON (`class_names.json` и т.д.)](data_formats.md#формат-class_namesjson)

### Примеры
- [Полный цикл работы](examples.md#пример-1-полный-цикл-работы-с-датасетами)
- [Работа с несколькими классами](examples.md#пример-2-работа-с-несколькими-классами-сиз)
- [Использование очереди](examples.md#пример-3-использование-системы-очереди)
- [Автоматизация](examples.md#пример-13-автоматизация-с-помощью-скриптов)

### API
- [datasets_json_former.py функции](api_reference.md#datasets_json_formerpy)
- [dataset_former.py функции](api_reference.md#dataset_formerpy)
- [model_training_module.py функции](api_reference.md#model_training_modulepy)
- [training_queue.py функции](api_reference.md#training_queuepy)

## Рекомендуемый порядок чтения

1. **Новичок**: Начните с корневого [README.md](../README.md), раздел "Быстрый старт"
2. **Работа с датасетами**: [datasets.md](datasets.md) → при необходимости [data_formats.md](data_formats.md) и [workspace.md](workspace.md)
3. **Пользователь**: Изучите [examples.md](examples.md) для практических примеров
4. **Разработчик**: Ознакомьтесь с [api_reference.md](api_reference.md) для деталей реализации
5. **Администратор**: [data_formats.md](data_formats.md) для полной спецификации форматов

## Поиск информации

### По задачам

**Хочу прочитать описание датасетов (что за наборы, классы, пути):**
- [datasets.md](datasets.md) — навигация; итоговый каталог после сканирования: **`datasets_info.json`** (см. [формат](data_formats.md#формат-datasets_infojson))
- Команда: `smartrain scan`

**Хочу проанализировать датасеты:**
- [datasets_json_former.py](api_reference.md#datasets_json_formerpy)
- [Пример анализа](examples.md#пример-1-полный-цикл-работы-с-датасетами)

**Хочу объединить датасеты:**
- Команда: `smartrain fusion`
- [dataset_former.py](api_reference.md#dataset_formerpy)
- [Пример объединения](examples.md#пример-1-полный-цикл-работы-с-датасетами)

**Хочу обучить модель:**
- [model_training_module.py](api_reference.md#model_training_modulepy)
- [Пример обучения](examples.md#пример-2-работа-с-несколькими-классами-сиз)

**Хочу поставить задачи в очередь:**
- [training_queue.py](api_reference.md#training_queuepy)
- [Пример использования очереди](examples.md#пример-3-использование-системы-очереди)

### По форматам и описанию датасетов

**Описание конкретных датасетов (имена, классы, пути):**
- [datasets.md](datasets.md) — точка входа; фактическое содержимое — **`datasets_info.json`**, см. [формат](data_formats.md#формат-datasets_infojson)

**Формат датасета:**
- [Структура датасетов YOLO](data_formats.md#структура-датасетов-yolo)

**Формат аннотаций:**
- [Формат аннотаций YOLO](data_formats.md#формат-аннотаций-yolo)

**Конфигурационные файлы:**
- [data.yaml](data_formats.md#формат-datayaml)
- [datasets_info.json](data_formats.md#формат-datasets_infojson)
- [class_names.json](data_formats.md#формат-class_namesjson)
- [training_metadata.json](training_metadata.md#формат-файла)

### По проблемам

**Ошибки при работе:**
- [Устранение неполадок](../README.md#устранение-неполадок) — корневой README
- [Примеры обработки ошибок](examples.md#пример-11-обработка-ошибок)

**Вопросы по форматам:**
- [data_formats.md](data_formats.md) - полное описание всех форматов

**Вопросы по API:**
- [api_reference.md](api_reference.md) - справочник всех функций

## Контакты и поддержка

Для получения дополнительной информации о YOLO моделях обратитесь к официальной документации:
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)

