# Навигация по документации

## Основные документы

1. **[implementation.md](implementation.md)** - Детали реализации для разработчиков
   - Архитектура системы
   - Алгоритмы и структуры данных
   - Обработка ошибок
   - Рекомендации по расширению

2. **[data_formats.md](data_formats.md)** - Подробное описание форматов данных
   - Структура датасетов YOLO
   - Формат аннотаций
   - Формат конфигурационных файлов
   - Формат JSON файлов
   - Кодировка и стандарты

3. **[api_reference.md](api_reference.md)** - Справочник API
   - Описание всех функций
   - Параметры функций
   - Возвращаемые значения
   - Константы и настройки

4. **[examples.md](examples.md)** - Примеры использования
   - Полный цикл работы
   - Работа с разными форматами
   - Автоматизация процессов
   - Обработка ошибок

5. **[training_metadata.md](training_metadata.md)** - Метаданные обучения
   - Формат файла training_metadata.json
   - Структура метаданных
   - Отслеживание статуса и ошибок
   - Именование директорий с результатами

## Быстрая навигация по темам

### Начало работы
- [Быстрый старт](../README.md#быстрый-старт) - в корневом README.md
- [Основные команды](../README.md#основные-команды) - в корневом README.md

### Скрипты
- [datasets_json_former.py](api_reference.md#datasets_json_formerpy) - Анализ датасетов
- [dataset_former.py](api_reference.md#dataset_formerpy) - Объединение датасетов
- [model_training_module.py](api_reference.md#model_training_modulepy) - Обучение моделей
- [training_queue.py](api_reference.md#training_queuepy) - Система очереди

### Форматы данных
- [Структура датасетов](data_formats.md#структура-датасетов-yolo)
- [Формат аннотаций](data_formats.md#формат-аннотаций-yolo)
- [Формат data.yaml](data_formats.md#формат-datayaml)
- [Формат JSON файлов](data_formats.md#формат-datasets_infojson)

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
2. **Пользователь**: Изучите [examples.md](examples.md) для практических примеров
3. **Разработчик**: Ознакомьтесь с [api_reference.md](api_reference.md) для деталей реализации
4. **Администратор**: Изучите [data_formats.md](data_formats.md) для понимания форматов данных

## Поиск информации

### По задачам

**Хочу проанализировать датасеты:**
- [datasets_json_former.py](api_reference.md#datasets_json_formerpy)
- [Пример анализа](examples.md#пример-1-полный-цикл-работы-с-датасетами)

**Хочу объединить датасеты:**
- [dataset_former.py](api_reference.md#dataset_formerpy)
- [Пример объединения](examples.md#пример-1-полный-цикл-работы-с-датасетами)

**Хочу обучить модель:**
- [model_training_module.py](api_reference.md#model_training_modulepy)
- [Пример обучения](examples.md#пример-2-работа-с-несколькими-классами-сиз)

**Хочу поставить задачи в очередь:**
- [training_queue.py](api_reference.md#training_queuepy)
- [Пример использования очереди](examples.md#пример-3-использование-системы-очереди)

### По форматам

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
- [Устранение неполадок](../README.md#устранение-неполадок) - в корневом README.md
- [Примеры обработки ошибок](examples.md#пример-11-обработка-ошибок)

**Вопросы по форматам:**
- [data_formats.md](data_formats.md) - полное описание всех форматов

**Вопросы по API:**
- [api_reference.md](api_reference.md) - справочник всех функций

## Контакты и поддержка

Для получения дополнительной информации о YOLO моделях обратитесь к официальной документации:
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)

