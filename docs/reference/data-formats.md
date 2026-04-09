# Справочник: форматы данных

## Основные артефакты

- `datasets/datasets_info.json` — каталог датасетов.
- `datasets/class_names.json` — нормализация имён классов.
- `datasets/datasets_scan_summary.json` — сводка изменений после `scan`.
- `datasets/<dataset>/dataset_passport.json` — паспорт преобразований.
- `runs/.../training_metadata.json` — метаданные обучения.
- `queue.txt` и `tmp/status.txt` — очередь и её статусы.

## Ключевые поля каталогов

- `datasets_info.json`: `classes`, `structure`, `elements_count`.
- Служебные поля синхронизации: `dataset_hash`, `source_hash`, `source_ref`, `source_signature`, `modified`.
- Опциональные поля: `data_path`, `tags`, `roi_auto`.

## Поддерживаемые структуры датасетов

- `split`
- `flat`
- `subset_flat`
- `nested_split`
- `darknet`
- `cvat11`

## Аннотации

Базовый формат YOLO bbox:

`class_id center_x center_y width height`

Также поддерживаются сегментационные полигоны в формате `class_id x1 y1 x2 y2 ...`.

## Очередь и статусы

- Файл очереди в рабочем каталоге: `queue.txt` (одна команда на строку).
- Файл статусов: `tmp/status.txt`.
- Базовые статусы исполнителя: `Waiting to be completed`, `Running`, `Done`, `Error`.
