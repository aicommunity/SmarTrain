> English version: [../../reference/data-formats.md](../../reference/data-formats.md)

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

## Поддерживаемые структуры датасетов (официальные названия и internal IDs)

В `datasets_info.json` поле `structure` хранит внутренние идентификаторы структур.
Эти идентификаторы являются частью стабильного контракта проекта и не переименовываются.

- Официальное название: **YOLO split directories layout**
  - Internal ID: `split`
  - Типовая структура: `<dataset>/<train|val|test>/<images|labels>/...`
  - Примечание: общепринятая split-организация YOLO-датасета.

- Официальное название: **YOLO flat paired directories layout**
  - Internal ID: `flat`
  - Типовая структура: `<dataset>/images/*` и `<dataset>/labels/*`
  - Примечание: общепринятая парная flat-структура без явных split-папок.

- Официальное название: **YOLO flat with subset subfolders** (термин SmarTrain)
  - Internal ID: `subset_flat`
  - Ближайший общепринятый формат: YOLO split-style organization
  - Ключевое отличие: имена подпапок подмножеств произвольные, не ограничены `train/val/test`.

- Официальное название: **YOLO nested split under images/labels** (термин SmarTrain)
  - Internal ID: `nested_split`
  - Ближайший общепринятый формат: Ultralytics YOLO split directories layout
  - Ключевое отличие: сплиты вложены как `images/<split>` и `labels/<split>`.

- Официальное название: **Darknet YOLO dataset layout**
  - Internal ID: `darknet`
  - Типовая структура: `obj.data`, `obj.names`, `train.txt`/`valid.txt`, `obj_<subset>_data/`
  - Примечание: legacy-упаковка детекционного датасета в стиле Darknet.

- Официальное название: **CVAT for images 1.1 layout**
  - Internal ID: `cvat11`
  - Типовая структура: `annotations.xml` + `images/` (экспорт CVAT for images 1.1)
  - Примечание: в SmarTrain этот формат обрабатывается через внутренний идентификатор структуры `cvat11`.

## Terminology policy

- В пользовательской документации по возможности используйте **официальные названия форматов** (например, "CVAT for images 1.1 layout", "Darknet YOLO dataset layout").
- **Internal IDs** (`split`, `flat`, `subset_flat`, `nested_split`, `darknet`, `cvat11`) используйте только там, где речь о поведении кода, контрактах метаданных или `datasets_info.json`.
- Для SmarTrain-специфичных терминов (`subset_flat`, `nested_split`) всегда добавляйте:
  - ближайший общепринятый формат;
  - ключевое отличие в одну строку.
- Не переименовывайте internal IDs в документации как будто это внешние официальные стандарты; это стабильные значения контрактов проекта.
- В смешанном контексте при первом упоминании используйте обе формы: `CVAT for images 1.1 (internal ID: cvat11)`.

## Аннотации

Базовый формат YOLO bbox:

`class_id center_x center_y width height`

Также поддерживаются сегментационные полигоны в формате `class_id x1 y1 x2 y2 ...`.

## Очередь и статусы

- Файл очереди в рабочем каталоге: `queue.txt` (одна команда на строку).
- Файл статусов: `tmp/status.txt`.
- Базовые статусы исполнителя: `Waiting to be completed`, `Running`, `Done`, `Error`.
