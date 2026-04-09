# CLI: датасеты

## `scan`

Обновляет индекс датасетов и синхронизирует источники в рабочем каталоге.

- Выходные файлы: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Поддерживает источники из `raw_data/`, `--dataset`, `--datasets-list`.
- Полезные режимы: `--mode refresh`, `--purge-processed-raw`.

## `fusion`

Собирает новый датасет из нескольких источников:

- выбор входов: `--dataset` (повторяемый) или `--datasets` (CSV);
- управление классами: `--classes`, `--merge-classes`, `--common-classes-only`;
- разбиение: `--fusion-split train,val,test`.

## `augment`, `balance`, `orient`, `roi`

- `augment` — автономные аугментации с записью нового датасета;
- `balance` — балансировка классов;
- `orient` — коррекция поворота кадров;
- `roi` — кроп по ROI-модели.

Все перечисленные команды формируют `dataset_passport.json` в новом каталоге датасета.

## `hash`

Проверка и вычисление хеша структуры датасета:

```bash
smartrain hash --dataset my_dataset
smartrain hash /abs/path/to/dataset --validate a1b2c3d4
```

Коды выхода `--validate`: `0` совпадение, `1` несовпадение, `2` ошибка.

## `stats`

Подкоманды:

- `classes`
- `datasets`
- `compare`
