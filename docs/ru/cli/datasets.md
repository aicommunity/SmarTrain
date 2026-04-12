> English version: [../../cli/datasets.md](../../cli/datasets.md)

# CLI: датасеты

## `scan`

Обновляет индекс датасетов и синхронизирует источники в рабочем каталоге.

- Выходные файлы: `datasets_info.json`, `class_names.json`, `datasets_scan_summary.json`.
- Поддерживает источники из `raw_data/`, `--dataset`, `--datasets-list`.
- Полезные режимы: `--mode refresh`, `--purge-processed-raw`.

## `normalize-data-yaml`

Перезаписывает `data.yaml` во всех вложенных каталогах `datasets/**/data.yaml`: убирает `path`, делает `train`/`val`/`test` относительными. Чужие абсолютные пути (другая машина) заменяются на `train/images`, `val/images` и т.п., если такие папки реально есть в этом датасете.

Пример: `smartrain normalize-data-yaml --workspace /path/to/workspace` или `--datasets-dir ... --dry-run`.

## `fusion`

Собирает новый датасет из нескольких источников:

- выбор входов: `--dataset` (повторяемый) или `--datasets` (CSV);
- управление классами: `--classes`, `--merge-classes`, `--common-classes-only`;
- разбиение: `--fusion-split train,val,test`.

## `augment`, `balance`, `orient`, `roi`

- `augment` — автономные аугментации с записью нового датасета;
- `balance` — балансировка классов; после балансировки по умолчанию действует `--eval-coverage` — при необходимости перераспределяет элементы между `train`/`val`/`test`, чтобы eval не был пустым и реже отсутствовали классы в `val`/`test`; отключить — `--no-eval-coverage`;
- `orient` — коррекция поворота кадров;
- `roi` — кроп по ROI-модели.

Все перечисленные команды формируют `dataset_passport.json` в новом каталоге датасета.

В `data.yaml` для переносимости корень датасета задаётся **каталогом, в котором лежит сам файл** (ключ `path` не обязателен); пути `train`/`val`/`test` — относительные к этому каталогу, без ведущего `./`, в духе Ultralytics.

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
