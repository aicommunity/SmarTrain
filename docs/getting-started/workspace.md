# Рабочий каталог (workspace)

`smartrain` ориентирован на единый корень рабочего каталога:

- через переменную `SMART_TRAIN_WORKSPACE`;
- или через глобальный флаг `smartrain --workspace /path/to/ws ...`.

`--workspace` имеет приоритет над переменной окружения.

## Структура каталогов

- `raw_data/` — внешние источники датасетов;
- `datasets/` — рабочие датасеты и индекс (`datasets_info.json`, `class_names.json`);
- `runs/` — результаты обучения;
- `analytics/` — артефакты аналитики (`analyze export-table` и др.);
- `models/` — промоутированные модели (`registry models-add`);
- `tmp/` — служебные файлы, включая `tmp/status.txt`.

Файл очереди по умолчанию: `queue.txt` в корне рабочего каталога.

## Инициализация

```bash
export SMART_TRAIN_WORKSPACE=/path/to/workspace
smartrain deploy
```
