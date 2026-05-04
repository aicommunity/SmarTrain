> English version: [../../cli/registry.md](../../cli/registry.md)

# CLI: реестр

`smartrain registry` управляет артефактами запусков и каталогом моделей в рабочем каталоге.

## Подкоманды

- `runs-list`
- `runs-info`
- `runs-metrics`
- `models-add`
- `models-list`
- `models-info`
- `models-remove`

## `models-add`

Переносит run в `models/<friendly_name>/` и создаёт `model_manifest.json`.

В бандл копируется (если есть у run):

- каталог `models/` — все веса и sidecar-файлы, с той же структурой, что у run;
- `train/` и все каталоги `train-*/` (например `train-ultralytics/`), **без** подкаталога `weights/` с чекпойнтами;
- унаследованный корневой `test/` и целиком дерево `tests/` (каноничная вёрстка тестов, метрики, манифесты);
- `training_metadata.json` (пути к основному `.pt` приводятся к относительным от корня бандла), `test_metrics*.csv` из корня run и `_runtime_data_*.yaml` из корня run или `tmp/`.

Поле `weights_file` в манифесте — путь относительно каталога бандла (обычно `models/<имя_каталога_run>.pt`). Ранее промотированные каталоги с файлом `<friendly_name>.pt` в корне не меняются.

## Примеры

```bash
smartrain registry runs-list
smartrain registry runs-info 3
smartrain registry models-add 3
smartrain registry models-list
```
