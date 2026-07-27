> English version: [../../cli/registry.md](../../cli/registry.md)

# CLI: registry

`smartrain registry` управляет **инвентарём runs** и **registry-бандлами** промоутированных моделей.

Это **не** то же самое, что `smartrain model release` / `model comment` (каталог релизов `models/<dataset>/<run_id>/` со stem `detect_*` и `releases_manifest.json`). См. [`overview.md`](overview.md) и [`../../refactor/run-layout.md`](../../refactor/run-layout.md).

## Подкоманды

- `runs-list`
- `runs-info`
- `runs-metrics`
- `models-add`
- `models-list`
- `models-info`
- `models-remove`

## `models-add`

Промоутит run в **registry-бандл** `models/<friendly_name>/` и пишет `model_manifest.json`.

В бандл копируется (если есть на run):

- `models/` — веса и sidecar (та же раскладка, что в run).
- `train/` и все `train-*/` (например `train-ultralytics/`), без checkpoint в `weights/`.
- Legacy `test/` в корне run, плюс полное дерево `tests/`.
- `training_metadata.json` (пути к `.pt` приводятся к путям относительно бандла), `test_metrics*.csv` из корня run, `_runtime_data_*.yaml` из корня или `tmp/`.

`weights_file` в манифесте — путь относительно бандла (обычно `models/<stem>.pt` с detect_* или legacy stem имени run). Старые деревья с `<friendly_name>.pt` в корне бандла не меняются.

## Примеры

```bash
smartrain registry runs-list
smartrain registry runs-info 3
smartrain registry models-add 3
smartrain registry models-list
```
