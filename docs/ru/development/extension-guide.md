> English version: [../../development/extension-guide.md](../../development/extension-guide.md)

# Расширение проекта

## Добавление новой CLI-команды

1. Добавить роутинг в `smartrain/cli.py`.
2. Реализовать argparse-модуль с `build_*_arg_parser()` и `main(argv)`.
3. Обновить разделы:
   - `docs/cli/overview.md`
   - профильный файл в `docs/cli/`
   - при необходимости `docs/development/architecture.md`.

## Изменение контрактов данных

При изменениях `datasets_info.json`, `training_metadata.json`, `model_manifest.json` обязательно обновлять:

- `docs/reference/data-formats.md`
- `docs/reference/training-metadata.md`
- диаграмму контрактов в `docs/development/architecture.md`.
