> English version: [../../development/extension-guide.md](../../development/extension-guide.md)

# Расширение проекта

## Куда класть код

- **CLI / argparse / связка с Typer** → `smartrain/workflows/<область>/` или тонкие обёртки в `smartrain/cli_entrypoints/` (как `train_app.py`). Каталог датасетов и интерактивный выбор: `workflows/datasets/dataset_cli_catalog.py` / `dataset_cli_common.py`.
- **Переиспользуемая логика без argparse** → `smartrain/services/` (**без** импортов `workflows`; только через `smartrain/core/workflow_adapters/`).
- **Контракты workspace и run/model contract** → `smartrain/core/runtime/`, `smartrain/run_model_contract/`.

Карта каталогов: [package-layout.md](../../development/package-layout.md).

## Добавление новой CLI-команды

1. Добавить роутинг в `smartrain/cli.py`.
2. Реализовать argparse-модуль с `build_*_arg_parser()` и `main(argv)`.
3. Обновить разделы:
   - `docs/cli/overview.md`
   - профильный файл в `docs/cli/`
   - при необходимости `docs/development/architecture.md`.

### Как Typer передаёт управление в argparse

Используется `_forward_argparse_command` в `smartrain/cli.py` (найти определение по имени). Основные параметры:

| Параметр | Назначение |
|----------|------------|
| `module` | Путь импорта модуля с `main(argv)`, например `smartrain.workflows.datasets.datasets_entry`. |
| `build_parser` | Опционально: фабрика `ArgumentParser`; вместе с `prog` подмешивает примеры из `ARGPARSE_HELP_EXAMPLES`. |
| `prog` | Ключ для epilog с примерами (должен совпадать со строкой, переданной в `_forward_argparse_command`). |
| `prepend_args` | Токены перед `ctx.args` (например имя подкоманды, если модуль ожидает его в argv). |
| `empty_args_mode` | `"help"` \| `"invoke"` \| `"invoke_if_tty_else_help"` — поведение при пустых аргументах. |

Пример (иллюстрация):

```python
@app.command("myfeature")
def cmd_myfeature(ctx: typer.Context) -> None:
    from smartrain.workflows.myfeature import myfeature_cli

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.myfeature.myfeature_cli",
        build_parser=myfeature_cli.build_arg_parser,
        prog="smartrain myfeature",
        prepend_args=[],
        empty_args_mode="invoke_if_tty_else_help",
    )
```

Целевой модуль:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(...)
    ...
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ...
```

### Чеклист PR для новой команды

- [ ] Зарегистрировано в `cli.py`.
- [ ] Обновлены `docs/cli/overview.md` и профильный `docs/cli/*.md`.
- [ ] Примеры в документации совпадают с `smartrain <cmd> --help`.
- [ ] Добавлен тест в `tests/` (ориентир — паттерн из `tests/test_cli_replay.py` или тест существующей команды).

## Изменение контрактов данных

При изменениях `datasets_info.json`, `training_metadata.json`, `model_manifest.json` обязательно обновлять:

- `docs/reference/data-formats.md`
- `docs/reference/training-metadata.md`
- диаграмму контрактов в `docs/development/architecture.md`.

## Добавление или изменение внешних провайдеров

Используйте отдельный инженерный гайд:

- `docs/ru/development/provider-development.md`

При изменениях провайдерной подсистемы также обновляйте:

- `docs/ru/cli/providers.md`
- `docs/ru/providers/overview.md`
- профильные страницы в `docs/ru/providers/`
