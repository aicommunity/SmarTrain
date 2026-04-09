# Установка

Документ описывает минимальные шаги, чтобы начать работу с `smartrain` из исходников.

## Требования

- Python `3.10+`

## Установка из исходников

```bash
cd /path/to/smart-train
pip install -e .
```

Команда `smartrain` регистрируется через `pyproject.toml`.

## Дополнительные зависимости

- Для разработки: `pip install -e ".[dev]"`
- Для ClearML: `pip install -e ".[clearml]"`
- Для SAHI: `pip install -e ".[sahi]"`

## Проверка

```bash
smartrain --help
smartrain train --help
```
