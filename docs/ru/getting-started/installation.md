> English version: [../../getting-started/installation.md](../../getting-started/installation.md)

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

| Extra | Установка | Назначение |
|-------|-----------|------------|
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, mypy |
| `clearml` | `pip install -e ".[clearml]"` | интеграция ClearML |
| `sahi` | `pip install -e ".[sahi]"` | SAHI tiled inference |
| `export` | `pip install -e ".[export]"` или `smartrain deps install` | WeasyPrint PDF-движок (опционально; pandoc/ODT из базовой установки) |

Несколько extras: `pip install -e ".[dev,export]"`.

## Экспорт отчётов (PDF/ODT)

**Базовая** установка включает `pypandoc-binary` (bundled `pandoc`) и fallback `fpdf2`/`odfpy` — достаточно для ODT и базового PDF в `smartrain analyze all` и `smartrain dataset report`.

Опциональный PDF-движок **WeasyPrint** (часто лучше на Linux) — extra `export`:

```bash
smartrain deps install
smartrain deps doctor
```

Вручную:

```bash
pip install -e ".[export]"
```

Если `smartrain deps doctor` сообщает о проблемах WeasyPrint на Ubuntu/Debian:

```bash
sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

Системный `pandoc`: переменная `PANDOC=/full/path/to/pandoc` (см. [переменные окружения](../../reference/environment-variables.md)).

Базовые `fpdf2` и `odfpy` остаются fallback, если pandoc недоступен.

## Проверка

```bash
smartrain --help
smartrain train --help
smartrain deps doctor
```

См. также: [CLI: deps](../cli/deps.md).
