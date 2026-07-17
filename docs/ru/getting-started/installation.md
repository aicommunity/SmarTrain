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
| `export` | `pip install -e ".[export]"` или `smartrain deps install` | PDF/ODT отчёты (`pypandoc-binary`, `weasyprint`) |

Несколько extras: `pip install -e ".[dev,export]"`.

## Экспорт отчётов (PDF/ODT)

**Базовой** установки достаточно для обучения, Markdown-отчётов `analyze` и PNG-примеров `dataset report`.

PDF и ODT для `smartrain analyze all` и `smartrain dataset report` требуют optional extra **`export`** (`pypandoc-binary` включает bundled `pandoc`; `weasyprint` — PDF-движок при наличии).

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
