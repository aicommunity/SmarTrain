# CLI: deps

> English version: [../../cli/deps.md](../../cli/deps.md)

Вспомогательные команды для политики PyTorch и optional pip extras.

## Подкоманды

| Команда | Описание |
|---------|----------|
| `sync-torch` | Применить политику CUDA/torch wheels в текущем окружении |
| `doctor` | Проверить зависимости экспорта отчётов (pandoc, weasyprint, fallback) |
| `install` | Установить optional extras (по умолчанию: `export`) |

## Примеры

```bash
smartrain deps sync-torch
smartrain deps doctor
smartrain deps doctor --verbose
smartrain deps install
smartrain deps install --extra clearml
smartrain deps install --all-extras
smartrain deps install --dry-run
```

## Экспорт отчётов (extra `export`)

`analyze all` и `dataset report` по умолчанию создают Markdown и PNG. PDF/ODT требуют:

- `pypandoc-binary` (bundled pandoc)
- `weasyprint` (PDF-движок при наличии)

Установка:

```bash
smartrain deps install
```

Проверка:

```bash
smartrain deps doctor
```

Код выхода `0`, если pandoc доступен; `1`, если зависимости экспорта отсутствуют.

Эквивалент pip из исходников:

```bash
pip install -e ".[export]"
```

### Системные библиотеки WeasyPrint (Ubuntu/Debian)

Если PDF не собирается после `install`, `doctor` может подсказать:

```bash
sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

### Переопределение `PANDOC`

`PANDOC=/full/path/to/pandoc` — использовать системный pandoc вместо bundled.

## Известные optional extras

| Extra | Пакеты (кратко) | Назначение |
|-------|-----------------|------------|
| `export` | `pypandoc-binary`, `weasyprint` | PDF/ODT отчёты |
| `dev` | `pytest`, `ruff`, `mypy` | Разработка |
| `clearml` | `clearml` | Трекинг экспериментов |
| `sahi` | `sahi` | SAHI inference |

См. также: [Установка](../../getting-started/installation.md), [переменные окружения](../../reference/environment-variables.md).
