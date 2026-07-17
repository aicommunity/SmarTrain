> Russian version: [../ru/getting-started/installation.md](../ru/getting-started/installation.md)

# Installation

The document describes the minimal steps to get started with `smartrain` from source.

## Requirements

- Python `3.10+`

## Installation from source

```bash
cd /path/to/smart-train
pip install -e .
```

The `smartrain` command is registered via `pyproject.toml`.

## Additional dependencies

| Extra | Install | Purpose |
|-------|---------|---------|
| `dev` | `pip install -e ".[dev]"` | pytest, ruff, mypy |
| `clearml` | `pip install -e ".[clearml]"` | ClearML experiment tracking |
| `sahi` | `pip install -e ".[sahi]"` | SAHI tiled inference |
| `export` | `pip install -e ".[export]"` or `smartrain deps install` | PDF/ODT report export (`pypandoc-binary`, `weasyprint`) |

Or install multiple extras: `pip install -e ".[dev,export]"`.

## Report export (PDF/ODT)

The **base** install is enough for training, analyze Markdown reports, and dataset sample PNG reports.

PDF and ODT export for `smartrain analyze all` and `smartrain dataset report` require the optional **`export`** extra (`pypandoc-binary` ships a bundled `pandoc` binary; `weasyprint` is used as a PDF engine when available).

```bash
smartrain deps install
smartrain deps doctor
```

Manual equivalent:

```bash
pip install -e ".[export]"
```

If `smartrain deps doctor` reports WeasyPrint issues on Ubuntu/Debian, install system libraries:

```bash
sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

To use a system `pandoc` instead of the bundled one, set `PANDOC=/full/path/to/pandoc` (see [environment variables](../reference/environment-variables.md)).

Base dependencies `fpdf2` and `odfpy` remain available as fallbacks when pandoc export is skipped or fails.

## Checking

```bash
smartrain --help
smartrain train --help
smartrain deps doctor
```

See also: [CLI: deps](../cli/deps.md).
