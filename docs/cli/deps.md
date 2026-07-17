# CLI: deps

> Russian version: [../ru/cli/deps.md](../ru/cli/deps.md)

Dependency management helpers for PyTorch policy and optional pip extras.

## Subcommands

| Command | Description |
|---------|-------------|
| `sync-torch` | Apply default CUDA/torch wheel policy in the current environment |
| `doctor` | Check report-export dependencies (pandoc, weasyprint, fallbacks) |
| `install` | Install optional pip extras (default: `export`) |

## Examples

```bash
smartrain deps sync-torch
smartrain deps doctor
smartrain deps doctor --verbose
smartrain deps install
smartrain deps install --extra clearml
smartrain deps install --all-extras
smartrain deps install --dry-run
```

## Report export (`export` extra)

`analyze all` and `dataset report` build Markdown and PNG by default. PDF/ODT need:

- `pypandoc-binary` (bundled pandoc)
- `weasyprint` (PDF engine when available)

Install:

```bash
smartrain deps install
```

Verify:

```bash
smartrain deps doctor
```

Exit code `0` when pandoc is available; `1` when export dependencies are missing.

Manual pip equivalent from source:

```bash
pip install -e ".[export]"
```

### Ubuntu/Debian WeasyPrint libraries

If PDF export fails after `install`, `doctor` may suggest:

```bash
sudo apt-get install -y libcairo2 libpango-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

### `PANDOC` override

Set `PANDOC=/full/path/to/pandoc` to use a system binary instead of bundled pandoc.

## Known optional extras

| Extra | Packages (high level) | Purpose |
|-------|----------------------|---------|
| `export` | `pypandoc-binary`, `weasyprint` | PDF/ODT reports |
| `dev` | `pytest`, `ruff`, `mypy` | Development |
| `clearml` | `clearml` | Experiment tracking |
| `sahi` | `sahi` | SAHI inference |

See also: [Installation](../getting-started/installation.md), [environment variables](../reference/environment-variables.md).
