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

- For development: `pip install -e ".[dev]"`
- For ClearML: `pip install -e ".[clearml]"`
- For SAHI: `pip install -e ".[sahi]"`
- For Docker / CUDA 12.8 image: see [docker.md](./docker.md)

## Checking

```bash
smartrain --help
smartrain train --help
```
