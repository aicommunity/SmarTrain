# CLI Command Inventory

## Core Entry

- `smartrain` (`smartrain/cli.py`)

## High-Priority Commands

- `smartrain train` -> `smartrain/model_training_module.py`
- `smartrain test` -> `smartrain/workflows/testing/model_test_cli.py`
- `smartrain inference` -> `smartrain/inference_cli.py`
- `smartrain analyze` -> `smartrain/results_analyzer.py`
- `smartrain balance` -> `smartrain/dataset_balance.py`
- `smartrain augment` -> `smartrain/dataset_augment.py`

## Analyze Subcommands (argparse)

- `all`, `scan`, `export-table`, `compare`, `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`, `leaderboard`

## Interactive Paths

- Interactive behavior depends on `smartrain/interactive_contract.py`.
- Most commands have `-y/--non-interactive` or equivalent.
- Replay generation is handled by `smartrain/cli_replay.py`.

## Standardization Target

- Replace command-specific prompt patterns with shared prompt core.
- Use shared request/response envelope for every command flow.
