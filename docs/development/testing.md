> Russian version: [../ru/development/testing.md](../ru/development/testing.md)

# Testing

Project pytest defaults (`pyproject.toml`, `[tool.pytest.ini_options]`): `testpaths = ["tests"]`, `pythonpath = ["."]` — run commands from the repository root.

Optional local coverage (not enforced in CI):

```bash
pip install pytest-cov
pytest --cov=smartrain.cli --cov=smartrain.cli_entrypoints --cov-report=term-missing -q
```

## Launch

```bash
pip install -e ".[dev]"
pytest
```

## Recommended cuts

```bash
pytest tests/test_train_*.py tests/test_training_metadata*.py
pytest tests/test_results_analyzer.py
pytest tests/test_training_queue.py
pytest tests/test_inference_cli.py -q
pytest tests/regression/test_train_service_guardrails.py -q
pytest -k "replay" tests/test_cli_replay.py -q
```

Useful patterns:

```bash
pytest tests/test_cli_replay.py -q
pytest -k "analyze" tests/test_results_analyzer_workflows.py -q
```

Reference tests for CLI patterns: `tests/test_cli_replay.py`.

Canonical / integration smoke: `tests/integration/test_canonical_consumers.py`.

## What to check after documentation changes

- commands and flags in the examples correspond to the current `--help`;
- exit codes and backup paths are described correctly;
- links between sections are not broken.
