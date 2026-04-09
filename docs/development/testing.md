> Russian version: [../ru/development/testing.md](../ru/development/testing.md)

# Testing

## Launch

```bash
pip install -e ".[dev]"
pytest
```

## Recommended cuts

```bash
pytest tests/test_model_training_module.py
pytest tests/test_results_analyzer.py
pytest tests/test_training_queue.py
```

## What to check after documentation changes

- commands and flags in the examples correspond to the current `--help`;
- exit codes and backup paths are described correctly;
- links between sections are not broken.
