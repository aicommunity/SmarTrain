> Russian version: [../ru/reference/training-metadata.md](../ru/reference/training-metadata.md)

# `training_metadata.json` format

The file is saved in the run directory next to `test_metrics.csv`.

## Key sections

- `training_info` — model, dataset, hyperparameters.
- `timestamps` — training and testing intervals.
- `status` — success or error with the trace.
- `paths` - relative paths to artifacts.
- `inference` — `val()` parameters, if available.

## Purpose

- run reproducibility;
- transparent audit of parameters and errors;
- input for `analyze` and `registry`.
