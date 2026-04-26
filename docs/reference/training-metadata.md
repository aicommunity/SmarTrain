> Russian version: [../ru/reference/training-metadata.md](../ru/reference/training-metadata.md)

# `training_metadata.json` format

The file is saved in the run directory next to `test_metrics.csv`.

## Key sections

- `training_info` — model, dataset, hyperparameters.
- `timestamps` — training and testing intervals.
- `status` — success or error with the trace.
- `paths` - relative paths to artifacts.
- `inference` — `val()` parameters, if available.
- `system_profile` — machine profile captured at run save time.

## `system_profile` fields

- `cpu` — model, architecture, logical/physical cores.
- `ram` — total RAM (`total_bytes`, `total_gb`).
- `gpu` — CUDA availability, list of devices, per-GPU VRAM and total VRAM.
- `disk` — filesystem and capacity for the mount where run artifacts are stored.
- `platform` — OS, kernel, Python version, hostname.
- `capture_warnings` — probing warnings when some fields were unavailable.

## Purpose

- run reproducibility;
- transparent audit of parameters and errors;
- input for `analyze` and `registry`.
