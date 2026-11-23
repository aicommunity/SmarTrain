# Release Notes

## 0.0.6

### Removed: canonical read legacy fallback environment flags

The following environment variables are **removed** from runtime. Setting them has no effect and does not emit warnings:

- `SMARTTRAIN_CANONICAL_READ` — ignored; unified/canonical read is always on.
- `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK` — ignored; legacy read fallback is gone.

Use `smartrain update` / migration tooling for on-disk legacy layouts. See [`docs/refactor/legacy-compat-inventory.md`](refactor/legacy-compat-inventory.md).

## 0.0.3

### Run/model contract package (`run_model_contract`)

- Python import path: `smartrain/run_model_contract/` (`domain/`, `io/`, `gateway.py`). Replaces `smartrain/unified/` and `smartrain.orchestrators.unified_gateway`.
- On-disk snapshots unchanged: `.smartrain/unified/` (read fallback for legacy `.smartrain/canonical/`).
- Environment: `SMARTTRAIN_UNIFIED_WRITE`, `SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE` (legacy `SMARTTRAIN_CANONICAL_*` write aliases accepted with deprecation warning).
- CLI: `smartrain migrate unified` (hidden alias: `migrate canonical`).
- Run layout helpers renamed: `preferred_run_model_path`, `materialize_preferred_run_model`, `normalize_ultralytics_run_layout`.
