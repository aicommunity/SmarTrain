# Release Notes

## 0.0.3 (planned)

### Unified layer package rename

- Bounded context `canonical` renamed to **unified**: `smartrain/unified/` (`domain/`, `io/`), `orchestrators/unified_gateway.py`.
- On-disk snapshots: `.smartrain/unified/` (read fallback for legacy `.smartrain/canonical/`).
- Environment: `SMARTTRAIN_UNIFIED_WRITE`, `SMARTTRAIN_UNIFIED_DUAL_WRITE_MODE` (legacy `SMARTTRAIN_CANONICAL_*` accepted with deprecation warning).
- CLI: `smartrain migrate unified` (hidden alias: `migrate canonical`).
- Run layout helpers renamed: `preferred_run_model_path`, `materialize_preferred_run_model`, `normalize_ultralytics_run_layout`.

### Deprecations: canonical read legacy fallback environment flags

The following environment variables are deprecated and will be removed in `0.0.3`:

- `SMARTTRAIN_CANONICAL_READ`
  - Deprecated. Canonical read is always enabled now; this flag is ignored.
  - Removal target: `0.0.3`.
- `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK`
  - Deprecated. Legacy fallback is removed as part of the canonical cutover.
  - Removal target: `0.0.3`.

If both variables are set in a way that previously enabled legacy fallback, deterministic `[DEPRECATION] ...` warnings are emitted to `stderr`.

