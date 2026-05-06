# Release Notes

## 0.0.3 (planned)

### Deprecations: canonical read legacy fallback environment flags

The following environment variables are deprecated and will be removed in `0.0.3`:

- `SMARTTRAIN_CANONICAL_READ`
  - Deprecated. Canonical read is always enabled now; this flag is ignored.
  - Removal target: `0.0.3`.
- `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK`
  - Deprecated. Legacy fallback is removed as part of the canonical cutover.
  - Removal target: `0.0.3`.

If both variables are set in a way that previously enabled legacy fallback, deterministic `[DEPRECATION] ...` warnings are emitted to `stderr`.

