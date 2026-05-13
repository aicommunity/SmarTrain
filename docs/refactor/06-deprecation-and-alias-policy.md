# Deprecation and Alias Policy

## Rules

- Prefer additive alias before removing old flags/fields.
- Emit deterministic warning message with replacement guidance.
- Keep deprecation window for at least one release cycle.

## Warning Format

- `[DEPRECATION] <old> is deprecated; use <new>. Removal target: <version/date>.`

## Removal Gate

- Removal is allowed only after:
  - migration tooling exists,
  - usage checks are green,
  - release notes are published.

## Current deprecation targets (canonical read legacy env flags)

- `SMARTTRAIN_CANONICAL_READ`
  - Deprecated: canonical read legacy toggle is removed as part of the Wave 6–7 cutover.
  - Removal target: `0.0.3`.
- `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK`
  - Deprecated: legacy fallback behavior is removed as part of the Wave 6–7 cutover.
  - Removal target: `0.0.3`.
