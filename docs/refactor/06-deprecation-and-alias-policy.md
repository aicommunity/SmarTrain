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
