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

## Removed targets (canonical read legacy env flags)

- `SMARTTRAIN_CANONICAL_READ`
  - **Removed** from runtime (ignored if set). Unified/canonical read is always enabled.
- `SMARTTRAIN_ALLOW_LEGACY_READ_FALLBACK`
  - **Removed** from runtime (ignored if set). Legacy read fallback no longer exists.

Docs that still describe these flags as working emergency switches are obsolete.
