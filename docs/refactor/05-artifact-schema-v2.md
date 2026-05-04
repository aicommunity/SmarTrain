# Artifact Schema v2

## Required Top-Level Fields

- `schema_version`
- `task_type`
- `backend_type`
- `producer`
- `created_at`
- `artifacts`
- `metrics`
- `provenance`

## Artifact Sections

- `artifacts.model`
- `artifacts.test`
- `artifacts.inference`
- `artifacts.analyze` (when generated)

## Metrics Rules

- Metrics are namespaced by task type.
- Detection metrics keep compatibility aliases for transition period.
- Every metric payload stores source backend and calculation origin.

## Compatibility

- Reader supports legacy payloads via adapters.
- Writer defaults to v2.
