# Run/Model Canonical Schema

## Purpose

Unify data access for entities that currently come from either `runs` or `models` layouts.

## Canonical Entities

- `CanonicalArtifactRef`: common identity (`id`, `source_kind`, `created_at`, `task_type`, `backend_type`)
- `CanonicalModelRef`: model payload (`model_id`, `format`, `weights_path`, `config_path`, `provenance`)
- `CanonicalRunRef`: run payload (`run_id`, `workspace`, `dataset_ref`, `tests`, `inferences`)
- `CanonicalMetricsRef`: (`namespace`, `primary_metrics`, `secondary_metrics`, `producer`)
- `CanonicalPredictionRef`: (`task_type`, `schema_version`, `items_path`, `count`)

## Required Invariants

- `task_type` consistency across linked refs.
- Non-empty canonical path fields after normalization.
- `schema_version` must be present and valid.

## Read/Write Contract

- Read adapters map legacy source to canonical payload.
- Write adapters persist canonical payload and manifest.

### Canonical Gateway API (PR 6.5)

- `load_target(ref, source_kind?, options?) -> CanonicalPayload`
- `resolve_task_context(ref, source_kind?, options?) -> TaskContext`
- `load_metrics(ref, source_kind?, format_name?, options?) -> list[CanonicalMetricsRef]`
- `load_predictions(ref, source_kind?, format_name?, split?, options?) -> list[CanonicalPredictionRef]`

Notes:

- `load_predictions` currently uses conservative file discovery (`debug_*` jsonl and `*pred*.json[l]`) until a strict prediction bundle is formalized.
- `load_metrics` namespaces follow `{task_type}/test_{format}` and must pass canonical validators.
