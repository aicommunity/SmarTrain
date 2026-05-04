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
