# Run/Model Unified Schema

## Purpose

Unify data access for entities that currently come from either `runs` or `models` layouts.

## Unified Entities

- `UnifiedArtifactRef`: common identity (`id`, `source_kind`, `created_at`, `task_type`, `backend_type`)
- `UnifiedModelRef`: model payload (`model_id`, `format`, `weights_path`, `config_path`, `provenance`)
- `UnifiedRunRef`: run payload (`run_id`, `workspace`, `dataset_ref`, `tests`, `inferences`)
- `UnifiedMetricsRef`: (`namespace`, `primary_metrics`, `secondary_metrics`, `producer`)
- `UnifiedPredictionRef`: (`task_type`, `schema_version`, `items_path`, `count`)

## Required Invariants

- `task_type` consistency across linked refs.
- Non-empty unified path fields after normalization.
- `schema_version` must be present and valid.

## Read/Write Contract

- Read adapters map legacy source to unified payload.
- Write adapters persist unified payload and manifest.

### Unified Gateway API (PR 6.5)

- `load_target(ref, source_kind?, options?) -> UnifiedPayload`
- `resolve_task_context(ref, source_kind?, options?) -> TaskContext`
- `load_metrics(ref, source_kind?, format_name?, options?) -> list[UnifiedMetricsRef]`
- `load_predictions(ref, source_kind?, format_name?, split?, options?) -> list[UnifiedPredictionRef]`

Notes:

- `load_predictions` uses file discovery under the run/model root; **strict mode** (`UnifiedGatewayOptions(predictions_strict=True)`) limits templates to normative paths only (no `*pred*` globs). Default remains backward-compatible. See also [`05-artifact-schema-v2.md`](./05-artifact-schema-v2.md) (prediction bundle paths).
- `load_metrics` namespaces follow `{task_type}/test_{format}` and must pass unified validators.

### Opt-in unified snapshot write (G1 hooks)

When `SMARTTRAIN_CANONICAL_WRITE=1`, a shared helper [`maybe_dual_write_unified_snapshot`](../../smartrain/adapters/unified/write/snapshot_hook.py) may run after successful steps:

| Pipeline | Trigger |
|----------|---------|
| Model test | After `persist_target_test_artifacts_state(..., status="ok")` (existing); dual-write mode: `SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE`. |
| Train (builtin) | After successful train + test and `save_training_metadata`. |
| Train (test-only) | After successful test-only run and metadata save. |
| Train (external provider) | When external train returns `rc==0` and test phase succeeded. |
| Inference (local / external) | After `inference_results.json` is written for the job. |

Legacy writer hooks for model test remain unchanged (`dual_write_*` modes with non–`unified_only`).
