# Tech debt: layer boundaries

Operational waves: layer-boundary refactor (waves 0–8). Journal: [`09-tech-debt.md`](./09-tech-debt.md).  
Continuation (closed): [`layer-boundary-continuation.md`](./layer-boundary-continuation.md).

## Register

| id | status | area | deferred_decision | resolution |
|----|--------|------|-------------------|------------|
| TD-LB-000 | DONE | baseline | — | LOC baseline + guardrails (2026-05-15) |
| TD-LB-001 | DONE | naming | Two `TaskContext` types | `TaskTypeLabel`, `CanonicalIdentity` |
| TD-LB-002 | DONE | naming | `train_backend_registry` vs execution `TrainBackend` | `ultralytics_model_alias_registry` |
| TD-LB-003 | DONE | orchestrators | `canonical_gateway` → workflows | `adapters/canonical/read/metrics_csv` |
| TD-LB-004 | DONE | backends | Adapters proxy workflows | `implementations/ultralytics/inference.py` |
| TD-LB-005 | DONE | analyze | Split workflows/services | `services/analyze/` |
| TD-LB-006 | DONE | workflows | Fat CLI | Thin facades; execution in `services/` (2026-05-16) |
| TD-LB-007 | DONE | testing | test_backends → MTM profile | `core/runtime/system_profile` |
| TD-LB-008 | DONE | inference | Wrong run discovery import | `core.runtime.run_discovery` |
| TD-LB-009 | DONE | runtime | Legacy model path fallback | `resolve_run_model` (no rglob) |
| TD-LB-010 | DONE | LOC | Monolith modules | Analyze report split + `services/reporting/document_export.py` (2026-05-16) |
| TD-LB-011 | DONE | datasets | workflows/datasets monolith | Logic in `services/datasets/`; thin `workflows/datasets/*` facades (2026-05-16) |
| TD-LB-012 | DONE | LOC | MTM / model_test_cli | Parsers + resume in `services/training/`; test CLI in `services/testing/model_test_cli_service.py`; MTM ~1k LOC (2026-05-16) |
| TD-LB-013 | DONE | train/inference delta | LB-D1–D4 execution path | `train_wiring`, `train_cli_main`, `TrainRuntimeOps`, slim `inference_cli` (2026-05-16) |
| TD-LB-014 | DONE | canonical read | Silent `detection` default | `ModelAdapter` requires provenance; `migrate_model_task_provenance.py` (2026-05-16) |
| TD-LB-015 | DONE | testing LOC | `format_runners.py` ~2k LOC | Split: `format_runners_support.py`, `format_runners_ultralytics.py`, `format_runners_native.py`, thin facade (2026-05-16) |
| TD-LB-099 | DONE | closure | Program audit | Continuation closed 2026-05-16 |

## Burn-down checklist

- [x] TD-LB-000 … TD-LB-009
- [x] TD-LB-006
- [x] TD-LB-010
- [x] TD-LB-011
- [x] TD-LB-012
- [x] TD-LB-013
- [x] TD-LB-014
- [x] TD-LB-015
- [x] TD-LB-099

## Transitional guardrail allowlist

(empty — patch surface: `services/analyze/workflow_dispatch.py`)
