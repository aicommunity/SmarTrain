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
| TD-LB-010 | PARTIAL | LOC | Monolith modules | `report_writer` remains (~3.4k LOC); LB-C6 deferred |
| TD-LB-099 | DONE | closure | Program audit | Continuation closed 2026-05-16 |

## Burn-down checklist

- [x] TD-LB-000 … TD-LB-009
- [x] TD-LB-006
- [x] TD-LB-010 (PARTIAL — `report_writer` only)
- [x] TD-LB-099

## Transitional guardrail allowlist

- `services/analyze/report_writer.py` → lazy `workflows.datasets` export
- `services/analyze/cli_commands.py` → `workflows.analyze` facade dispatch
