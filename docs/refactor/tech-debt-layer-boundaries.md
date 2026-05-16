# Tech debt: layer boundaries

Operational waves: layer-boundary refactor (waves 0–8). Journal: [`09-tech-debt.md`](./09-tech-debt.md).  
Continuation: [`layer-boundary-continuation.md`](./layer-boundary-continuation.md).

## Register

| id | status | area | deferred_decision | resolution |
|----|--------|------|-------------------|------------|
| TD-LB-000 | DONE | baseline | — | LOC baseline + guardrails (2026-05-15) |
| TD-LB-001 | DONE | naming | Two `TaskContext` types | `TaskTypeLabel`, `CanonicalIdentity` |
| TD-LB-002 | DONE | naming | `train_backend_registry` vs execution `TrainBackend` | `ultralytics_model_alias_registry` |
| TD-LB-003 | DONE | orchestrators | `canonical_gateway` → workflows | `adapters/canonical/read/metrics_csv` |
| TD-LB-004 | PARTIAL | backends | Adapters proxy workflows | `BackendRegistry` alias; LB-C1 in continuation |
| TD-LB-005 | DONE | analyze | Split workflows/services | `services/analyze/` |
| TD-LB-006 | PARTIAL | workflows | Fat CLI | Thin `results_analyzer`; MTM/test_backends remain (LB-C3/C4) |
| TD-LB-007 | DONE | testing | test_backends → MTM profile | `core/runtime/system_profile` |
| TD-LB-008 | DONE | inference | Wrong run discovery import | `core.runtime.run_discovery` |
| TD-LB-009 | DONE | runtime | Legacy model path fallback | `resolve_run_model` (no rglob) |
| TD-LB-010 | PARTIAL | LOC | Monolith modules | Analyze moved; report_writer/MTM/test_backends open (LB-C6) |
| TD-LB-099 | DONE | closure | Program audit | `layer-boundary-continuation.md` |

## Burn-down checklist

- [x] TD-LB-000
- [x] TD-LB-001
- [x] TD-LB-002
- [x] TD-LB-003
- [x] TD-LB-004 (PARTIAL — see continuation LB-C1)
- [x] TD-LB-005
- [x] TD-LB-006 (PARTIAL — see LB-C3/C4)
- [x] TD-LB-007
- [x] TD-LB-008
- [x] TD-LB-009
- [x] TD-LB-010 (PARTIAL — see LB-C6)
- [x] TD-LB-099

## Transitional guardrail allowlist (remove with continuation items)

- `backends/ultralytics_adapter.py`, `external_provider_adapter.py` → LB-C1
- `services/analyze/metrics_reader.py`, `report_writer.py`, `ultralytics_test_artifacts.py` → LB-C2
- `services/analyze/cli_commands.py` → `workflows.analyze` (facade) → LB-C5
