# Tech debt: layer boundaries

Operational waves: layer-boundary refactor plan (waves 0–8). Journal: [`09-tech-debt.md`](./09-tech-debt.md).

## Register

| id | status | area | deferred_decision | resolution |
|----|--------|------|-------------------|------------|
| TD-LB-000 | DONE | baseline | — | LOC 2049/2954/1431/3476; guardrails + allowlist for TD-LB-003/004 |
| TD-LB-001 | DONE | naming | Two `TaskContext` types | `TaskTypeLabel`, `CanonicalIdentity` |
| TD-LB-002 | DONE | naming | `train_backend_registry` vs execution `TrainBackend` | `ultralytics_model_alias_registry` |
| TD-LB-003 | OPEN | orchestrators | `canonical_gateway` imports `workflows.analyze` | Wave 3 → `adapters/canonical/read/metrics_csv` |
| TD-LB-004 | OPEN | backends | `ultralytics_adapter` proxies `workflows.inference` | Wave 5 → `backends/implementations/ultralytics` |
| TD-LB-005 | OPEN | analyze | Logic split `workflows/analyze` vs `services/analyze_*` | Wave 2 → `services/analyze/` |
| TD-LB-006 | OPEN | workflows | Fat CLI modules import `services` | Wave 2/4 thin CLI |
| TD-LB-007 | OPEN | testing | `model_test_backends` → `model_training_module` | Wave 4b → `core/runtime/system_profile` |
| TD-LB-008 | OPEN | inference | `inference_cli` imports `find_run_directories` via `results_analyzer` | Wave 2b → `run_discovery` |
| TD-LB-009 | OPEN | runtime | `resolve_run_model_with_legacy_fallback` | Wave 6 → `resolve_run_model` |
| TD-LB-010 | OPEN | LOC | MTM, test_backends, results_analyzer, analyze_report monoliths | Waves 2/4 split |

## Burn-down checklist

- [x] TD-LB-000
- [x] TD-LB-001
- [x] TD-LB-002
- [ ] TD-LB-003
- [ ] TD-LB-004
- [ ] TD-LB-005
- [ ] TD-LB-006
- [ ] TD-LB-007
- [ ] TD-LB-008
- [ ] TD-LB-009
- [ ] TD-LB-010
