# Plan Conformance Audit Baseline

Date: 2026-05-08
Source plan: `/home/user/.cursor/plans/рефакторинг_cli_и_ml_ядра_798a9741.plan.md`

## Audit baseline decision

The current repository state is accepted as a baseline for further gap-closure work against the refactor target-state.

This baseline confirms:
- major architecture directions are in place (canonical layer, backend/task contracts, migration/cutover path, guardrails);
- full target-state is not yet reached in encapsulation, duplication removal, runtime neutrality, and analyze unification;
- next work should prioritize P0/P1 deltas that block safe extensibility for new tasks and backends.

## Baseline findings snapshot

### Implemented foundations
- canonical domain and adapters: `smartrain/domain/canonical/*`, `smartrain/adapters/canonical/*`, `smartrain/orchestrators/canonical_gateway.py`;
- task-aware contracts and adapters: `smartrain/tasks/*`;
- backend contracts and capability registry: `smartrain/backends/contracts.py`, `smartrain/backends/registry.py`, `smartrain/backends/train_test_registry.py`;
- service-level decomposition progress in training and analyze workflows.

### Residual high-priority gaps
- service boundary violations (`services` importing `workflows` and private helpers);
- remaining detection/ultralytics-centric defaults in runtime-critical paths;
- partial gateway-first completion in analyze read paths;
- large facade-orchestrator modules with high cognitive load.

## Scope for follow-up wave planning

The next implementation wave should target only actionable P0/P1 gaps and preserve existing behavior through compatibility wrappers and regression-first delivery.
