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
- run/model contract: `smartrain/run_model_contract/` (`domain/`, `io/`, `gateway.py`);
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

## Layer boundaries (post wave 8, 2026-05-16)

Layer-boundary refactor (waves 0–8, continuation LB-C1–C8) is **closed**. Conformance matrix and evidence: [`layer-boundary-continuation.md`](./layer-boundary-continuation.md). Register: [`tech-debt-layer-boundaries.md`](./tech-debt-layer-boundaries.md).

Train CLI: [`train_entry.py`](../../smartrain/workflows/training/train_entry.py) + [`train_wiring.py`](../../smartrain/workflows/training/train_wiring.py); orchestration and interactive callbacks in [`services/training/`](../../smartrain/services/training/) (`train_cli_main.py`, `train_cli_callbacks.py`). `model_training_module.py` removed (LB-D8 / TD-LB-016, 2026-05-16). `workflows/testing/model_test_cli.py` remains a thin facade — execution in `services/testing/`.
