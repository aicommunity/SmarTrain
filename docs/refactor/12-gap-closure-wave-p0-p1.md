# Gap Closure Wave (P0/P1)

Date: 2026-05-08
Baseline: `docs/refactor/11-plan-conformance-audit.md`

## Goal

Close the highest-impact conformance gaps against the original refactor target-state:
- strict service encapsulation boundaries;
- runtime task/backend neutrality;
- analyze gateway-first completion;
- reduced cognitive complexity in remaining large orchestrators.

## Delivery policy

- Keep behavior stable unless a policy requires otherwise.
- No destructive migration steps in this wave.
- Each step must include targeted tests and full regression (`pytest -q`).
- Update `docs/refactor/09-tech-debt.md` after each step with residual debt status.

## Step 1 (P0): Service boundary hardening

### Scope
- Remove direct `services -> workflows` imports where services consume workflow-private helpers.
- Introduce explicit public helper APIs in service-safe modules and route consumers through them.

### Candidate files
- `smartrain/services/train_service.py`
- `smartrain/services/inference_service.py`
- `smartrain/services/test_backend_dispatch.py`
- `smartrain/services/testing/model_test_runner.py`

### Definition of done
- No service module imports workflow-private symbols (`_name` helpers).
- Service modules depend only on contracts/public helpers.
- Existing CLI behavior is unchanged in regression tests.

## Step 2 (P0): Runtime neutrality cleanup (task/backend)

### Scope
- Remove hardcoded `detection/ultralytics` defaults in canonical model-source mapping where provenance allows stronger inference.
- Ensure runtime backend execution path follows capability resolution, not fixed adapter assumptions.

### Candidate files
- `smartrain/adapters/canonical/read/model_adapter.py`
- `smartrain/workflows/inference/inference_backends.py`
- `smartrain/services/inference_service.py`
- `smartrain/backends/train_test_registry.py` (if routing contracts need tightening)

### Definition of done
- No unconditional `detection`/`ultralytics` fallback in canonical model mapping for supported metadata cases.
- Capability result is reflected in actual runtime backend selection.
- Detection scenarios remain stable; cls/seg coverage is not regressed.

## Step 3 (P1): Analyze gateway-first completion

### Scope
- Collapse remaining ad-hoc read branches in analyze services to canonical gateway contracts where policy does not allow legacy bypass.
- Move format-compare metric selection to task-aware contract points.

### Candidate files
- `smartrain/services/analyze_format_compare_service.py`
- `smartrain/workflows/analyze/results_analyzer.py`
- `smartrain/run_model_contract/gateway.py`
- `smartrain/workflows/analyze/metrics_reader.py` (fallback policy hooks only)

### Definition of done
- Canonical-enabled analyze paths read via gateway contracts.
- Legacy fallback paths are explicit, policy-gated, and documented.
- No hidden bypass for canonical-enabled mode.

## Step 4 (P1): Orchestrator thinning (targeted slices) — **closed (2026-05-16)**

### Outcome
- `results_analyzer.py` — thin CLI facade; analyze logic in `services/analyze/`.
- Train: `model_training_module.py` **removed** (LB-D8 / TD-LB-016); entry `train_entry` + `train_wiring`; CLI and callbacks in `services/training/` (`train_cli_main.py`, `train_cli_callbacks.py`).

### Historical scope (2026-05-08 plan)
- Reduce cognitive complexity in `results_analyzer` and train orchestration by extracting mixed-responsibility blocks into `services/*`.

### Definition of done (met)
- Large blocks behind named service boundaries; workflow entry modules are facades/wiring only.
- No behavior drift in interactive/non-interactive workflows (regression suite green).

## Test protocol (minimum per step)

- Targeted:
  - `pytest -q tests/test_imports.py`
  - `pytest -q tests/test_inference_cli.py`
  - `pytest -q tests/test_results_analyzer_workflows.py`
  - `pytest -q tests/regression/test_canonical_cutover.py tests/regression/test_no_legacy_branch_usage.py`
- Full:
  - `pytest -q`

## Commit plan (small semantic commits)

1. `Harden service boundaries by removing workflow-private dependencies.`
2. `Align runtime backend execution with capability-driven routing.`
3. `Finish gateway-first analyze reads and policy-gated legacy fallback.`
4. `Extract remaining orchestration blocks into focused workflow services.`

## Completion criteria for this wave

- P0 items are closed with no critical regressions.
- P1 items are closed or explicitly moved to `Operational Limits` with rationale.
- `docs/refactor/09-tech-debt.md` reflects post-wave residual debt state.
