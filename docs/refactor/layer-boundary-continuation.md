# Layer boundary refactor — continuation (closed)

**Audit date:** 2026-05-15  
**Closed:** 2026-05-16  
**Register:** [tech-debt-layer-boundaries.md](./tech-debt-layer-boundaries.md)

## Conformance summary

| # | Goal | Status | Evidence |
|---|------|--------|----------|
| G1 | Thin `workflows/` CLI | Met | `results_analyzer.py`, `model_test_backends.py` (~40 LOC facades); argparse in services where noted |
| G2 | Analyze in `services/analyze/` | Met | 30+ modules under `services/analyze/` |
| G3 | Unambiguous names | Met | `TaskTypeLabel`, `UnifiedIdentity`, `ultralytics_model_alias_registry` |
| G4 | No new path fallbacks | Met | `resolve_run_model` (no rglob) |
| G5 | Class-based extension | Met | `AnalyzeCommandRegistry`, `CapabilityRegistry` / `BackendRegistry` |
| G6 | `services` ⊄ `workflows` | Met | `tests/regression/test_layer_import_guardrails.py` + minimal allowlist |
| G7 | `run_model_contract` ⊄ `workflows` | Met | `run_model_contract.gateway` → `run_model_contract.io.read.metrics_csv` (via `core.analyze.run_metrics_discovery`) |
| G8 | `backends` ⊄ `workflows` | Met | `backends/implementations/ultralytics/inference.py` |
| G9 | Train/test not monoliths in workflows | Met | `train_yolo_execution_service`, `services/testing/backends/{native_eval,format_runners}`; thin workflow facades |
| G10 | Run/model contract metrics read | Met | `run_model_contract/io/read/metrics_csv.py` |

**Score:** 10 Met, 0 Partial for mandatory continuation items.

## Completed items

| id | Result |
|----|--------|
| LB-C1 | `backends/implementations/ultralytics/inference.py` |
| LB-C2 | `core/testing/*`; shared export in `services/reporting/document_export.py` |
| LB-C3 | `train_yolo_execution_service` + metadata IO in `train_metadata_io_service` |
| LB-C4 | `services/testing/backends/native_eval.py` + `format_runners.py`; `model_test_backends.py` facade |
| LB-C5 | `prompts.py` + `_workflow_attr` facade dispatch |
| LB-C6 | `report_markdown.py`, `report_odt.py`, thin `report_writer.py` |
| LB-C8 | CI: `pytest tests/regression/test_layer_import_guardrails.py` in phase8 workflow |
| LB-C7 | `services/datasets/` (full logic); `workflows/datasets/` thin facades + `datasets_entry` |

## Deferred (non-blocking)

(none for layer-boundary continuation)

## Key paths after closure

| Area | Location |
|------|----------|
| Train execution | `smartrain/services/training/train_yolo_execution_service.py` |
| Test backends | `smartrain/services/testing/backends/format_runners.py`, `native_eval.py` |
| Test facade | `smartrain/workflows/testing/model_test_backends.py` |
| Report export | `smartrain/services/reporting/document_export.py` |
| Datasets | `smartrain/services/datasets/` + thin `workflows/datasets/*` facades |
| Analyze report | `smartrain/services/analyze/report_{markdown,odt,writer}.py` |
| TensorRT utils | `smartrain/core/models/tensorrt_checks.py` |
| Validator core | `smartrain/services/testing/unified_validator_core.py` |

## Links

- [03-target-architecture.md](./03-target-architecture.md)
- [package-layout.md](../development/package-layout.md)
