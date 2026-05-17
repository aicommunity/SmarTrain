> Russian link: [../ru/development/package-layout.md](../ru/development/package-layout.md)

# SmarTrain package layout

One-line guide to `smartrain/` so you can navigate by **function** without reading the whole refactor archive. Details and wave status: [refactor/13-project-current-state.md](../refactor/13-project-current-state.md).

## Entry points

- `cli.py` — Typer router; forwards to argparse modules or `cli_entrypoints/*`.
- `__main__.py`, `__init__.py` — package entry.

## workflows/

Thin CLI facades: argparse `main()`, Typer glue, re-exports into `services/`. Business logic lives in `services/`, not here.

Subpackages (each maps to commands or command groups):

- `training/` — train CLI (`train_entry`, `train_wiring` for resume/calc-confidence; execution in `services/training/`).
- `datasets/` — thin facades (`dataset_former.py`, `datasets_json_former.py`, …) → `services/datasets/`.
- `testing/` — `model_test_cli.py`, `model_test_backends.py` (facade) → `services/testing/backends/`.
- `inference/` — inference CLI, SAHI/heatmap helpers; runtime in `services/inference_service.py`.
- `analyze/` — `results_analyzer.py` (facade) → `services/analyze/cli_commands.py`.
- `queue/` — training queue (`training_queue.py`).
- `registry/` — registry CLI.
- `migration/` — migration CLI.
- `models/` — convert / release CLI.

**Rule:** new CLI surface area lives here first (`build_*_arg_parser`, `main(argv)`).

## services/

Use-case layer: `analyze/`, `datasets/`, `training/`, `testing/`, `inference_service.py`, `reporting/`. **Must not** import `smartrain.workflows.*` (use `core/workflow_adapters/` where a workflow entry is still required).

## core/

Shared mechanics: `runtime/` (workspace paths, env), `training/` (profiles, catalogs), `workflow_adapters/` (thin facades to workflows for services), `inference/` (shared inference helpers).

## Run/model contract (`smartrain/run_model_contract/`)

Read legacy run & model layouts, validate schema v2, expose payload API via `gateway`, optional snapshots under `.smartrain/unified/`.

| Path | Role |
|------|------|
| `run_model_contract/gateway.py` | `load_target`, `load_metrics`, `resolve_task_context`, predictions API |
| `run_model_contract/domain/` | DTOs (`UnifiedPayload`, `UnifiedIdentity`) and validation |
| `run_model_contract/io/` | Disk read/write adapters, legacy mapper, snapshot hook |
| `run_model_contract/refs.py` | Resolve model path from run or model directory |
| `run_model_contract/schema.py` | Artifact schema v2 helpers |
| `run_model_contract/env.py` | `SMARTTRAIN_UNIFIED_WRITE` and dual-write mode |

On-disk snapshots: `{run|model}/.smartrain/unified/` (legacy read fallback: `.smartrain/canonical/`).

## Glossary: backend vs model alias

| Term | Location | Meaning |
|------|----------|---------|
| **Execution backend** | `backends/contracts.py` (`TrainBackend`, `TestBackend`, `InferenceBackend`) | Runtime engine (ultralytics, onnxruntime, tensorrt) |
| **Ultralytics model alias** | `core/training/ultralytics_model_alias_registry.py` | YOLO YAML name aliases (`yolo11n`, …), not ONNX/TRT |

## backends/

Capability contracts, registry, Ultralytics and external provider adapters.

## tasks/

Task-specific metric adapters (detection / classification / segmentation).

## cli_entrypoints/

Typer-forwarded thin apps (`train_app.py`, `test_app.py`, …) and `support/` (`cli_replay.py`, `cli_contracts.py`, argparse helpers, `--nit` handling).

## external_providers/

Subprocess launchers and installer glue for external training/inference providers.

## providers/

CLI and index for optional external provider packages (`providers/cli.py`, `providers/core/global_index.py`).
