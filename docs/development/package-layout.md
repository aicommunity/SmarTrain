> Russian link: [../ru/development/package-layout.md](../ru/development/package-layout.md)

# SmarTrain package layout

One-line guide to `smartrain/` so you can navigate by **function** without reading the whole refactor archive. Details and wave status: [refactor/13-project-current-state.md](../refactor/13-project-current-state.md).

## Entry points

- `cli.py` — Typer router; forwards to argparse modules or `cli_apps/*`.
- `__main__.py`, `__init__.py` — package entry.

## workflows/

User-facing CLI flows and orchestration: argparse `main()`, Typer glue, dataset/train/test/inference/analyze pipelines.

Subpackages (each maps to commands or command groups):

- `training/` — train CLI (`train_entry`, `model_training_module`, `train_*_service`).
- `datasets/` — scan, fusion, augment, balance, prune, orient, report, CVAT helpers; `dataset_access.py` for filesystem layout; `dataset_cli_catalog.py` / `dataset_cli_common.py` for catalog + interactive selection.
- `testing/` — model test CLI and backends.
- `inference/` — inference CLI, backends, SAHI/heatmap helpers.
- `analyze/` — analyze subcommands; many `analyze_*_service.py` modules; `results_analyzer.py` builds the argparse tree.
- `queue/` — training queue (`training_queue.py`).
- `registry/` — registry CLI.
- `migration/` — migration CLI.
- `models/` — convert / release CLI.

**Rule:** new CLI surface area lives here first (`build_*_arg_parser`, `main(argv)`).

## services/

Use-case helpers shared across flows. **Must not** import `smartrain.workflows.*` (use `core/workflow_adapters/`).

## core/

Shared mechanics: `runtime/` (workspace paths, env), `training/` (profiles, catalogs), `workflow_adapters/` (thin facades to workflows for services), `inference/` (shared inference helpers).

## orchestrators/

- `canonical_gateway.py` — canonical reads, metrics, task context.

## Three canonical packages (do not confuse)

| Package | Role |
|---------|------|
| `smartrain/canonical/` | Schema v2, deprecation policy, refs |
| `smartrain/domain/canonical/` | DTOs (`CanonicalIdentity`, payloads) |
| `smartrain/adapters/canonical/` | Disk read/write, legacy mapper |

## Glossary: backend vs model alias

| Term | Location | Meaning |
|------|----------|---------|
| **Execution backend** | `backends/contracts.py` (`TrainBackend`, `TestBackend`, `InferenceBackend`) | Runtime engine (ultralytics, onnxruntime, tensorrt) |
| **Ultralytics model alias** | `core/training/ultralytics_model_alias_registry.py` | YOLO YAML name aliases (`yolo11n`, …), not ONNX/TRT |

## domain/ and adapters/

- `domain/canonical/` — DTOs and validation.
- `adapters/canonical/` — read/write adapters and snapshots.

## backends/

Capability contracts, registry, Ultralytics and external provider adapters.

## tasks/

Task-specific metric adapters (detection / classification / segmentation).

## cli_support/

`cli_replay.py`, `cli_contracts.py`, argparse/shared prompts — replay strings and CLI contracts.

## external_providers/

Subprocess launchers and installer glue for external training/inference providers.

## providers/

CLI and index for optional external provider packages (`providers/cli.py`, `providers/core/global_index.py`).
