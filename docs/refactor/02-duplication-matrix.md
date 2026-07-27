# Duplication Matrix

Updated: 2026-07-26 (post–project audit P0+P1+P2 wave).

| Area | Files | Status | Notes / next |
|---|---|---|---|
| Dead augment compat shims | `augment_{cli,pipeline,donors,yolo_io}.py` | **Closed (P0.2)** | Removed |
| MFEL ConvModule shim | `mfel_*_launcher.py` | **Closed (P0.4)** | [`mfel_shim.py`](../../smartrain/external_providers/launchers/mfel_shim.py) |
| Interactive dataset preamble | dataset_* | **Partial (P0.3)** | [`cli_interactive.py`](../../smartrain/cli_entrypoints/support/cli_interactive.py) |
| Registry metadata read | `registry_cli.py` | **Closed (P0.1)** | [`run_fields.py`](../../smartrain/services/registry/run_fields.py) + gateway |
| `_resolve_run_ref` | registry / inference / release | **Closed (P1.5)** | [`run_refs.py`](../../smartrain/core/runtime/run_refs.py) |
| CLI prompts dual stack | train_prompts / ROI | **Partial (P1.4)** | yes_no/int shared via `cli_prompts` |
| God `model_convert` | was workflow monolith | **Closed (P1.1)** | [`model_convert_service.py`](../../smartrain/services/models/model_convert_service.py) + thin facade |
| God `dataset_balance` | monolith | **Partial (P1.3)** | `balance_{strategies,eval_coverage,presets}.py` extracted |
| God `dataset_augment` | monolith | **Partial (P1.2)** | `augment_{cli_parser,budget}.py` extracted; orchestrator still large |
| Replay builders | train/test/inference/dataset | Partial | Prefer `cli_replay.py` |
| P2 science/MLOps | IRFS/LRP/SAHI-FT/queue/Docker/harness/core↔services | **Done (execution=1)** | [09-tech-debt.md](./09-tech-debt.md), [audit](../audit/2026-07-26-project-audit.md) |

## Priority (remaining follow-ups)

1. Further thin `dataset_augment` / analyze gods if needed
2. Full prompt API consolidation (train `prompt_input` completer path)
3. Raise coverage floor; expand mypy on `run_model_contract` without ignore_errors
