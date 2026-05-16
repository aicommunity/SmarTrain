# Tech debt: CLI replay and `--nit`

## Register

| id | status | context | deferred_decision | resolution |
|----|--------|---------|-------------------|------------|
| TD-R01 | DONE | `cli.py` / `python -m` | Typer-only `--nit`; `python -m` without Typer does not strip `--nit`. | WONTFIX for unsupported `python -m` full replay strings; use `smartrain ...` or strip `--nit` before argparse-only tests. Documented in `docs/development/architecture.md`. |
| TD-R02 | DONE | `train` argparse | Unify `--nit` with `--yes` / `-y` dest or keep Typer-only `--nit` for train. | Typer-only `--nit` on the outer command; train keeps `--yes`/`-y` for workspace confirmations. Documented in `docs/development/architecture.md`. |
| TD-R03 | DONE | Replay call graph | Ensure all replay paths append `--nit` once. | Implemented in `build_non_interactive_command` (used by `emit_replay` and direct `print_replay_command` callers). |
| TD-R04 | DONE | Orchestrator H2 (`_formats_option_explicit_in_argv`) | Skip backend prompts when `--formats` in argv without Typer `--nit`. | Removed; replay always appends `--nit`; see `docs/cli/replay-and-non-interactive.md` (Legacy replay). |

### Audit: `build_non_interactive_command` / `emit_replay` (grep)

| Location | API |
|----------|-----|
| `smartrain/cli_support/cli_contracts.py` | `emit_replay` → `build_non_interactive_command` |
| `smartrain/workflows/testing/model_test_cli.py` | `emit_replay` |
| `smartrain/services/model_test_orchestrator.py` | `emit_replay` |
| `smartrain/services/training/train_cli_main.py` | `emit_replay` via callback |
| `smartrain/workflows/training/training_cli_orchestration_service.py` | `emit_replay_cb` |
| `smartrain/services/train_service.py` | `emit_replay` |
| `smartrain/workflows/inference/inference_cli.py` | `emit_replay` |
| `smartrain/workflows/analyze/analyze_all_finalize_service.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_prune.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_orient.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_balance.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_augment.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_stats.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_report.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_roi_yolo.py` | `build_non_interactive_command` |
| `smartrain/workflows/datasets/dataset_former.py` | `build_non_interactive_command` |

## Burn-down checklist

- [x] TD-R01
- [x] TD-R02
- [x] TD-R03
