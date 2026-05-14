# Tech debt: ultralytics test restoration

## Register

| id | status | context | deferred_decision | resolution |
|----|--------|---------|-------------------|------------|
| TD-001 | DONE | `model_training_module` post-train / `train_service` | Post-training `test_yolo` still uses `plots=False`; full plot bundle is produced by `smartrain test` and resume path. | Documented: intentional; train keeps fast smoke test. |
| TD-002 | DONE | `tests/test_run_ultralytics_val_kwargs.py` | Plan §4.2: assert `YOLO.val` kwargs (`project`→`tests`, `name`, `plots`, `save`). | Implemented: patched `YOLO` + post-val hooks; assert canonical kwargs. |
| TD-003 | DONE | `ultralytics_test_contract.ultralytics_pt_rich_files_required` | Plan §0.2: explicit segmentation branch vs detection plot set. | Same tuple as detection: Ultralytics `SegmentMetrics` still runs box curves with `Box*` prefix before mask plots. |
| TD-004 | WONTFIX | CI / `pytest -m slow` | Plan §4.4: disk snapshot after real `val` on CPU. | Deferred: full suite already ~8m; add when a tiny public seg/det weights fixture is standardized for CI. |
| TD-005 | DONE | `model_test_orchestrator.run_model_test_after_setup` | Plan §3.1 alt: MPL before PT if entry bypasses CLI. | `ensure_matplotlib_training_runtime` already called at start of orchestrator (non_interactive from args). |
| TD-006 | WONTFIX | `_finalize_ultralytics_pt_test_dir` PR CSV writes | Plan §3.2: skip overwriting `pr.csv` without `--force`. | Deferred: naive `not args.force` breaks normal re-test; needs `should_rerun` or separate `overwrite_pr_metrics` API — out of scope for this closure. |

## Burn-down checklist

- [x] TD-001
- [x] TD-002
- [x] TD-003
- [x] TD-005
- [x] TD-004 (WONTFIX — documented)
- [x] TD-006 (WONTFIX — documented)
