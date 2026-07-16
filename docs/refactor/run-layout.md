# Run layout

SmarTrain normalizes each training run directory under `runs/<dataset>/<run_name>/`.

## Standard tree

```
<run_dir>/
  models/                    # <stem>.pt|onnx|engine|trt (+ .meta.json / sidecar .json)
                             # Prefer detect_* stem (e.g. detect_yolo11s_…_b16.pt);
                             # legacy <run_name>.pt still resolves.
  tmp/                       # _runtime_data_{train,test}.yaml
  tests/
    test-ultralytics/        # PT Ultralytics val artifacts
    test_{format}/           # onnx, engine, trt, pt_uni
    test_metrics*.csv
    val_metrics*.csv
    confidence_recommendations_*.json
    val-recs-{format}/       # ephemeral val runs for confidence recs (may be empty → removed)
    val-ultralytics-recs/    # builtin train val-recs (ephemeral)
  train-ultralytics/         # single Ultralytics train save dir
  training_metadata.json
```

Dot dirs: `.smartrain_cache/` (analyze cache), `.smartrain/unified/` (unified snapshot dual-write; legacy `.smartrain/canonical/` read fallback), `.ultralytics_*` (scratch; removed when empty).

## Weight path API

| API | Meaning |
|-----|---------|
| `preferred_run_model_path(run_dir)` | Canonical path under `models/<stem>.pt` (does **not** create files or directories). Stem from metadata / detect_* / sole `.pt` / folder name. Ignores legacy metadata refs like `train/weights/best.pt`. |
| `resolve_run_model(run_dir)` | First **existing** weight: preferred, release layouts, then legacy (`train-ultralytics/weights/{best,last}.pt`, `train/weights/…`). |
| `materialize_preferred_run_model(run_dir)` | Copy/move from `resolve_run_model` into the preferred path when missing. |
| `ensure_run_layout(run_dir)` | Creates `models/`, `tmp/`, `tests/`, migrates legacy run paths. **Do not** call on release catalog dirs — use `ensure_runtime_tmp_dir` / `ensure_runtime_layout_for_yaml` (tmp only). |

## Run vs release layouts

Training runs live under `runs/`. Published weights live under workspace `models/` (release catalog). Compatible release shapes:

| Layout | Path pattern | Notes |
|--------|--------------|-------|
| **R3 (current)** | `models/<dataset>/<run_id>/detect_*.pt` | Folder name = training run id; weight stem = `detect_*…`. Sidecar `detect_*.json` + comment. |
| **R1 (compat)** | `models/<dataset>/<detect_stem>/models/<detect_stem>.pt` | Nested `models/` inside release folder. |
| **R2 (compat)** | `models/<dataset>/<stem>.pt` next to `models/<dataset>/<stem>/` | Flat sibling weight. |
| **Registry bundle** | `models/<friendly_name>/` + `model_manifest.json` | From `registry models-add`; not the same as `model release`. |

Comments: `models/releases_manifest.json` keys as `<dataset>/<weight_stem>`; lookup also accepts `<dataset>/<folder>` for R3 when the PT is only under nested `models/`.

Inference / analyze / ROI discover resolve all of the above via `resolve_run_model` / `discover_model_entries`.

## Legacy → action

| Legacy path | Action |
|-------------|--------|
| `train/` | merge → `train-ultralytics/` |
| `train-ultralytics-2`, `train-ultralytics3`, … | merge → `train-ultralytics/` |
| `test/`, `test-ultralytics/` at run root | merge → `tests/test-ultralytics/` |
| `tests/test-ultralytics2`, `tests/test-ultralytics-2`, … | merge or delete empty → `tests/test-ultralytics/` |
| `val-recs-*` at **run root** | move to `tests/val-recs-*` or delete if empty |
| empty any dir under run/tests | removed by `prune_empty_subdirs` |

## Notes

Training uses `finalize_train_kwargs` with `exist_ok=True` and empty `train-ultralytics/` preflight removal.

See also: [`../cli/overview.md`](../cli/overview.md) (release/comment), [`../cli/registry.md`](../cli/registry.md) (registry vs release).
