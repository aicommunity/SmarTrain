# Run layout (canonical)

SmarTrain normalizes each training run directory under `runs/<dataset>/<run_name>/`.

## Canonical tree

```
<run_dir>/
  models/                    # <run_name>.pt|onnx|engine|trt (+ .meta.json)
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

Dot dirs: `.smartrain_cache/` (analyze cache), `.smartrain/canonical/` (dual-write), `.ultralytics_*` (scratch; removed when empty).

## Legacy → action

| Legacy path | Action |
|-------------|--------|
| `train/` | merge → `train-ultralytics/` |
| `train-ultralytics-2`, `train-ultralytics3`, … | merge → `train-ultralytics/` |
| `test/`, `test-ultralytics/` at run root | merge → `tests/test-ultralytics/` |
| `tests/test-ultralytics2`, `tests/test-ultralytics-2`, … | merge or delete empty → `tests/test-ultralytics/` |
| `val-recs-*` at **run root** | move to `tests/val-recs-*` or delete if empty |
| empty any dir under run/tests | removed by `prune_empty_subdirs` |

## API

`ensure_run_layout(run_dir)` creates `models/`, `tmp/`, `tests/`, migrates legacy paths, then calls `canonicalize_run_ultralytics_layout(run_dir)` (idempotent).

Training uses `finalize_train_kwargs` with `exist_ok=True` and empty `train-ultralytics/` preflight removal.
