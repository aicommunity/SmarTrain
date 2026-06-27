> Russian version: [../ru/cli/training.md](../ru/cli/training.md)

# CLI: training

## `train`

Core training/validation team.

```bash
smartrain train --data my_dataset -y
smartrain train --data my_seg_dataset --task segment --model yolo11s-seg.pt -y
smartrain train --test-only --model-dir /path/to/run --data /path/to/dataset
```

### Instance segmentation

- Use `--task segment` (or `segmentation` in metadata) with a `*-seg.pt` model alias (e.g. `yolo11s-seg.pt`).
- Dataset labels must be YOLO polygons (`class_id x1 y1 x2 y2 ...`). See [data formats](../reference/data-formats.md).
- Interactive mode filters model list to `-seg` aliases when task is segment.
- Post-train smoke test uses Ultralytics `val`; full plot bundle: `smartrain test --formats pt --task segment`.
- Native ONNX/engine/TRT test for segmentation is skipped by default (bbox-only native eval). Use PT test for mask metrics.
- Experimental bypass: `smartrain test --formats onnx --task segment --force-native-seg-test` (bbox-shaped native eval; mask metrics unreliable).

Parameter sources and priority:

`CLI > --ultralytics_yaml > --config > defaults`

The `data` field from `--ultralytics_yaml` is ignored, the selected `--data` is used.
Also ignored from `--ultralytics_yaml`: `project`, `name`, `exist_ok`, `cfg`, `device`, `model_dir`, `target_path`, `workspace`.

That is, command line options always have the highest priority.

Model selection:

- To list aliases supported by the default training backend, run `smartrain info`.
- In interactive mode, `train` offers model selection from this list and provides `<manual>` for custom values.
- `--model` accepts both YOLO aliases and explicit weights path; for plain YOLO aliases `.pt` is added automatically.
- For external providers, provider-scoped refs are supported: `provider:model` (example: `dr-yolo:yolov8n`).
- External provider aliases are strictly validated against provider catalog; unsupported aliases fail before launch.
- If `--external-provider` is set and `--model` is not specified, provider default model alias is used automatically.

Device selection:

- `--device` accepts: `cpu`, GPU index (`0`), `cuda:N`, or GPU name token in interactive mode.
- Default behavior (all commands): `GPU 0` if CUDA is available, otherwise `cpu`.
- In interactive mode you can select by:
  - list number,
  - explicit token (`cpu`, `0`, `cuda:0`),
  - GPU name (exact/normalized match).
- The same selection model is used across `train`, `test`, and `inference`.

## `clearml-upload`

A separate command to load run artifacts into ClearML:

```bash
smartrain clearml-upload /path/to/run_folder --project MyProject
```

During training, you can enable integration with the `--clearml` flag.
