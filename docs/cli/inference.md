> Russian version: [../ru/cli/inference.md](../ru/cli/inference.md)

# CLI: inference

`smartrain inference` runs inference on a folder, archive, or dataset split (detection, instance segmentation, classification depending on model/task).

It writes:

- `inference_results.json` (main report)
- `environment_profile.json` (machine/runtime profile)
- by default a YOLO autolabel dataset under `<basename>_autolabeled/` (pseudo-labels from predictions)
- optional `pred_overlays/` with rendered predictions (default on when dataset export is on)
- optional polygon overlay images when `--save-overlay` is set (instance segmentation, legacy)

Both primary JSON files are saved under:

- `workspace/inference/<model>/<timestamp-source>/`

## YOLO autolabel export

Enabled by default (`--export-dataset`, disable with `--no-export-dataset`).

By default export is split into **independent YOLO sub-datasets** (`--export-split-dirs`, disable with `--no-export-split-dirs`). Each `part_XXX/` holds up to `--export-files-per-dir` **actually exported** images (after label confidence filter; default `500`).

Split layout (default):

```
<basename>_autolabeled/
  autolabel_manifest.json          # root index (layout=independent_parts)
  part_000/
    images/
    labels/
    data.yaml
    autolabel_manifest.json
  part_001/
    ...
pred_overlays/
  part_000/
  part_001/
```

Flat layout (`--no-export-split-dirs`):

```
<basename>_autolabeled/
  images/
  labels/
  data.yaml
  autolabel_manifest.json
```

- **`basename`** is the source folder or archive name (`--source` / `--source-dir`) or `{dataset}-{split}` for `dataset-split`.

## Data sources

| Mode | Flags | Archive support |
|------|-------|-----------------|
| `folder` | `--source` or `--source-dir` | Yes: `.zip`, `.tar`, `.tar.gz`, `.tgz` — extracted to `tmp/extracted_datasets/` |
| `dataset-split` | `--dataset`, `--split` | Yes when `data_path` in `datasets_info.json` points to an archive |

Archives are unpacked into the workspace cache (`tmp/extracted_datasets/`) with mtime/size invalidation. When an archive was used, `source.source_archive_*` in the report keeps the original archive path.

- Only frames with **≥1 detection/segment** after the export confidence filter are included.
- **`autolabel_manifest.json`** records model, inference/export parameters, summary stats, and `file_mapping` (per part when split; root index lists parts).

Export / inference flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--export-dataset` / `--no-export-dataset` | on | Export YOLO dataset |
| `--export-label-conf-min` | `0.25` | Min confidence for label export |
| `--export-label-conf-max` | `1.0` | Max confidence for label export |
| `--export-visualize` / `--no-export-visualize` | on when export-dataset | `pred_overlays/` vis-style renders |
| `--export-split-dirs` / `--no-export-split-dirs` | on | Independent `part_XXX/` sub-datasets (+ mirrored overlays) |
| `--export-files-per-dir` | `500` | Max exported images per sub-dataset |
| `--batch-size` | `8` | Local Ultralytics inference batch size (ignored for external providers) |

`--conf` is the inference threshold; `--export-label-conf-*` further filters labels written to the dataset from predictions already returned by the model.

- Classification is not exported to YOLO (warning, no dataset folder created).
- If no images pass the export confidence filter, `<basename>_autolabeled/` and `pred_overlays/` are not created.
- Large `--batch-size` with high `--img-size` may OOM; lower `--batch-size` if needed.

## External provider capability matrix (task × payload)

External adapters may return a degraded contract when task-specific fields are missing. Reports surface this via `images[].capability_gap`, `images[].capability_gap_reason`, and `summary.capability_gap_reasons`.

| Task | Expected in `task_outputs` | `capability_gap` when |
|------|----------------------------|------------------------|
| `classification` | `classification` object (per-image) | object missing or empty |
| `segmentation` | `segments` list (per-image) | list missing or empty |
| `detection` | `detections` list (legacy flat field also accepted) | not used for gap detection in the same way; relies on list lengths |

Stable reason tokens (examples): `missing_task_outputs.classification`, `missing_task_outputs.segments`.

## Supported model types

Local model artifacts:

- `pt`
- `onnx`
- `engine`
- `trt`

External model references:

- provider-scoped refs like `provider:model` (`--weights` / `--model-name`) with `--external-provider` flow.

Note: `pt_uni` is an internal metrics-comparison mode (PT vs PT-uni, test/val) and is not a user-facing inference model type.

## Quick examples

```bash
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --device cpu
smartrain inference --model-name my_model --data-mode folder --source raw_data/images.zip --device cpu
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --no-export-dataset
smartrain inference --model-name my_model --data-mode folder --source-dir ./images --export-label-conf-min 0.4 --export-label-conf-max 0.9
smartrain inference --weights ./runs/ds/run_001/models/run_001.engine --data-mode folder --source-dir ./images
smartrain inference --weights dr-yolo:yolov8n --external-repo /opt/dr-yolo --data-mode folder --source-dir ./images
smartrain inference --weights yolo11s-seg.pt --data-mode folder --source-dir ./images --save-overlay
```

### Instance segmentation overlay

For `*-seg.pt` models, inference JSON includes per-image `segments` (polygon vertices). Use `--save-overlay` to write RGB preview images with GT-style polygon outlines next to the JSON report (under the same `workspace/inference/...` output directory).

```bash
smartrain inference --weights runs/ds/run_seg/models/best-seg.pt --data-mode folder --source-dir ./images --save-overlay
```

## Device selection

- `--device` supports `cpu`, GPU index (`0`), or `cuda:N`.
- Interactive mode supports number, token, or GPU name input.
- Default device is `GPU 0` when CUDA is available, otherwise `cpu`.
- The same rules are used in `train` and `test`.

## Input resolution (`--img-size`)

When `--img-size` is omitted, inference resolves input size from model context (priority order):

1. `training_metadata.json`, `args.yaml`, and other metadata files next to the model or in ancestor directories up to `models/`
2. sidecar `*.meta.json` next to the weights file
3. artifact filename token `_imgsz{N}x{N}_` (e.g. ONNX after `model convert`)
4. static ONNX graph input H/W

If no source is found, fallback **640** is used with a `[WARN]` message. Explicit `--img-size` always wins (`img_size_source: cli`).

`inference_results.json` → `parameters.img_size_source` records the resolved source label (e.g. `training_metadata`, `artifact_filename`, `fallback_640`).

## Performance contract

Report contains dual profile under `performance`:

- `performance.end_to_end` - image I/O + preprocessing + inference call + postprocessing + report update
- `performance.infer_only` - backend inference call timing
- `performance.stage_breakdown_ms` - stage-level timings when backend exposes them
- `performance.methodology` - profiling context and caveats

Latency stats format (for both `end_to_end` and `infer_only`):

- `images_total`
- `warmup_images`
- `duration_s`
- `throughput_img_s`
- `latency_ms.all` with `count/mean/p50/p90/p95/p99/min/max/std`
- `latency_ms.steady` with the same fields after warmup exclusion

## Artifacts contract

`inference_results.json` top-level sections:

- `created_at`
- `workspace`
- `model`
- `parameters`
- `source`
- `output`
- `summary`
- `performance`
- `artifacts`
- `images`

`artifacts.environment_profile`:

- `path_absolute`
- `path_relative`

`artifacts.autolabel_dataset` (when `--export-dataset`):

- `path_absolute`, `path_relative`
- `manifest_absolute` → `autolabel_manifest.json`
- `images_exported`, `labels_total`

`artifacts.pred_overlays` (when `--export-visualize`):

- `path_absolute`, `path_relative`
- `images_rendered`

`environment_profile.json` includes:

- host/OS info (platform, kernel, cpu count, machine)
- python info (version, executable, implementation)
- framework versions (`torch`, `ultralytics`, `onnxruntime`, `tensorrt`, `numpy`, `pillow`)
- best-effort GPU info (`nvidia-smi` output, CUDA availability via torch)

## Caveats

- External providers currently may not expose per-image telemetry; in this case detailed stage timings are unavailable.
- Stage breakdown is backend-dependent and can be partially populated.
- Cross-backend comparisons must account for runtime differences (provider selection, CUDA/TRT runtime, precision, and model export specifics).
- `infer_only` is useful for backend-level comparison, while `end_to_end` is better for user-facing pipeline latency.
- Exported labels are pseudo-labels from the model, not ground truth.
- With export-dataset enabled, `pred_overlays/` contains only frames written to the autolabel dataset.
