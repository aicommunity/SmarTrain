> Russian version: [../ru/cli/inference.md](../ru/cli/inference.md)

# CLI: inference

`smartrain inference` runs object detection inference on either a folder or a dataset split.

It writes:

- `inference_results.json` (main report)
- `environment_profile.json` (machine/runtime profile)

Both files are saved under:

- `workspace/inference/<model>/<timestamp-source>/`

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
smartrain inference --weights ./runs/ds/run_001/models/run_001.engine --data-mode folder --source-dir ./images
smartrain inference --weights dr-yolo:yolov8n --external-repo /opt/dr-yolo --data-mode folder --source-dir ./images
```

## Device selection

- `--device` supports `cpu`, GPU index (`0`), or `cuda:N`.
- Interactive mode supports number, token, or GPU name input.
- Default device is `GPU 0` when CUDA is available, otherwise `cpu`.
- The same rules are used in `train` and `test`.

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
