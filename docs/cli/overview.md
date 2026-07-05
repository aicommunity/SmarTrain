> Russian version: [../ru/cli/overview.md](../ru/cli/overview.md)

# CLI: overview

Entry point: `smartrain` (Typer router with unified command behavior).

## Command groups

- Datasets: `scan`, `normalize-data-yaml`, `fusion`, `augment`, `balance`, `prune`, `orient`, `rotate`, `roi`, `inference`, `hash`, `stats`, `report dataset`
- Training: `train`, `clearml-upload`
- Providers: `providers`
- Info: `info`
- Queue: `queue`, `queue-run`
- Analytics: `analyze`, `plot` (outdated wrapper)
- Register: `registry`
- Models: `model convert`, `model release`, `model rename`
- Format tools: `cvat`, `sahi`, `heatmap`
- Migration: `migrate`, `migrate-models`

## Reference

```bash
smartrain --help
smartrain <command> --help
```

For nested commands:

```bash
smartrain queue list --help
smartrain analyze inference-benchmark --help
smartrain model convert --help
```

For argparse-forwarded commands exposed by Typer wrappers, use:

```bash
smartrain <command> -- --help
smartrain <group> <subcommand> -- --help
```

Unified interactive contract:

- interactive mode starts only when a command is run with zero arguments (TTY required);
- for `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `rotate`, `report dataset`, `model convert`, `model release`, `model rename`, empty invocation enters interactive mode;
- if any arguments are provided but required ones are missing, command exits with a clear "incomplete arguments" error (no interactive prompts).
Most important commands and groups also include `Examples` / `Quick examples` directly in help output.

`smartrain info` highlights:

- Prints `Supported train models` section for copy-paste use in `smartrain train --model ...`.
- Includes default backend aliases plus provider-scoped aliases for installed external providers.

Model convert highlights:

- `smartrain model convert` exports `.pt` to `onnx`, `tensorrt-engine`, and `tensorrt-trt`, and supports `.onnx -> tensorrt-trt`.
- Defaults: static batch mode, `--batch 1`, `--precision fp32`.
- ONNX export options are configured in `model convert` (`--opset`, `--simplify/--no-simplify`, `--half/--no-half`).
- Interactive mode auto-discovers `.pt/.onnx` candidates in workspace `models/` and `runs/` and allows source selection by number or manual path input.
- Target selection is model-based (`onnx`, `engine`, `trt`) with multi-select input (`1,2` or `onnx,trt`), and unavailable targets are shown with reason.
- For run sources, interactive discovery uses canonical run artifacts (`<run_dir>/<run_dir_name>.<ext>`). Legacy run layouts are canonized automatically on first access.

Inference highlights:

- `smartrain inference` supports local model artifacts `pt`, `onnx`, `engine`, `trt` through unified backend routing, plus external provider references.
- Inference report now includes dual performance profile (`performance.end_to_end` and `performance.infer_only`) with warmup-separated steady stats.
- Inference run saves `environment_profile.json` next to `inference_results.json` with machine and key framework/python versions for reproducibility.
- Full inference JSON/artifact contract: [`inference.md`](inference.md).
- `pt_uni` is internal-only and used for PT vs PT-uni comparison table generation (test/val), not as a user-facing inference format. The model-test internal compare path supports detection/classification/segmentation task-aware routing.

Model release highlights:

- `smartrain model release` publishes canonical run model `<run_dir_name>.pt` from a selected run into `models/<dataset>/<task>_<model>_<train_datetime>.pt`.
- A sidecar JSON with the same basename is created next to the model file and includes source/training/metrics/classes/io specification.
- Re-running for the same run with the same source hash performs a no-op skip.

Model rename highlights:

- `smartrain model rename` renames a released model in `models/<dataset>/` by changing the release stem (`.pt`, sidecar `.json`, release artifact directory, and converted ONNX/engine/trt files with matching prefix).
- Registry-promoted bundles (`model_manifest.json`) and run models under `runs/` are not affected.
- Interactive mode lists released models and pre-fills the current stem for editing.

Migration highlights:

- `smartrain migrate unified --mode dry-run` previews unified migration without writing files.
- `smartrain migrate unified --mode apply` writes unified snapshots and reports.

Balance and stats additions:

- `smartrain balance` supports `weights`, `rfs`, and `hybrid` strategies, plus weight/rfs tuning flags.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default,hybrid-aug-tail-budget}` applies tuned defaults for common scenarios.
- For `--strategy hybrid-aug`, a constrained-growth tail-first mode is enabled by default: `--aug-total-bbox-cap-mult 1.10`, `--aug-budget-tail-first`, `--aug-budget-tail-gamma 1.0`, `--train-head-bbox-undersample median-factor`, `--train-head-bbox-cap-mult 5.0`, plus conservative eval head trimming `--eval-head-bbox-undersample median-factor --eval-head-bbox-cap-mult 8.0 --eval-head-bbox-min-count 30 --eval-head-bbox-max-remove-frac 0.35` (override with explicit flags).
- `smartrain balance --eval-coverage` (default on) adjusts the balanced train pool so `val`/`test` stay non-empty when possible and missing classes in eval splits are filled from train; `--no-eval-coverage` disables this. Interactive `balance` prompts for the same choice.
- `smartrain stats --balance-ready` prints imbalance metrics and balancing recommendations.
- `smartrain stats --after-augment` compares per-class train bbox counts after balance vs after augment (reads `balance_manifest.json` from hybrid-aug outputs).
- Standalone `smartrain augment --preset augment-tail-safe` enables class-aware geo, bbox cap 1.10×, and tail-first budget (same augment knobs as `balance --preset hybrid-aug-tail-budget` without running balance).
- `smartrain prune empty` removes empty image/label pairs into a new `<dataset>_pruned` dataset.
- `smartrain prune dedup` removes duplicate images by file content into `<dataset>_deduped` (global split priority: train > val > test).
- `smartrain prune classes` removes unused classes from metadata into `<dataset>_classes_pruned` (files kept; `class_id` remapped).
- `smartrain scan --strip-unused-classes` strips unused classes for **new** datasets during scan (default **on**; use `--no-strip-unused-classes` to disable).
- `smartrain report dataset` writes a multilingual per-class sample report (Markdown + PNG; default folder `analytics/datasets-reports/<dataset>_<timestamp>/`). Default dependencies include bundled pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2`, and `odfpy` for PDF/ODT. WeasyPrint may need OS libraries (Cairo, Pango) if wheels are unavailable.
