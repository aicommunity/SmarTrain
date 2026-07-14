> Russian version: [../ru/cli/overview.md](../ru/cli/overview.md)

# CLI: overview

Entry point: `smartrain` (Typer router with unified command behavior).

## Command groups

- Datasets: `scan`, `normalize-data-yaml`, `fusion`, `augment`, `balance`, `prune`, `filter`, `orient`, `rotate`, `roi`, `inference`, `hash`, `stats`, `dataset report`, `dataset rename`
- Training: `train`, `clearml-upload`
- Providers: `providers`
- Workspace: `deploy`, `quickstart`, `info`, `sync`
- Queue: `queue`, `queue-run`
- Analytics: `analyze`, `plot` (outdated wrapper)
- Register: `registry`
- Models: `model convert`, `model release`, `model comment`, `model rename`
- Dataset catalog: `dataset report`, `dataset rename`
- Format tools: `dataset convert`, `sahi`, `heatmap`, `vis`
- Migration: `migrate`, `migrate-models`
- Maintenance: `deps sync-torch`

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
- for `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `rotate`, `dataset report`, `dataset rename`, `model convert`, `model release`, `model comment`, `model rename`, empty invocation enters interactive mode;
- if any arguments are provided but required ones are missing, command exits with a clear "incomplete arguments" error (no interactive prompts).
Most important commands and groups also include `Examples` / `Quick examples` directly in help output.

Completion:

- auto: best-effort setup runs on first `smartrain` launch;
- manual fallback:
  - `smartrain --install-completion`
  - `smartrain --show-completion`

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
- Released models under `models/<dataset>/` use the source run folder name as identity stem (`models/<dataset>/<run_name>/<run_name>.pt` + matching `.json`); convert writes short `{stem}.onnx`/`.engine`/`.trt` next to the `.pt` for both runs and releases.
- Dedicated `*_trtprep.onnx` is an internal cache for `trtexec` only: created when TensorRT-trt is requested, kept for training runs after success, and cleaned up for catalog/release models. Pure `--format onnx` does not write `*_trtprep`.

Inference highlights:

- `smartrain inference` supports local model artifacts `pt`, `onnx`, `engine`, `trt` through unified backend routing, plus external provider references.
- `smartrain inference` writes `inference/<model>/<timestamp>-<source>/inference_results.json`. By default it also exports a YOLO autolabel dataset under `<basename>_autolabeled/` split into independent `part_XXX/` sub-datasets (with `autolabel_manifest.json`) and optional `pred_overlays/`; use `--no-export-dataset` or `--no-export-split-dirs` to change that. Empty exports (no labels after the confidence filter) do not create dataset or overlay folders. The `vis` command runs inference internally with export disabled.
- Inference report now includes dual performance profile (`performance.end_to_end` and `performance.infer_only`) with warmup-separated steady stats.
- Inference run saves `environment_profile.json` next to `inference_results.json` with machine and key framework/python versions for reproducibility.
- Full inference JSON/artifact contract: [`inference.md`](inference.md).
- `pt_uni` is internal-only and used for PT vs PT-uni comparison table generation (test/val), not as a user-facing inference format. The model-test internal compare path supports detection/classification/segmentation task-aware routing.

Model release highlights:

- `smartrain model release` publishes canonical run model `<run_dir_name>.pt` from a selected run into a self-contained folder `models/<dataset>/<run_dir_name>/` (same detailed name as the training run: weights, sidecar JSON, and copied train artifacts).
- A global catalog `models/releases_manifest.json` stores one-line comments for all release models; the same comment is duplicated in each model's sidecar JSON.
- Interactive mode prompts for an optional one-line comment (any language); non-interactive mode accepts `--comment`.
- Re-running for the same run with the same source hash performs a no-op skip.

Model comment highlights:

- `smartrain model comment` sets or updates the one-line comment for a released model in `releases_manifest.json` and the sidecar JSON.
- Interactive mode lists released models (with current comments) and pre-fills the comment field for editing.

Model rename highlights:

- `smartrain model rename` renames a released model in `models/<dataset>/` by changing the release stem (`.pt`, sidecar `.json`, release folder, and converted ONNX/engine/trt files with matching prefix).
- Registry-promoted bundles (`model_manifest.json`) and run models under `runs/` are not affected.
- Interactive mode lists released models and pre-fills the current stem for editing.

Analyze highlights:

- `smartrain analyze` (TTY, no subcommand) runs the interactive `analyze all` workflow (compare, metrics, optional speed/PR, report).
- Quality artifacts use training-time metrics; a separate `smartrain test` run is not required for compare and test-metrics plots.
- For `profile=full`, speed (inference benchmark) resolves frames from `test`, then `val`, then `train` split; if no split images exist, speed/PR stages degrade with warnings and the session report still completes.
- Runtime `_runtime_data_*.yaml` paths are resolved via the `path:` field in data.yaml (not relative to the yaml file location in `run/tmp/`).
- Use `--strict-diagnostics` only when missing PR/metric_sources artifacts must fail the session.

Dataset report highlights:

- `smartrain dataset report` writes a multilingual per-class sample report (Markdown + PNG; default folder `analytics/datasets-reports/<dataset>_<timestamp>/`). PDF/ODT export helpers (`pypandoc-binary`, `weasyprint`) are installed via optional extra `pip install -e ".[export]"`; `fpdf2` and `odfpy` remain base dependencies. WeasyPrint may need OS libraries (Cairo, Pango) if wheels are unavailable.

Dataset rename highlights:

- `smartrain dataset rename` renames a workspace dataset key and directory (`datasets/<name>/`), plus related `runs/<name>/` and `models/<name>/` folders when present.
- Updates `datasets_info.json`, `dataset_passport.json` references, run metadata, `queue.txt`, and analytics artifacts.
- Use `--dry-run` to preview the plan; use `--move-data-path` when `data_path` points outside the default `datasets/<name>/` layout.

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
- `smartrain filter` removes edge-truncated bbox annotations into `<dataset>_fltd` (baseline inset stats + relative/absolute thresholds; audit under `_filter_audit/`; `--stats-only`, `--drop-images`, interactive preview).
- `smartrain scan --strip-unused-classes` strips unused classes for **new** datasets during scan (default **on**; use `--no-strip-unused-classes` to disable).
