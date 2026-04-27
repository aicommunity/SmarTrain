> Russian version: [../ru/cli/overview.md](../ru/cli/overview.md)

# CLI: overview

Entry point: `smartrain` (Typer router with unified command behavior).

## Command groups

- Datasets: `scan`, `normalize-data-yaml`, `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`, `inference`, `hash`, `stats`, `report dataset`
- Training: `train`, `clearml-upload`
- Providers: `providers`
- Info: `info`
- Queue: `queue`, `queue-run`
- Analytics: `analyze`, `plot` (outdated wrapper)
- Register: `registry`
- Models: `model convert`, `model release`
- Format tools: `cvat`, `sahi`, `heatmap`
- Migration: `migrate-models`

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

Unified interactive contract:

- interactive mode starts only when a command is run with zero arguments (TTY required);
- for `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `report dataset`, `model convert`, `model release`, empty invocation enters interactive mode;
- if any arguments are provided but required ones are missing, command exits with a clear "incomplete arguments" error (no interactive prompts).
Most important commands and groups also include `Examples` / `Quick examples` directly in help output.

`smartrain info` highlights:

- Prints `Supported train models` section for copy-paste use in `smartrain train --model ...`.
- Includes default backend aliases plus provider-scoped aliases for installed external providers.

Model convert highlights:

- `smartrain model convert` exports `.pt` to `onnx`, `tensorrt`, or `both`, and also supports direct `.onnx -> tensorrt` conversion.
- Defaults: static batch mode, `--batch 1`, `--precision fp32`.
- ONNX export options are configured in `model convert` (`--opset`, `--simplify/--no-simplify`, `--half/--no-half`).
- Interactive mode auto-discovers `.pt/.onnx` candidates in workspace `models/` and `runs/` and allows selection by number or manual path input.

Model release highlights:

- `smartrain model release` publishes only `train/weights/best.pt` from a selected run into `models/<dataset>/<task>_<model>_<train_datetime>.pt`.
- A sidecar JSON with the same basename is created next to the model file and includes source/training/metrics/classes/io specification.
- Re-running for the same run with the same source hash performs a no-op skip.

Balance and stats additions:

- `smartrain balance` supports `weights`, `rfs`, and `hybrid` strategies, plus weight/rfs tuning flags.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` applies tuned defaults for common scenarios.
- `smartrain balance --eval-coverage` (default on) adjusts the balanced train pool so `val`/`test` stay non-empty when possible and missing classes in eval splits are filled from train; `--no-eval-coverage` disables this. Interactive `balance` prompts for the same choice.
- `smartrain stats --balance-ready` prints imbalance metrics and balancing recommendations.
- `smartrain prune empty` removes empty image/label pairs into a new `<dataset>_pruned` dataset.
- `smartrain prune dedup` removes duplicate images by file content into `<dataset>_deduped` (global split priority: train > val > test).
- `smartrain report dataset` writes a multilingual per-class sample report (Markdown + PNG; default folder `analytics/datasets-reports/<dataset>_<timestamp>/`). Default dependencies include bundled pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2`, and `odfpy` for PDF/ODT. WeasyPrint may need OS libraries (Cairo, Pango) if wheels are unavailable.
