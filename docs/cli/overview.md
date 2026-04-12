> Russian version: [../ru/cli/overview.md](../ru/cli/overview.md)

# CLI: overview

Entry point: `smartrain` (Typer router with unified command behavior).

## Command groups

- Datasets: `scan`, `fusion`, `augment`, `balance`, `prune`, `orient`, `roi`, `hash`, `stats`, `report dataset`
- Training: `train`, `clearml-upload`
- Queue: `queue`, `queue-run`
- Analytics: `analyze`, `plot` (outdated wrapper)
- Register: `registry`
- Format tools: `cvat`, `sahi`, `heatmap`

## Reference

```bash
smartrain --help
smartrain <command> --help
```

For nested commands:

```bash
smartrain queue list --help
smartrain analyze inference-benchmark --help
```

Unified interactive contract:

- interactive mode starts only when a command is run with zero arguments (TTY required);
- for `train`, `fusion`, `augment`, `balance`, `stats`, `roi`, `orient`, `report dataset`, empty invocation enters interactive mode;
- if any arguments are provided but required ones are missing, command exits with a clear "incomplete arguments" error (no interactive prompts).
Most important commands and groups also include `Examples` / `Quick examples` directly in help output.

Balance and stats additions:

- `smartrain balance` supports `weights`, `rfs`, and `hybrid` strategies, plus weight/rfs tuning flags.
- `smartrain balance --preset {weights-safe,rfs-aggressive,hybrid-default}` applies tuned defaults for common scenarios.
- `smartrain balance --eval-coverage` (default on) adjusts the balanced train pool so `val`/`test` stay non-empty when possible and missing classes in eval splits are filled from train; `--no-eval-coverage` disables this. Interactive `balance` prompts for the same choice.
- `smartrain stats --balance-ready` prints imbalance metrics and balancing recommendations.
- `smartrain prune empty` removes empty image/label pairs into a new `<dataset>_pruned` dataset.
- `smartrain prune dedup` removes duplicate images by file content into `<dataset>_deduped` (global split priority: train > val > test).
- `smartrain report dataset` writes a multilingual per-class sample report (Markdown + PNG; default folder `analytics/datasets-reports/<dataset>_<timestamp>/`). Default dependencies include bundled pandoc (`pypandoc-binary`), WeasyPrint, `fpdf2`, and `odfpy` for PDF/ODT. WeasyPrint may need OS libraries (Cairo, Pango) if wheels are unavailable.
