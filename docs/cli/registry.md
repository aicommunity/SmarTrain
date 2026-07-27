> Russian version: [../ru/cli/registry.md](../ru/cli/registry.md)

# CLI: registry

`smartrain registry` manages **run inventory** and **registry-promoted model bundles**.

This is **not** the same as `smartrain model release` / `model comment` (those publish into the release catalog under `models/<dataset>/<run_id>/` with `detect_*` stems and `releases_manifest.json`). See [`overview.md`](overview.md) and [`../refactor/run-layout.md`](../refactor/run-layout.md).

## Subcommands

- `runs-list`
- `runs-info`
- `runs-metrics`
- `models-add`
- `models-list`
- `models-info`
- `models-remove`

## `models-add`

Promotes a run into a **registry bundle** `models/<friendly_name>/` and writes `model_manifest.json`.

Copied into the bundle (when present on the run):

- `models/` — all exported weights and sidecars (same layout as under the run).
- `train/` and every `train-*/` directory (for example `train-ultralytics/`), excluding `weights/` checkpoints.
- Legacy `test/` at the run root, plus the full `tests/` tree (canonical test layout, metrics, manifests).
- `training_metadata.json` (paths to the primary `.pt` are adjusted to bundle-relative form), `test_metrics*.csv` from the run root, and `_runtime_data_*.yaml` from the run root or `tmp/`.

`weights_file` in the manifest is the path relative to the bundle directory (typically `models/<stem>.pt` with detect_* or legacy run-folder stem). Older promoted trees that used `<friendly_name>.pt` at the bundle root are unchanged.

## Examples

```bash
smartrain registry runs-list
smartrain registry runs-info 3
smartrain registry models-add 3
smartrain registry models-list
```
