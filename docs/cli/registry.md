> Russian version: [../ru/cli/registry.md](../ru/cli/registry.md)

# CLI: registry

`smartrain registry` manages run artifacts and the `models/` catalog in the workspace.

## Subcommands

- `runs-list`
- `runs-info`
- `runs-metrics`
- `models-add`
- `models-list`
- `models-info`
- `models-remove`

## `models-add`

Promotes a run into `models/<friendly_name>/` and writes `model_manifest.json`.

Copied into the bundle (when present on the run):

- `models/` — all exported weights and sidecars (same layout as under the run).
- `train/` and every `train-*/` directory (for example `train-ultralytics/`), excluding `weights/` checkpoints.
- Legacy `test/` at the run root, plus the full `tests/` tree (canonical test layout, metrics, manifests).
- `training_metadata.json` (paths to the primary `.pt` are adjusted to bundle-relative form), `test_metrics*.csv` from the run root, and `_runtime_data_*.yaml` from the run root or `tmp/`.

`weights_file` in the manifest is the path relative to the bundle directory (typically `models/<run_dir_name>.pt`). Older promoted trees that used `<friendly_name>.pt` at the bundle root are unchanged.

## Examples

```bash
smartrain registry runs-list
smartrain registry runs-info 3
smartrain registry models-add 3
smartrain registry models-list
```
