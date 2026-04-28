> Russian version: [../ru/providers/overview.md](../ru/providers/overview.md)

# Providers overview

External providers are fork-specific training/inference subsystems integrated through a common adapter/runner layer.

## Supported providers

- `dr-yolo`
- `leaf-yolo`
- `mfel-yolo`
- `mp-yolo`
- `ssdm-yolo`
- `enhanced-yolov8`

## Integration layers

- Registry: `smartrain/external_providers/registry.py`
  - declares provider id, repo URL/branch, and nominal entrypoints.
- Installer: `smartrain/external_providers/installer.py`
  - clones provider repo, prepares per-provider `venv`, installs runtime deps, updates global index.
- Global index: `smartrain/provider_global_index.py`
  - stores installed provider locations (`repo_path`, `venv_path`, state, diagnostics).
- Adapters: `smartrain/external_providers/adapters.py`
  - maps normalized Smart Train arguments to provider launcher arguments.
- Runners: `smartrain/external_providers/runner.py`
  - executes launchers in provider `venv`.
- Launchers: `smartrain/external_providers/launchers/*.py`
  - provider-facing wrappers around provider/fork runtime APIs.

## Runtime contract

For every provider run, Smart Train normalizes artifacts to:

- `<run_dir_name>.pt` in run root
- `test/` directory
- `test_metrics.csv`
- `training_metadata.json`

This contract keeps `analyze`, `registry`, and downstream automation provider-agnostic.
