> Russian version: [../ru/development/provider-development.md](../ru/development/provider-development.md)

# Developing new external providers

This guide describes the current integration contract for adding a new external provider.

## 1. Add provider spec

File: `smartrain/external_providers/registry.py`

Add a new `ExternalProviderSpec` with:

- unique `id` (lowercase, kebab-case),
- display name,
- source repository URL/branch,
- nominal train/infer entrypoint names.

## 2. Implement installation and probing compatibility

Files:

- `smartrain/external_providers/installer.py`
- `smartrain/external_providers/probe.py`
- `smartrain/provider_global_index.py`

Requirements:

- installation must be idempotent;
- provider `venv` must be isolated;
- all required runtime deps should be installed in provider `venv`;
- index record must include valid `repo_path` and `venv_path`.

## 3. Adapter and launcher mapping

Files:

- `smartrain/external_providers/adapters.py`
- `smartrain/external_providers/launchers/*.py`
- `smartrain/external_providers/runner.py`

Rules:

- Adapter maps Smart Train args to provider launcher args.
- Launcher should run under provider `venv` and support deterministic CLI contract.
- If provider has custom blocks/imports, patch missing symbols inside launcher (provider-local, not global).

## 4. Provider model catalog and strict validation

File: `smartrain/train_model_catalog.py`

Add provider aliases to `_EXTERNAL_PROVIDER_FALLBACK_ALIASES` and optional dynamic discovery.

Validation path:

- `train`: `model_training_module.py` with `is_supported_external_provider_model(...)`.
- `inference`: `inference_cli.py` with the same strict provider-scoped validation.

If alias is not supported, command must fail with clear error and list of supported aliases.

## 5. Defaults for external providers

Current behavior when `--external-provider` is set and explicit args are missing:

- default model is selected from provider catalog;
- launcher defaults are applied for missing values: `epochs=70`, `batch=8`, `img_size=640`.

This logic is in `model_training_module.py` (`_apply_external_provider_defaults`).

## 6. Run naming and artifact normalization

Run name must be path-safe and stable:

- `YYYY-MM-DD_HH-MM_<provider>_<model>_<epochs>epochs_b<batch>-<dataset_hash>`

Model token must be sanitized from filenames/paths to avoid nested invalid directories.

Required normalized output contract per run:

- `<run_dir_name>.pt` in run root
- `test/`
- `test_metrics.csv`
- `training_metadata.json`

Normalization helpers are in `model_training_module.py`.

## 7. MFEL-style fallback pattern (custom provider compatibility)

When built-in `test_yolo` cannot load custom provider checkpoint:

- use provider-side validation launcher fallback (example: `mfel_val_launcher.py`);
- produce machine-readable test CSV (`results.csv` and run-root `test_metrics.csv`);
- keep `training_metadata.json` status aligned with actual fallback result.

## 8. Documentation updates required

For each new provider update:

- `docs/cli/providers.md`
- `docs/providers/overview.md`
- provider profile page `docs/providers/<provider-id>.md`
- Russian mirrors in `docs/ru/...`

Also update indexes:

- `docs/index.md`
- `docs/ru/index.md`

## 9. Minimum test checklist

- Adapter unit tests: `tests/test_external_providers_adapters.py`
- Runner unit tests: `tests/test_external_providers_runner.py`
- CLI behavior (`provider:model`, validation): `tests/test_train_interactive.py`, `tests/test_inference_cli.py`
- Launcher tests for provider specifics (example: MFEL): `tests/test_mfel_launchers.py`
- End-to-end smoke: one-epoch `train` for provider with metadata and artifact checks.
