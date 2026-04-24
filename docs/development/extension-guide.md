> Russian version: [../ru/development/extension-guide.md](../ru/development/extension-guide.md)

# Project extensibility

## Adding a new CLI command

1. Add routing to `smartrain/cli.py`.
2. Implement an argparse module with `build_*_arg_parser()` and `main(argv)`.
3. Update sections:
   - `docs/cli/overview.md`
   - a profile page in `docs/cli/`
   - `docs/development/architecture.md`, if needed.

## Changing data contracts

When `datasets_info.json`, `training_metadata.json`, or `model_manifest.json` change, also update:

- `docs/reference/data-formats.md`
- `docs/reference/training-metadata.md`
- the contracts diagram in `docs/development/architecture.md`.

## Adding or changing external providers

Use the dedicated provider engineering guide:

- `docs/development/provider-development.md`

When provider behavior changes, also update:

- `docs/cli/providers.md`
- `docs/providers/overview.md`
- provider profile pages in `docs/providers/`
