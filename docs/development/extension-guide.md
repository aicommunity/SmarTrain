> Russian version: [../ru/development/extension-guide.md](../ru/development/extension-guide.md)

# Project extensibility

## Where to put code

- **CLI argparse / Typer glue / command scripts** → `smartrain/workflows/<domain>/` (or `smartrain/cli_entrypoints/` for thin wrappers such as `train_app.py`). Dataset catalog + interactive dataset pick: `workflows/datasets/dataset_cli_catalog.py` / `dataset_cli_common.py`.
- **Reusable logic without argparse** → `smartrain/services/` (must not import `workflows`; use `smartrain/core/workflow_adapters/`).
- **Workspace file contracts and run/model contract** → `smartrain/core/runtime/`, `smartrain/run_model_contract/` (`gateway.py`, `domain/`, `io/`).

Folder map: [package-layout.md](package-layout.md).

## Adding a new CLI command

1. Add routing to `smartrain/cli.py`.
2. Implement an argparse module with `build_*_arg_parser()` and `main(argv)`.
3. Update sections:
   - `docs/cli/overview.md`
   - a profile page in `docs/cli/`
   - `docs/development/architecture.md`, if needed.

### How Typer forwards to argparse

Most commands use `_forward_argparse_command` in `smartrain/cli.py` (search for its definition). Important parameters:

| Parameter | Role |
|-----------|------|
| `module` | Import path whose `main(argv)` runs the flow, e.g. `smartrain.workflows.datasets.datasets_entry`. |
| `build_parser` | Optional callable returning `ArgumentParser`; used with `ARGPARSE_HELP_EXAMPLES` when `prog` is set. |
| `prog` | Key into `ARGPARSE_HELP_EXAMPLES` for extra epilog text (must match the string passed to `_forward_argparse_command`). |
| `prepend_args` | Tokens injected before Typer `ctx.args` (subcommand names when the module expects them). |
| `empty_args_mode` | `"help"` \| `"invoke"` \| `"invoke_if_tty_else_help"` — controls behavior when the user passes no flags. |

Example (illustrative):

```python
@app.command("myfeature")
def cmd_myfeature(ctx: typer.Context) -> None:
    from smartrain.workflows.myfeature import myfeature_cli

    _forward_argparse_command(
        ctx,
        module="smartrain.workflows.myfeature.myfeature_cli",
        build_parser=myfeature_cli.build_arg_parser,
        prog="smartrain myfeature",
        prepend_args=[],
        empty_args_mode="invoke_if_tty_else_help",
    )
```

Target module shape:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(...)
    ...
    return p

def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ...
```

### PR checklist for a new command

- [ ] Registered in `cli.py` (Typer or `_forward_argparse_command`).
- [ ] `docs/cli/overview.md` and the relevant `docs/cli/*.md` page updated.
- [ ] `smartrain <cmd> --help` matches examples in docs.
- [ ] At least one test under `tests/` (often mirror `tests/test_cli_replay.py` or an existing command test).

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
