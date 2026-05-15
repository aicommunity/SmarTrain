# CLI replay and non-interactive mode

> Russian summary: see [../ru/development/architecture.md](../ru/development/architecture.md) (section «CLI: интерактив и replay»).

## Typer vs argparse

`smartrain` is a Typer entry point. Each subcommand forwards remaining tokens to an argparse `main(argv)` in a workflow module. Two layers apply:

1. **Typer** (`smartrain/cli.py`, `_forward_argparse_command`) — strips meta flags, sets `SMART_TRAIN_INTERACTIVE_ALLOWED`, invokes `main(filtered_argv)`.
2. **Argparse** — parses subcommand flags; some commands also define `-y` / `--non-interactive` (e.g. `test`).

## Meta flags (Typer only)

| Flag | Role |
|------|------|
| `--nit` | Canonical non-interactive meta flag (user-facing and replay suffix). |
| `--smartrain-replay` | Long synonym; stripped like `--nit`. |

These tokens are removed before `module.main` and never reach argparse parsers. Implementation: `smartrain/cli_support/typer_non_interactive.py`.

**Canonical form:** pass `--nit` as a separate argv token (`store_true` style). Do not rely on `--nit=1` or `--nit=true` in scripts; Typer may forward them as a single token. The stripper still removes `--nit=…` and `--smartrain-replay=…` so Typer routing treats the session as non-interactive, but argparse-only entry points (`python -m …`) may still see unknown options if you pass `=`-forms.

## Legacy and environment

| Mechanism | Stripped from forwarded argv? | Effect on Typer interactivity |
|-----------|--------------------------------|------------------------------|
| `-y`, `--non-interactive` on forwarded argv | No (passed to argparse where defined) | Forces non-interactive when present in raw argv before strip |
| `SMART_TRAIN_FORCE_NON_INTERACTIVE=1` | N/A | Forces non-interactive without argv flags |
| `SMART_TRAIN_INTERACTIVE_ALLOWED` | N/A | Set by Typer; modules use `is_interactive_allowed()` |

## Mode table

| Mode | TTY | `--nit` | Incomplete required args | Behavior |
|------|-----|---------|--------------------------|----------|
| Manual | yes | no | yes / no | Module prompts may run when args are incomplete; otherwise argparse errors |
| Manual | yes | yes | no | No Typer-driven interactivity; modules behave as in non-interactive mode |
| Manual | yes | yes | yes | Error (`parser.error` / explicit message), no interactive fill-in |
| Replay (paste) | yes | yes (in the printed line) | — | Predictable non-interactive run |

## Replay strings

`build_non_interactive_command` / `emit_replay` (`smartrain/cli_support/cli_replay.py`, `cli_contracts.py`) serialize a resolved `argparse.Namespace` and append a single trailing `--nit` when missing. Always paste the full printed line so Typer strips `--nit` and disables prompts.

### Legacy replay (removed orchestrator heuristic)

Previously, `smartrain test` replay could append only `--non-interactive`, and the orchestrator skipped backend prompts when `--formats` appeared in `request.argv` even without Typer non-interactive mode (H2). That argv heuristic was removed. Correct behavior now:

- **Replay** — always ends with `--nit` (Typer non-interactive).
- **Subcommand** — `test` may still use `-y` / `--non-interactive` / `--nit` on the forwarded argv for argparse and `args.non_interactive`.

A line with `--formats` but **without** trailing `--nit` is not a supported replay contract; regenerate via the tool’s printed replay.

## `train` vs Typer `--nit`

`smartrain train` uses `--yes` / `-y` for workspace and output confirmations inside the train argparse parser. That is separate from Typer `--nit`, which applies to the outer `smartrain` invocation. Replay lines for train include trailing `--nit` for Typer; inner train flags stay as serialized from the namespace.

## Unsupported entry points

Invoking a workflow with `python -m smartrain.workflows…` without the Typer wrapper does not strip `--nit`. Prefer `smartrain …` or set `SMART_TRAIN_FORCE_NON_INTERACTIVE`. See [../refactor/tech-debt-cli-replay-nit.md](../refactor/tech-debt-cli-replay-nit.md).
