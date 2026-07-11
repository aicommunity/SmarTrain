"""Argparse forwarding helpers for the Typer CLI router."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from contextlib import contextmanager
from typing import Callable

import typer

from smartrain.cli_entrypoints.help_texts import ARGPARSE_HELP_EXAMPLES
from smartrain.cli_entrypoints.support.typer_non_interactive import (
    env_forces_non_interactive_cli,
    strip_typer_meta_non_interactive_flags,
)
from smartrain.core.runtime.interactive_contract import INTERACTIVE_ALLOWED_ENV
from smartrain.core.runtime.workspace_coordination import (
    WorkspaceLockBusy,
    WorkspaceSession,
    classify_command,
    get_active_session,
    try_resolve_layout_from_argv,
)


def _format_columns(items: tuple[str, ...], *, max_columns: int = 4) -> list[str]:
    if not items:
        return []
    width = shutil.get_terminal_size(fallback=(100, 20)).columns
    col_width = max(len(x) for x in items) + 2
    if col_width <= 0:
        return list(items)
    cols = max(1, min(max_columns, width // col_width))
    if cols <= 1:
        return list(items)
    rows_count = (len(items) + cols - 1) // cols
    lines: list[str] = []
    for row in range(rows_count):
        parts: list[str] = []
        for col in range(cols):
            idx = col * rows_count + row
            if idx >= len(items):
                continue
            cell = items[idx]
            if col < cols - 1:
                parts.append(cell.ljust(col_width))
            else:
                parts.append(cell)
        lines.append("".join(parts).rstrip())
    return lines


def _is_detection_model_alias(alias: str) -> bool:
    lowered = alias.lower()
    non_detection_markers = ("-seg", "-cls", "-pose", "-obb")
    return not any(marker in lowered for marker in non_detection_markers)


def _invoke_module_main(module: str, args: list[str]) -> None:
    m = importlib.import_module(module)
    fn = getattr(m, "main")
    fn(args)


def _strip_coordination_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--no-peer-warn", "--force-resource-lock"):
            continue
        if tok in ("--wait-for-scan", "--catalog-lock-timeout"):
            skip_next = True
            continue
        if tok.startswith("--wait-for-scan=") or tok.startswith("--catalog-lock-timeout="):
            continue
        out.append(tok)
    return out


def _command_label(prog: str | None, filtered: list[str]) -> list[str]:
    if prog:
        parts = prog.split()
        if parts and parts[0] == "smartrain" and len(parts) > 1:
            return parts[1:] + filtered
        return parts + filtered
    return filtered


@contextmanager
def _interactive_flag_env(allowed: bool):
    prev = os.environ.get(INTERACTIVE_ALLOWED_ENV)
    os.environ[INTERACTIVE_ALLOWED_ENV] = "1" if allowed else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(INTERACTIVE_ALLOWED_ENV, None)
        else:
            os.environ[INTERACTIVE_ALLOWED_ENV] = prev


def _run_with_coordination(
    module: str,
    filtered: list[str],
    *,
    prog: str | None,
    ensure_scan: bool,
    auto_scan_disabled: bool,
    interactive_allowed: bool,
) -> None:
    from smartrain.services.datasets.dataset_scan_preflight import maybe_run_auto_scan

    layout = try_resolve_layout_from_argv(filtered)
    clean = _strip_coordination_flags(filtered)
    if layout is None:
        maybe_run_auto_scan(filtered, ensure_scan=ensure_scan, auto_scan_disabled=auto_scan_disabled)
        with _interactive_flag_env(interactive_allowed):
            _invoke_module_main(module, clean)
        return

    cmd_argv = _command_label(prog, filtered)
    with WorkspaceSession(layout, cmd_argv):
        maybe_run_auto_scan(filtered, ensure_scan=ensure_scan, auto_scan_disabled=auto_scan_disabled)
        policy = classify_command(filtered, prog=(prog or "").split()[-1] if prog else None)
        session = get_active_session()
        if session is None:
            raise RuntimeError("WorkspaceSession active but get_active_session() returned None")
        try:
            with policy.locks(session):
                with _interactive_flag_env(interactive_allowed):
                    _invoke_module_main(module, clean)
        except WorkspaceLockBusy as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise typer.Exit(1) from exc


def _forward_argparse_command(
    ctx: typer.Context,
    *,
    module: str,
    build_parser: Callable[[], object] | None = None,
    prog: str | None = None,
    prepend_args: list[str] | None = None,
    empty_args_mode: str = "help",
    ensure_scan: bool = False,
) -> None:
    raw = list(prepend_args or []) + list(ctx.args)
    filtered, meta_stripped = strip_typer_meta_non_interactive_flags(raw)
    auto_scan_disabled = any(
        tok == "--no-auto-scan" or tok.startswith("--no-auto-scan=") for tok in raw
    )
    legacy_ni = any(tok in raw for tok in ("-y", "--non-interactive"))
    if meta_stripped or legacy_ni or env_forces_non_interactive_cli():
        interactive_allowed = False
    elif len(filtered) == 0 and empty_args_mode in ("invoke", "invoke_if_tty_else_help"):
        interactive_allowed = True
    else:
        interactive_allowed = bool(sys.stdin.isatty())

    def _enhance_parser_help(parser_obj: object) -> None:
        if prog is None:
            return
        examples = ARGPARSE_HELP_EXAMPLES.get(prog)
        if not examples:
            return
        if hasattr(parser_obj, "epilog"):
            existing = getattr(parser_obj, "epilog", None)
            setattr(parser_obj, "epilog", f"{existing}\n\n{examples}" if existing else examples)

    if not filtered:
        if empty_args_mode == "invoke":
            _run_with_coordination(
                module,
                filtered,
                prog=prog,
                ensure_scan=ensure_scan,
                auto_scan_disabled=auto_scan_disabled,
                interactive_allowed=interactive_allowed,
            )
            return
        if empty_args_mode == "invoke_if_tty_else_help":
            if sys.stdin.isatty():
                _run_with_coordination(
                    module,
                    filtered,
                    prog=prog,
                    ensure_scan=ensure_scan,
                    auto_scan_disabled=auto_scan_disabled,
                    interactive_allowed=interactive_allowed,
                )
                return
        if build_parser:
            parser = build_parser()
            if prog is not None and hasattr(parser, "prog"):
                parser.prog = prog
            _enhance_parser_help(parser)
            if hasattr(parser, "print_help"):
                parser.print_help()
            raise typer.Exit(0)

    if build_parser and any(tok in ("--help", "-h") for tok in filtered):
        parser = build_parser()
        if prog is not None and hasattr(parser, "prog"):
            parser.prog = prog
        _enhance_parser_help(parser)
        try:
            parser.parse_args(filtered)
        except SystemExit as e:
            code = e.code
            if code is None:
                code = 0
            raise typer.Exit(code if isinstance(code, int) else 1)

    _run_with_coordination(
        module,
        filtered,
        prog=prog,
        ensure_scan=ensure_scan,
        auto_scan_disabled=auto_scan_disabled,
        interactive_allowed=interactive_allowed,
    )
