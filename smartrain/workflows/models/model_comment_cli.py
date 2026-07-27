from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_numbered_options,
    prompt_choice,
    prompt_prefilled_text,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.models.release_model_rename_service import discover_release_models
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    format_release_entry_label,
    set_comment,
    sync_sidecar_comment,
)


def build_model_comment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Set or update a one-line comment for a released model (empty call starts interactive mode)"
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--release",
        type=str,
        default=None,
        help="Released .pt path or index from discovered release models list",
    )
    p.add_argument(
        "--comment",
        type=str,
        default=None,
        help="New one-line comment (may be empty)",
    )
    return p


def _resolve_release_ref(layout: WorkspaceLayout, ref: str, entries: list) -> object:
    from smartrain.services.models.release_model_rename_service import ReleaseModelEntry

    s = (ref or "").strip()
    if not s:
        raise ValueError("release reference is empty")
    if s.isdigit():
        idx = int(s)
        if idx < 1 or idx > len(entries):
            raise ValueError(f"release index {idx} is out of range 1..{len(entries)}")
        return entries[idx - 1]
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = (Path(layout.root) / p).resolve()
    else:
        p = p.resolve()
    for entry in entries:
        if entry.pt_path == p:
            return entry
    raise ValueError(f"release model not found: {p}")


def _pick_release_interactive(entries: list) -> object:
    if not entries:
        raise RuntimeError("no released models found in workspace models catalog")
    printable = [
        format_release_entry_label(entry.rel_path, entry.comment)
        for entry in entries
    ]
    options = [str(entry.pt_path) for entry in entries]
    print_numbered_options("released models", printable)
    picked = prompt_choice("Select release model", options, default=options[0], show_options=False)
    idx = options.index(picked)
    return entries[idx]


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_model_comment_arg_parser()
    args = parser.parse_args(argv)

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(layout.models, exist_ok=True)

    entries = discover_release_models(layout)
    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False

    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        entry = _pick_release_interactive(entries)
        old_comment = entry.comment
        new_comment = prompt_prefilled_text("Comment", default=old_comment)
        interactive_used = True
    else:
        if args.release is None or args.comment is None:
            parser.error(
                "incomplete arguments: use --release and --comment "
                "(or run command without arguments for interactive mode)."
            )
        try:
            entry = _resolve_release_ref(layout, str(args.release), entries)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            raise SystemExit(1) from e
        new_comment = str(args.comment)

    key = entry_key_for_pt(entry.pt_path)
    set_comment(layout, key, new_comment)
    sync_sidecar_comment(entry.release_json, new_comment)

    print(f"[OK] Updated comment for {entry.rel_path}")
    if new_comment.strip():
        print(f"[OK] Comment: {new_comment}")
    else:
        print("[OK] Comment cleared")

    if interactive_used:
        args.release = str(entry.pt_path)
        args.comment = new_comment
        replay_cmd = build_non_interactive_command("model comment", parser, args)
        print_replay_command("model comment", replay_cmd)


if __name__ == "__main__":
    main()
