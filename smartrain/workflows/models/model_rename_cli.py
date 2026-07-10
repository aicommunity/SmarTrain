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
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.models.release_model_rename_service import (
    ReleaseModelEntry,
    ReleaseRenameError,
    apply_release_rename,
    build_rename_plan,
    discover_release_models,
)


def build_model_rename_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Rename a released workspace model and related artifacts (empty call starts interactive mode)"
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
        "--new-name",
        type=str,
        default=None,
        help="New release stem (filename without extension)",
    )
    return p


def _resolve_release_ref(layout: WorkspaceLayout, ref: str, entries: list[ReleaseModelEntry]) -> ReleaseModelEntry:
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


def _pick_release_interactive(entries: list[ReleaseModelEntry]) -> ReleaseModelEntry:
    if not entries:
        raise RuntimeError("no released models found in workspace models catalog")
    printable = [entry.rel_path for entry in entries]
    options = [str(entry.pt_path) for entry in entries]
    print_numbered_options("released models", printable)
    picked = prompt_choice("Select release model", options, default=options[0], show_options=False)
    idx = options.index(picked)
    return entries[idx]


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_model_rename_arg_parser()
    args = parser.parse_args(argv)

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(layout.models, exist_ok=True)

    entries = discover_release_models(layout)
    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False

    entry: ReleaseModelEntry
    new_name: str

    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        entry = _pick_release_interactive(entries)
        new_name = prompt_prefilled_text("New release name", default=entry.stem)
        interactive_used = True
    else:
        if not args.release or not args.new_name:
            parser.error(
                "incomplete arguments: use --release and --new-name "
                "(or run command without arguments for interactive mode)."
            )
        try:
            entry = _resolve_release_ref(layout, str(args.release), entries)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            raise SystemExit(1) from e
        new_name = str(args.new_name)

    try:
        plan = build_rename_plan(entry, new_name)
    except ReleaseRenameError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if plan.old_stem == plan.new_stem:
        print(f"[OK] Nothing to do: release name unchanged ({plan.old_stem})")
        raise SystemExit(0)

    if interactive_used:
        print(f"[INFO] Current: {entry.rel_path}")
        print(f"[INFO] New stem: {plan.new_stem}")
        print(f"[INFO] Files to rename: {len(plan.operations)}")
        if not prompt_yes_no("Proceed with rename?", default=True):
            print("[INFO] Rename cancelled by user.")
            raise SystemExit(0)

    try:
        result = apply_release_rename(plan)
    except ReleaseRenameError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except OSError as e:
        print(f"[ERROR] Failed to rename files: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if result.skipped:
        print(f"[OK] Nothing to do: {result.reason}")
    else:
        print(f"[OK] Renamed release model: {result.old_stem} -> {result.new_stem}")
        print(f"[OK] Primary model: {plan.new_pt_path}")
        print(f"[OK] Renamed artifacts: {len(result.renamed_paths)}")

    if interactive_used:
        args.release = str(entry.pt_path)
        args.new_name = plan.new_stem
        replay_cmd = build_non_interactive_command("model rename", parser, args)
        print_replay_command("model rename", replay_cmd)


if __name__ == "__main__":
    main()
