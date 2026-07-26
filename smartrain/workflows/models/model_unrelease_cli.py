from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    print_numbered_options,
    prompt_choice,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.models.release_model_naming import load_release_metadata
from smartrain.services.models.release_model_rename_service import (
    ReleaseModelEntry,
    discover_release_models,
)
from smartrain.services.models.release_models_manifest import (
    entry_key_for_pt,
    format_release_entry_label,
    remove_entry,
)


def build_model_unrelease_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Move a released model back into runs/ and remove it from the release catalog "
        "(empty call starts interactive mode)"
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
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for destructive non-interactive mode",
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
        if entry.release_dir == p:
            return entry
    raise ValueError(f"release model not found: {p}")


def _pick_release_interactive(entries: list[ReleaseModelEntry]) -> ReleaseModelEntry:
    if not entries:
        raise RuntimeError("no released models found in workspace models catalog")
    printable = [format_release_entry_label(entry.rel_path, entry.comment) for entry in entries]
    options = [str(entry.pt_path) for entry in entries]
    print_numbered_options("released models", printable)
    picked = prompt_choice("Select release model", options, default=options[0], show_options=False)
    idx = options.index(picked)
    return entries[idx]


def _runs_destination(layout: WorkspaceLayout, entry: ReleaseModelEntry) -> Path:
    payload = load_release_metadata(entry.release_json) or {}
    src = payload.get("source") or {}
    rel = str(src.get("source_run_relative") or "").strip()
    if rel.startswith("runs/"):
        return (Path(layout.root) / rel).resolve()
    try:
        models_root = Path(layout.models).resolve()
        rel_models = entry.release_dir.resolve().relative_to(models_root)
        if len(rel_models.parts) >= 2:
            return (Path(layout.runs) / rel_models.parts[0] / rel_models.parts[1]).resolve()
    except Exception:
        pass
    return (Path(layout.runs) / entry.release_dir.name).resolve()


def _strip_release_overlay_and_normalize_layout(
    release_dir: Path,
    pt_path: Path,
    release_json: Path,
) -> None:
    stem = pt_path.stem
    models_sub = release_dir / "models"
    models_sub.mkdir(parents=True, exist_ok=True)

    if release_json.is_file():
        release_json.unlink()

    if pt_path.parent.resolve() != release_dir.resolve():
        return

    dest_pt = models_sub / pt_path.name
    if not dest_pt.exists():
        shutil.move(str(pt_path), str(dest_pt))

    for child in list(release_dir.iterdir()):
        if not child.is_file():
            continue
        if child.name == f"{stem}.pt":
            continue
        if child.name.startswith(f"{stem}.") or child.name.startswith(f"{stem}_"):
            dest = models_sub / child.name
            if not dest.exists():
                shutil.move(str(child), str(dest))


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_model_unrelease_arg_parser()
    args = parser.parse_args(argv)

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(layout.models, exist_ok=True)
    os.makedirs(layout.runs, exist_ok=True)

    entries = discover_release_models(layout)
    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False

    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        entry = _pick_release_interactive(entries)
        interactive_used = True
    else:
        if not args.release:
            parser.error(
                "incomplete arguments: use --release "
                "(or run command without arguments for interactive mode)."
            )
        try:
            entry = _resolve_release_ref(layout, str(args.release), entries)
        except ValueError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            raise SystemExit(1) from e

    runs_dir = _runs_destination(layout, entry)
    if runs_dir.exists():
        print(
            f"[ERROR] Target run directory already exists: {runs_dir}. "
            "Remove or rename it before unrelease.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if interactive_used:
        print(f"[INFO] Release dir: {entry.release_dir}")
        print(f"[INFO] Target runs dir: {runs_dir}")
        if not prompt_yes_no("Proceed with unrelease?", default=False):
            print("[INFO] Unrelease cancelled by user.")
            raise SystemExit(0)
    elif not args.yes:
        print(
            "[ERROR] Non-interactive unrelease requires --yes to confirm.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    entry_key = entry_key_for_pt(entry.pt_path)
    _strip_release_overlay_and_normalize_layout(entry.release_dir, entry.pt_path, entry.release_json)
    remove_entry(layout, entry_key)

    runs_parent = runs_dir.parent
    runs_parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(entry.release_dir), str(runs_dir))
    except Exception as e:
        print(f"[ERROR] Failed to move release back to runs/: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    print(f"[OK] Unreleased model moved to: {runs_dir}")
    print(f"[OK] Removed manifest entry: {entry_key}")

    if interactive_used:
        args.release = str(entry.pt_path)
        replay_cmd = build_non_interactive_command("model unrelease", parser, args)
        print_replay_command("model unrelease", replay_cmd)


if __name__ == "__main__":
    main()
