from __future__ import annotations

import argparse
import os
import sys

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
from smartrain.services.datasets.dataset_cli_common import load_dataset_catalog
from smartrain.services.datasets.dataset_rename_service import (
    DatasetRenameError,
    apply_dataset_rename,
    build_rename_plan,
    format_plan_report,
)


def build_dataset_rename_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Rename a workspace dataset and update related references (empty call starts interactive mode)"
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help=f"Workspace root (otherwise {WORKSPACE_ENV_VAR})",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Current dataset key from datasets_info.json",
    )
    p.add_argument(
        "--new-name",
        type=str,
        default=None,
        help="New dataset name (catalog key and default directory name)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show rename plan without applying changes",
    )
    p.add_argument(
        "--move-data-path",
        action="store_true",
        help="Relocate dataset root when data_path points outside datasets/<name>/",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(argv or [])
    parser = build_dataset_rename_arg_parser()
    args = parser.parse_args(argv)

    workspace_root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(workspace_root)
    os.makedirs(layout.datasets, exist_ok=True)

    catalog = load_dataset_catalog(layout)
    dataset_names = sorted(catalog.keys())
    interactive_allowed = is_interactive_allowed(argv)
    interactive_used = False

    old_name: str
    new_name: str

    if interactive_allowed and len(argv) == 0 and sys.stdin.isatty():
        if not dataset_names:
            print("[ERROR] No datasets found in datasets_info.json", file=sys.stderr)
            raise SystemExit(1)
        print_numbered_options("datasets", dataset_names)
        old_name = prompt_choice("Dataset to rename", dataset_names, default=dataset_names[0], show_options=False)
        new_name = prompt_prefilled_text("New dataset name", default=old_name)
        interactive_used = True
    else:
        if not args.dataset or not args.new_name:
            parser.error(
                "incomplete arguments: use --dataset and --new-name "
                "(or run command without arguments for interactive mode)."
            )
        old_name = str(args.dataset).strip()
        new_name = str(args.new_name).strip()

    try:
        plan = build_rename_plan(
            layout,
            old_name,
            new_name,
            move_data_path=bool(args.move_data_path),
        )
    except DatasetRenameError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if plan.old_name == plan.new_name:
        print(f"[OK] Nothing to do: dataset name unchanged ({plan.old_name})")
        raise SystemExit(0)

    print(format_plan_report(plan))

    if interactive_used and not args.dry_run:
        if not prompt_yes_no("Proceed with rename?", default=True):
            print("[INFO] Rename cancelled by user.")
            raise SystemExit(0)

    try:
        result = apply_dataset_rename(plan, dry_run=bool(args.dry_run))
    except DatasetRenameError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except OSError as e:
        print(f"[ERROR] Failed to rename dataset: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    if args.dry_run:
        print("[OK] Dry-run complete; no changes applied.")
    elif result.skipped:
        print(f"[OK] Nothing to do: {result.reason}")
    else:
        print(f"[OK] Renamed dataset: {result.old_name} -> {result.new_name}")
        print(f"[OK] Renamed directories: {len(result.renamed_dirs)}")
        print(f"[OK] Updated files: {len(result.updated_files)}")

    if interactive_used:
        args.dataset = plan.old_name
        args.new_name = plan.new_name
        replay_cmd = build_non_interactive_command("dataset rename", parser, args)
        print_replay_command("dataset rename", replay_cmd)


if __name__ == "__main__":
    main()
