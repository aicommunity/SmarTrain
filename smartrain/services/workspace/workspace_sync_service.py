from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_prefilled_text, prompt_yes_no
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.core.runtime.workspace_path_repair import repair_workspace_paths


@dataclass
class SyncStats:
    copied: int = 0
    skipped_exists: int = 0
    skipped_conflict: int = 0
    errors: int = 0


def build_sync_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Safely synchronize missing artifacts from another workspace copy.")
    p.add_argument("--workspace", type=str, default=None, help=f"Target workspace root (otherwise {WORKSPACE_ENV_VAR})")
    p.add_argument("--source", type=str, default=None, help="Source workspace to pull missing artifacts from.")
    p.add_argument("--dry-run", action="store_true", help="Show planned operations without writing.")
    p.add_argument("--yes", "-y", "--non-interactive", "--nit", action="store_true", dest="non_interactive")
    p.add_argument("--compat-threshold", type=float, default=0.3, help="Minimal common datasets ratio to treat workspaces as related.")
    return p


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_tree_or_file(src: str, dst: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=False)
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _datasets_related(dst_info: dict[str, Any], src_info: dict[str, Any], threshold: float) -> tuple[bool, set[str]]:
    dst_keys = {k for k, v in dst_info.items() if isinstance(v, dict)}
    src_keys = {k for k, v in src_info.items() if isinstance(v, dict)}
    common = dst_keys & src_keys
    if common:
        return True, common
    denom = max(len(dst_keys), len(src_keys), 1)
    ratio = len(common) / float(denom)
    return ratio >= float(threshold), common


def _dataset_hash(entry: dict[str, Any]) -> str | None:
    raw = entry.get("dataset_hash")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _copy_missing_dir_children(src_root: str, dst_root: str, stats: SyncStats, *, dry_run: bool) -> None:
    if not os.path.isdir(src_root):
        return
    os.makedirs(dst_root, exist_ok=True)
    for name in sorted(os.listdir(src_root)):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        if os.path.exists(dst):
            stats.skipped_exists += 1
            continue
        try:
            _copy_tree_or_file(src, dst, dry_run=dry_run)
            stats.copied += 1
        except Exception:
            stats.errors += 1


def _merge_datasets_list(src_layout: WorkspaceLayout, dst_layout: WorkspaceLayout, *, dry_run: bool) -> None:
    src_list = Path(src_layout.raw_data) / "datasets_list.txt"
    dst_list = Path(dst_layout.raw_data) / "datasets_list.txt"
    if not src_list.is_file():
        return
    src_lines = src_list.read_text(encoding="utf-8").splitlines()
    dst_lines = dst_list.read_text(encoding="utf-8").splitlines() if dst_list.is_file() else []
    seen = {line.strip() for line in dst_lines if line.strip()}
    out = list(dst_lines)
    for line in src_lines:
        s = line.strip()
        if not s or s.startswith("#") or s in seen:
            continue
        out.append(s)
        seen.add(s)
    if not dry_run:
        dst_list.parent.mkdir(parents=True, exist_ok=True)
        dst_list.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def _sync_datasets(
    src_layout: WorkspaceLayout,
    dst_layout: WorkspaceLayout,
    src_info: dict[str, Any],
    dst_info: dict[str, Any],
    stats: SyncStats,
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged_info = dict(dst_info)
    src_class_names = _read_json(src_layout.work_class_names_path())
    dst_class_names = _read_json(dst_layout.work_class_names_path())
    merged_class_names = dict(dst_class_names)
    for key, src_entry_raw in src_info.items():
        if not isinstance(src_entry_raw, dict):
            continue
        src_entry = dict(src_entry_raw)
        src_ds_dir = os.path.join(src_layout.datasets, key)
        dst_ds_dir = os.path.join(dst_layout.datasets, key)
        if key in merged_info:
            dst_hash = _dataset_hash(merged_info.get(key, {}))
            src_hash = _dataset_hash(src_entry)
            if dst_hash and src_hash and dst_hash != src_hash:
                stats.skipped_conflict += 1
            else:
                stats.skipped_exists += 1
            continue
        if os.path.isdir(src_ds_dir):
            try:
                _copy_tree_or_file(src_ds_dir, dst_ds_dir, dry_run=dry_run)
                stats.copied += 1
            except Exception:
                stats.errors += 1
                continue
        merged_info[key] = src_entry
        classes = src_entry.get("classes")
        if isinstance(classes, dict):
            for class_name in classes.keys():
                n = str(class_name)
                merged_class_names[n] = n
        if key in src_class_names:
            merged_class_names[str(key)] = str(src_class_names[key])
    return merged_info, merged_class_names


def _sync_runs_and_models(src_layout: WorkspaceLayout, dst_layout: WorkspaceLayout, stats: SyncStats, *, dry_run: bool) -> None:
    _copy_missing_dir_children(src_layout.runs, dst_layout.runs, stats, dry_run=dry_run)
    _copy_missing_dir_children(src_layout.models, dst_layout.models, stats, dry_run=dry_run)


def _print_summary(stats: SyncStats, *, dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}[OK] Sync summary:")
    print(f"{prefix}  copied: {stats.copied}")
    print(f"{prefix}  skipped_exists: {stats.skipped_exists}")
    print(f"{prefix}  skipped_conflict: {stats.skipped_conflict}")
    print(f"{prefix}  errors: {stats.errors}")


def main(argv: list[str] | None = None) -> None:
    parser = build_sync_arg_parser()
    args = parser.parse_args(argv)
    target_root = resolve_workspace_root(args.workspace)
    source_root = str(args.source or "").strip()
    if not source_root:
        if args.non_interactive:
            parser.error("Specify --source in non-interactive mode.")
        source_root = prompt_prefilled_text("Source workspace path", default=target_root).strip()
    if not source_root:
        parser.error("Empty source workspace path.")
    source_root = os.path.abspath(os.path.expanduser(source_root))
    if os.path.abspath(target_root) == source_root:
        parser.error("Source and target workspace are the same.")
    if not os.path.isdir(source_root):
        parser.error(f"Source workspace does not exist: {source_root}")

    src_layout = WorkspaceLayout(source_root)
    dst_layout = WorkspaceLayout(target_root)
    src_info = _read_json(src_layout.work_datasets_info_path())
    dst_info = _read_json(dst_layout.work_datasets_info_path())
    related, common = _datasets_related(dst_info, src_info, float(args.compat_threshold))
    if not related:
        parser.error(
            "Workspaces do not look related enough by dataset catalog intersection. "
            "Use a different source or lower --compat-threshold deliberately."
        )

    if not args.non_interactive and not args.dry_run:
        proceed = prompt_yes_no(
            f"Sync missing artifacts from '{source_root}' into '{target_root}'? common_datasets={len(common)}",
            default=False,
        )
        if not proceed:
            print("[INFO] Sync cancelled by user.")
            return

    stats = SyncStats()
    _copy_missing_dir_children(src_layout.raw_data, dst_layout.raw_data, stats, dry_run=bool(args.dry_run))
    merged_info, merged_class_names = _sync_datasets(
        src_layout,
        dst_layout,
        src_info,
        dst_info,
        stats,
        dry_run=bool(args.dry_run),
    )
    _sync_runs_and_models(src_layout, dst_layout, stats, dry_run=bool(args.dry_run))
    _merge_datasets_list(src_layout, dst_layout, dry_run=bool(args.dry_run))
    _write_json(dst_layout.work_datasets_info_path(), merged_info, dry_run=bool(args.dry_run))
    _write_json(dst_layout.work_class_names_path(), merged_class_names, dry_run=bool(args.dry_run))

    repair_workspace_paths(target_root, dry_run=bool(args.dry_run), include_datasets_list=True)
    if not args.dry_run:
        from smartrain.services.datasets.datasets_json_former import main as run_scan

        run_scan(["--workspace", target_root])
    _print_summary(stats, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
