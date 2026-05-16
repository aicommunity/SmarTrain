from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import prompt_choice, prompt_text
from smartrain.cli_support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.workflows.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    try_prompt_dataset_interactive,
)
from smartrain.workflows.datasets.dataset_former import _image_content_hash
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.workflows.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_PRIORITY = {"train": 0, "val": 1, "valid": 1, "test": 2}


@dataclass
class _ImageItem:
    split: str
    split_rank: int
    image_path: str
    label_path: str


def build_prune_empty_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Prune empty image/label pairs into a new dataset")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_pruned)")
    p.add_argument("--dry-run", action="store_true")
    return p


def build_prune_dedup_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Prune duplicated images by content into a new dataset")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_deduped)")
    p.add_argument("--allow-balanced-dedup", action="store_true", help="Allow dedup on datasets produced by balance")
    p.add_argument("--dry-run", action="store_true")
    return p


def build_prune_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Dataset pruning utilities: remove empty pairs or duplicates")
    sub = p.add_subparsers(dest="mode")
    sub.required = False
    sub.add_parser("empty", parents=[build_prune_empty_arg_parser()], add_help=False)
    sub.add_parser("dedup", parents=[build_prune_dedup_arg_parser()], add_help=False)
    return p


def _label_has_valid_yolo_line(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    int(float(parts[0]))
                except ValueError:
                    continue
                return True
    except OSError:
        return False
    return False


def _list_output_items(out_dir: str) -> list[_ImageItem]:
    items: list[_ImageItem] = []
    root = Path(out_dir)
    for split in ("train", "val", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.is_dir():
            continue
        for file in sorted(img_dir.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = file.relative_to(img_dir)
            lbl = lbl_dir / rel.parent / f"{file.stem}.txt"
            items.append(
                _ImageItem(
                    split=split,
                    split_rank=SPLIT_PRIORITY.get(split, 99),
                    image_path=str(file),
                    label_path=str(lbl),
                )
            )
    return items


def _copy_source_dataset(src_root: str, entry: dict[str, Any], out_dir: str, dataset_name: str, tmp_root: str) -> int:
    structure = str(entry.get("structure", "split"))
    buckets = iter_image_label_buckets(
        src_root,
        structure,
        entry,
        dataset_name=dataset_name,
        temp_root=tmp_root,
        exclude_test=False,
    )
    copied = 0
    for img_dir, lbl_dir in buckets:
        rel_img = os.path.relpath(img_dir, src_root)
        rel_lbl = os.path.relpath(lbl_dir, src_root)
        out_img_dir = os.path.join(out_dir, rel_img)
        out_lbl_dir = os.path.join(out_dir, rel_lbl)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        for name in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            src_img = os.path.join(img_dir, name)
            stem = os.path.splitext(name)[0]
            src_lbl = os.path.join(lbl_dir, f"{stem}.txt")
            dst_img = os.path.join(out_img_dir, name)
            dst_lbl = os.path.join(out_lbl_dir, f"{stem}.txt")
            shutil.copy2(src_img, dst_img)
            if os.path.isfile(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            copied += 1
    return copied


def _write_data_yaml(out_dir: str, class_map: dict[str, Any]) -> None:
    names = [k for k, _ in sorted(((str(k), int(v)) for k, v in class_map.items()), key=lambda kv: kv[1])]
    val_rel = "valid/images" if (Path(out_dir) / "valid" / "images").is_dir() else "val/images"
    Path(out_dir, "data.yaml").write_text(
        f"train: train/images\nval: {val_rel}\ntest: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def _update_datasets_sidecar(layout: WorkspaceLayout, output_key: str, class_map: dict[str, Any], target_dir: str, output_hash: str) -> None:
    info_path = layout.work_datasets_info_path()
    info: dict[str, Any] = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            info = loaded
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    info[output_key] = {
        "classes": {str(k): int(v) for k, v in sorted(((str(k), int(v)) for k, v in class_map.items()), key=lambda kv: kv[1])},
        "structure": "split",
        "elements_count": None,
        "data_path": rel,
        "dataset_hash": output_hash,
        "modified": False,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def _is_balanced_source(dataset_root: str) -> bool:
    passport_path = os.path.join(dataset_root, "dataset_passport.json")
    if os.path.isfile(passport_path):
        try:
            with open(passport_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and str(payload.get("command", "")).strip().lower() == "balance":
                return True
        except Exception:
            pass
    if os.path.isfile(os.path.join(dataset_root, "balance_manifest.json")):
        return True
    # Fallback heuristic for old balanced outputs.
    stems_total = 0
    stems_bal = 0
    pat = re.compile(r"bal_\d+$")
    for split in ("train", "val", "valid", "test"):
        p = Path(dataset_root) / split / "images"
        if not p.is_dir():
            continue
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                stems_total += 1
                if pat.search(f.stem):
                    stems_bal += 1
    return stems_total > 0 and (stems_bal / float(stems_total)) >= 0.6


def _prune_empty(out_dir: str) -> dict[str, int]:
    removed = 0
    scanned = 0
    for item in _list_output_items(out_dir):
        scanned += 1
        if _label_has_valid_yolo_line(item.label_path):
            continue
        try:
            os.remove(item.image_path)
        except OSError:
            pass
        if os.path.isfile(item.label_path):
            try:
                os.remove(item.label_path)
            except OSError:
                pass
        removed += 1
    return {"scanned": scanned, "removed": removed}


def _prune_dedup(out_dir: str) -> dict[str, int]:
    removed = 0
    scanned = 0
    cross_split_removed = 0
    groups: dict[str, list[_ImageItem]] = {}
    for item in _list_output_items(out_dir):
        scanned += 1
        h = _image_content_hash(item.image_path)
        group = groups.setdefault(h, [])
        exact_match = None
        for g in group:
            if os.path.getsize(g.image_path) != os.path.getsize(item.image_path):
                continue
            if Path(g.image_path).read_bytes() == Path(item.image_path).read_bytes():
                exact_match = g
                break
        if exact_match is None:
            group.append(item)
            continue
        keep = exact_match
        drop = item
        if (item.split_rank, item.image_path) < (exact_match.split_rank, exact_match.image_path):
            keep = item
            drop = exact_match
            group.remove(exact_match)
            group.append(item)
        if keep.split != drop.split:
            cross_split_removed += 1
        try:
            os.remove(drop.image_path)
        except OSError:
            pass
        if os.path.isfile(drop.label_path):
            try:
                os.remove(drop.label_path)
            except OSError:
                pass
        removed += 1
    return {"scanned": scanned, "removed": removed, "cross_split_removed": cross_split_removed}


def _interactive_fill(args: argparse.Namespace, mode: str, dataset_names: list[str]) -> None:
    print("[INFO] Interactive prune mode")
    args.dataset = prompt_choice("Dataset", dataset_names, default=(args.dataset or dataset_names[0]))
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")).strip() or None
    if mode == "dedup":
        # Keep strict safeguard by default.
        args.allow_balanced_dedup = False


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv and argv[0] in {"empty", "dedup"} else None
    mode_args = argv[1:] if mode else argv
    interactive_allowed = is_interactive_allowed(argv)

    if mode == "dedup":
        parser = build_prune_dedup_arg_parser()
    else:
        parser = build_prune_empty_arg_parser()
    args = parser.parse_args(mode_args)

    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = load_datasets_catalog(layout)
    if not catalog:
        print(EMPTY_DATASETS_INFO_MESSAGE)
        return

    interactive_used = False
    if mode is None:
        if not interactive_allowed:
            print("[ERROR] Incomplete arguments: specify prune mode (empty|dedup).")
            return
        if not sys.stdin.isatty():
            print("[ERROR] Interactive prune mode requires a terminal (TTY).")
            return
        mode = prompt_choice("Prune mode", ["empty", "dedup"], default="empty")
        parser = build_prune_dedup_arg_parser() if mode == "dedup" else build_prune_empty_arg_parser()
        args = parser.parse_args([])
        _interactive_fill(args, mode, sorted(catalog.keys()))
        interactive_used = True
    elif try_prompt_dataset_interactive(
        args=args,
        argv=argv,
        fill=lambda: _interactive_fill(args, mode, sorted(catalog.keys())),
    ):
        interactive_used = True

    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return

    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command(f"prune {mode}", parser, args)
        print_replay_command("before launch", replay_cmd)

    entry = catalog[args.dataset]
    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    if mode == "dedup" and not getattr(args, "allow_balanced_dedup", False) and _is_balanced_source(src_root):
        print(
            "[ERROR] Source dataset looks like output of `balance`. "
            "Refusing dedup by default. Use --allow-balanced-dedup to continue."
        )
        return

    out_base = args.output_name or f"{args.dataset}_{'pruned' if mode == 'empty' else 'deduped'}"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    if args.dry_run:
        print(f"[OK] dry-run: mode={mode}, dataset={args.dataset}, output={out_name}")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    copied = _copy_source_dataset(
        src_root,
        entry if isinstance(entry, dict) else {},
        out_dir,
        dataset_name=args.dataset,
        tmp_root=os.path.join(layout.root, "tmp"),
    )
    if mode == "empty":
        stats = _prune_empty(out_dir)
    else:
        stats = _prune_dedup(out_dir)

    class_map = entry.get("classes", {}) if isinstance(entry, dict) else {}
    if isinstance(class_map, dict):
        _write_data_yaml(out_dir, class_map)
    out_hash = calculate_dataset_hash(out_dir)
    if isinstance(class_map, dict):
        _update_datasets_sidecar(layout, out_name, class_map, out_dir, out_hash)

    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command=f"prune-{mode}",
        source_datasets=[{"name": args.dataset, "path": src_root, "dataset_hash": entry.get("dataset_hash")}],
        parameters=vars(args),
        workspace_root=layout.root,
        transformations=[{"mode": mode}],
        stats_before={"copied_images": copied},
        stats_after=stats | {"output_hash": out_hash},
        random_seed=None,
    )
    print(f"[OK] Dataset created: {out_dir}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)

