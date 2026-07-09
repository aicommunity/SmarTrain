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

from PIL import Image
from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice, prompt_text
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.bbox_edge_filter import bbox_geom_from_label
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_cli_common import update_datasets_sidecar
from smartrain.services.datasets.dataset_class_cleanup import strip_unused_classes
from smartrain.services.datasets.dataset_former import _image_content_hash
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.yolo_labels import read_yolo_labels, serialize_yolo_labels
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


def build_prune_classes_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Remove unused classes from dataset metadata and remap label ids")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_classes_pruned)")
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


def build_prune_size_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Prune labels smaller than NxM pixels into a new dataset")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Output dataset name (default <dataset>_size_pruned)")
    p.add_argument("--min-size", type=str, default="20x20", help="Minimum bbox size in pixels as NxM (default 20x20)")
    p.add_argument("--drop-empty-images", dest="drop_empty_images", action="store_true", default=True)
    p.add_argument("--no-drop-empty-images", dest="drop_empty_images", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    return p


def build_prune_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Dataset pruning utilities: remove empty pairs or duplicates")
    sub = p.add_subparsers(dest="mode")
    sub.required = False
    sub.add_parser("empty", parents=[build_prune_empty_arg_parser()], add_help=False)
    sub.add_parser("classes", parents=[build_prune_classes_arg_parser()], add_help=False)
    sub.add_parser("dedup", parents=[build_prune_dedup_arg_parser()], add_help=False)
    sub.add_parser("size", parents=[build_prune_size_arg_parser()], add_help=False)
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


def _copy_full_dataset(src_root: str, out_dir: str) -> None:
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(src_root, out_dir)


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


def _parse_min_size(raw: str) -> tuple[float, float]:
    text = str(raw or "").strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", text)
    if not m:
        raise ValueError(f"Invalid --min-size value: {raw!r}. Expected NxM, e.g. 20x20.")
    min_w = float(m.group(1))
    min_h = float(m.group(2))
    if min_w <= 0 or min_h <= 0:
        raise ValueError(f"Invalid --min-size value: {raw!r}. Both dimensions must be > 0.")
    return min_w, min_h


def _prune_small_labels(out_dir: str, *, min_w_px: float, min_h_px: float, drop_empty_images: bool) -> dict[str, int]:
    scanned = 0
    labels_before = 0
    removed_labels = 0
    labels_after = 0
    images_dropped_empty_after_size = 0
    kept_empty_images = 0
    image_size_cache: dict[str, tuple[int, int]] = {}

    for item in _list_output_items(out_dir):
        scanned += 1
        if not os.path.isfile(item.label_path):
            continue
        if item.image_path in image_size_cache:
            iw, ih = image_size_cache[item.image_path]
        else:
            with Image.open(item.image_path) as im:
                iw, ih = int(im.width), int(im.height)
            image_size_cache[item.image_path] = (iw, ih)
        labels = read_yolo_labels(item.label_path)
        if not labels:
            if drop_empty_images:
                try:
                    os.remove(item.image_path)
                except OSError:
                    pass
                try:
                    os.remove(item.label_path)
                except OSError:
                    pass
                images_dropped_empty_after_size += 1
            else:
                kept_empty_images += 1
            continue
        labels_before += len(labels)
        kept = []
        for lb in labels:
            geom = bbox_geom_from_label(lb, img_w=iw, img_h=ih)
            if geom is None:
                kept.append(lb)
                continue
            if geom.w_px < min_w_px or geom.h_px < min_h_px:
                removed_labels += 1
                continue
            kept.append(lb)
        labels_after += len(kept)
        if not kept:
            if drop_empty_images:
                try:
                    os.remove(item.image_path)
                except OSError:
                    pass
                try:
                    os.remove(item.label_path)
                except OSError:
                    pass
                images_dropped_empty_after_size += 1
            else:
                Path(item.label_path).write_text("", encoding="utf-8")
                kept_empty_images += 1
            continue
        Path(item.label_path).write_text(serialize_yolo_labels(kept), encoding="utf-8")

    return {
        "scanned": scanned,
        "labels_before": labels_before,
        "removed_labels": removed_labels,
        "labels_after": labels_after,
        "images_dropped_empty_after_size": images_dropped_empty_after_size,
        "kept_empty_images": kept_empty_images,
    }


def _interactive_fill(args: argparse.Namespace, mode: str, dataset_names: list[str]) -> None:
    print("[INFO] Interactive prune mode")
    args.dataset = prompt_choice("Dataset", dataset_names, default=(args.dataset or dataset_names[0]))
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")).strip() or None
    if mode == "dedup":
        # Keep strict safeguard by default.
        args.allow_balanced_dedup = False
    if mode == "size":
        args.min_size = (
            prompt_text("Minimum bbox size in px NxM (--min-size)", default=str(getattr(args, "min_size", "20x20")))
            .strip()
            or "20x20"
        )
        drop_empty_images = prompt_choice(
            "Drop images with no labels after size prune?",
            ["yes", "no"],
            default="yes" if bool(getattr(args, "drop_empty_images", True)) else "no",
        )
        args.drop_empty_images = drop_empty_images == "yes"


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    mode = argv[0] if argv and argv[0] in {"empty", "dedup", "classes", "size"} else None
    mode_args = argv[1:] if mode else argv
    interactive_allowed = is_interactive_allowed(argv)

    if mode == "dedup":
        parser = build_prune_dedup_arg_parser()
    elif mode == "classes":
        parser = build_prune_classes_arg_parser()
    elif mode == "size":
        parser = build_prune_size_arg_parser()
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
            print("[ERROR] Incomplete arguments: specify prune mode (empty|dedup|classes|size).")
            return
        if not sys.stdin.isatty():
            print("[ERROR] Interactive prune mode requires a terminal (TTY).")
            return
        mode = prompt_choice("Prune mode", ["empty", "dedup", "classes", "size"], default="empty")
        if mode == "dedup":
            parser = build_prune_dedup_arg_parser()
        elif mode == "classes":
            parser = build_prune_classes_arg_parser()
        elif mode == "size":
            parser = build_prune_size_arg_parser()
        else:
            parser = build_prune_empty_arg_parser()
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

    suffix = (
        "pruned"
        if mode == "empty"
        else "deduped"
        if mode == "dedup"
        else "classes_pruned"
        if mode == "classes"
        else "size_pruned"
    )
    out_base = args.output_name or f"{args.dataset}_{suffix}"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    if args.dry_run:
        print(f"[OK] dry-run: mode={mode}, dataset={args.dataset}, output={out_name}")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    entry_dict = entry if isinstance(entry, dict) else {}
    structure = str(entry_dict.get("structure", "split"))
    class_names_path = layout.work_class_names_path()
    class_names_map: dict[str, str] = {}
    if os.path.isfile(class_names_path):
        try:
            with open(class_names_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                class_names_map = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            pass

    min_w_px = 0.0
    min_h_px = 0.0
    if mode == "size":
        try:
            min_w_px, min_h_px = _parse_min_size(getattr(args, "min_size", "20x20"))
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return

    if mode == "classes":
        _copy_full_dataset(src_root, out_dir)
        strip_stats = strip_unused_classes(
            out_dir,
            structure,
            entry_dict,
            class_names_map=class_names_map,
            dry_run=False,
        )
        class_map = strip_stats.new_class_map or entry_dict.get("classes", {})
        out_hash = calculate_dataset_hash(out_dir)
        if isinstance(class_map, dict) and class_map:
            update_datasets_sidecar(
                layout=layout,
                output_key=out_name,
                class_map=class_map,
                target_dir=out_dir,
                output_hash=out_hash,
                structure=structure,
            )
        stats = {
            "classes_before": strip_stats.classes_before,
            "classes_after": strip_stats.classes_after,
            "removed_class_names": strip_stats.removed_class_names,
            "labels_remapped": strip_stats.labels_remapped,
        }
        copied = 0
    else:
        copied = _copy_source_dataset(
            src_root,
            entry_dict,
            out_dir,
            dataset_name=args.dataset,
            tmp_root=os.path.join(layout.root, "tmp"),
        )
        if mode == "empty":
            stats = _prune_empty(out_dir)
        elif mode == "size":
            stats = _prune_small_labels(
                out_dir,
                min_w_px=min_w_px,
                min_h_px=min_h_px,
                drop_empty_images=bool(getattr(args, "drop_empty_images", True)),
            )
        else:
            stats = _prune_dedup(out_dir)

        class_map = entry_dict.get("classes", {})
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
        stats_before={"copied_images": copied} if mode != "classes" else {"classes_before": stats.get("classes_before", 0)},
        stats_after=stats | {"output_hash": out_hash},
        random_seed=None,
    )
    print(f"[OK] Dataset created: {out_dir}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)

