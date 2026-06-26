from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2
from tqdm import tqdm

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.yolo_labels import read_yolo_labels, rotate_yolo_labels_90cw_k, write_yolo_labels

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ROTATE_ANGLES = (90, 180, 270)
ANGLE_TO_K = {90: 1, 180: 2, 270: 3}


def build_rotate_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(
        description="Rotate a YOLO dataset by a fixed angle (90/180/270 degrees clockwise) into datasets/<name>_rot<angle>"
    )
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset key from datasets_info.json")
    p.add_argument(
        "--angle",
        type=int,
        choices=ROTATE_ANGLES,
        default=None,
        help="Clockwise rotation angle in degrees: 90, 180, or 270",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output dataset name (default <dataset>_rot<angle>)",
    )
    p.add_argument("--dry-run", action="store_true", help="Count images only, do not write output dataset")
    p.add_argument("--no-legend", action="store_true", help="Disable tqdm progress bar")
    return p


def angle_to_k(angle: int) -> int:
    try:
        return ANGLE_TO_K[int(angle)]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Unsupported angle {angle!r}; choose one of {list(ROTATE_ANGLES)}") from e


def default_output_name(dataset: str, angle: int) -> str:
    return f"{dataset}_rot{int(angle)}"


def _rot_img_k(img, k: int):
    kk = int(k) % 4
    if kk == 0:
        return img
    if kk == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if kk == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _copy_data_yaml_if_exists(src_root: str, dst_root: str) -> None:
    for name in ("data.yaml", "data.yml"):
        sp = os.path.join(src_root, name)
        if os.path.isfile(sp):
            os.makedirs(dst_root, exist_ok=True)
            shutil.copy2(sp, os.path.join(dst_root, name))


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    entry: dict,
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    new_entry = dict(entry) if isinstance(entry, dict) else {}
    new_entry["data_path"] = rel
    new_entry["dataset_hash"] = output_hash
    new_entry["modified"] = False
    info_path = layout.work_datasets_info_path()
    prev: dict = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            prev = loaded
    prev[output_key] = new_entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=4)


def _count_images_in_buckets(buckets) -> int:
    total = 0
    for images_path, _labels_path in buckets:
        for name in os.listdir(images_path):
            _stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS:
                total += 1
    return total


def _interactive_fill(args, *, dataset_names: list[str]) -> None:
    print("[INFO] Interactive rotate mode")
    args.dataset = prompt_choice("Dataset", dataset_names, default=dataset_names[0])
    print("[INFO] Fixed clockwise rotation for every frame in the dataset:")
    print("  90  - quarter turn clockwise (width and height swap)")
    print("  180 - half turn")
    print("  270 - three-quarter turn clockwise (width and height swap)")
    angle_raw = prompt_choice(
        "Rotation angle in degrees CW (--angle)",
        [str(a) for a in ROTATE_ANGLES],
        default="90",
    )
    args.angle = int(angle_raw)
    auto_name = default_output_name(str(args.dataset), int(args.angle))
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=auto_name).strip() or None
    args.dry_run = prompt_yes_no("Dry-run only (--dry-run)?", default=bool(args.dry_run))


def rotate_dataset(
    *,
    src_root: str,
    out_dir: str,
    k: int,
    buckets,
    dry_run: bool = False,
    no_legend: bool = False,
) -> dict[str, int]:
    total = _count_images_in_buckets(buckets)
    processed = 0
    skipped = 0
    progress = tqdm(total=total, desc="rotate:images", unit="img", disable=bool(no_legend))
    for images_path, labels_path in buckets:
        rel_images = os.path.relpath(os.path.abspath(images_path), os.path.abspath(src_root))
        rel_labels = os.path.relpath(os.path.abspath(labels_path), os.path.abspath(src_root))
        dst_images = os.path.join(out_dir, rel_images)
        dst_labels = os.path.join(out_dir, rel_labels)
        if not dry_run:
            os.makedirs(dst_images, exist_ok=True)
            os.makedirs(dst_labels, exist_ok=True)

        for name in os.listdir(images_path):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(images_path, name)
            lbl_path = os.path.join(labels_path, f"{stem}.txt")
            if dry_run:
                processed += 1
                progress.update(1)
                continue

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARNING] Failed to read image: {img_path}")
                skipped += 1
                progress.update(1)
                continue
            rbgr = _rot_img_k(bgr, k)
            labels = read_yolo_labels(lbl_path)
            rotated_labels, _new_w, _new_h = rotate_yolo_labels_90cw_k(
                labels,
                w=int(bgr.shape[1]),
                h=int(bgr.shape[0]),
                k=int(k),
            )
            out_img = os.path.join(dst_images, name)
            out_lbl = os.path.join(dst_labels, f"{stem}.txt")
            cv2.imwrite(out_img, rbgr)
            write_yolo_labels(out_lbl, rotated_labels)
            processed += 1
            progress.update(1)
    progress.close()
    return {"total": total, "processed": processed, "skipped": skipped}


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_rotate_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return

    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = load_datasets_catalog(layout)
    if not catalog:
        print(EMPTY_DATASETS_INFO_MESSAGE)
        return

    interactive_used = try_prompt_dataset_interactive(
        args=args,
        argv=argv,
        fill=lambda: _interactive_fill(args, dataset_names=sorted(catalog.keys())),
    )

    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return
    if args.angle is None:
        print("[ERROR] Incomplete arguments: specify --angle (90, 180, or 270).")
        return

    k = angle_to_k(int(args.angle))
    entry = catalog[args.dataset]
    structure = str(entry.get("structure", "split"))
    if structure == "cvat11":
        print("[ERROR] structure=cvat11 is not supported for rotate (YOLO layout datasets only).")
        return

    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    buckets = iter_image_label_buckets(
        src_root,
        structure,
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    if not buckets:
        print("[ERROR] No images/labels pairs were found in the dataset.")
        return

    out_base = args.output_name or default_output_name(args.dataset, int(args.angle))
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("rotate", parser, args)
        print_replay_command("before launch", replay_cmd)

    print(f"[INFO] rotate: dataset={args.dataset} angle={int(args.angle)}cw output={out_name}")

    stats = rotate_dataset(
        src_root=src_root,
        out_dir=out_dir,
        k=k,
        buckets=buckets,
        dry_run=bool(args.dry_run),
        no_legend=bool(args.no_legend),
    )

    print("[OK] rotate summary")
    print(f"  source: {args.dataset}")
    print(f"  angle: {int(args.angle)} (clockwise)")
    print(f"  output: {out_name if not args.dry_run else '<dry-run>'}")
    print(f"  images_total: {stats['total']}")
    print(f"  images_processed: {stats['processed']}")
    if stats["skipped"]:
        print(f"  images_skipped: {stats['skipped']}")

    if args.dry_run:
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    _copy_data_yaml_if_exists(src_root, out_dir)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(layout, out_name, entry if isinstance(entry, dict) else {}, out_dir, out_hash)
    write_dataset_passport(
        output_dataset_dir=out_dir,
        command="rotate",
        source_datasets=[
            {
                "name": args.dataset,
                "dataset_hash": entry.get("dataset_hash") if isinstance(entry, dict) else None,
            }
        ],
        workspace_root=layout.root,
        parameters={
            "dataset": args.dataset,
            "output_name": out_name,
            "angle": int(args.angle),
        },
        transformations=[{"type": "fixed_rotation_cw", "angle": int(args.angle)}],
        stats_before={"images_total": stats["total"]},
        stats_after={"images_processed": stats["processed"], "output_hash": out_hash},
        random_seed=None,
    )
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)
