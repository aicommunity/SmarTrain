from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageEnhance
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from smartrain.cli_argparse import CliArgumentParser
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def build_augment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Офлайн-аугментация датасета в новый datasets/<name>")
    p.add_argument("--workspace", type=str, default=None, help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Имя исходного датасета из datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Имя выходного датасета (по умолчанию <dataset>_aug)")
    p.add_argument("--policy", choices=("basic", "robust", "longtail_copy_paste"), default="basic")
    p.add_argument("--multiplier", type=int, default=1, help="Сколько аугментированных копий на исходное изображение")
    p.add_argument("--splits", type=str, default="train", help="CSV: train,val,test")
    p.add_argument("--classes", type=str, default=None, help="Ограничить аугментацию классами CSV")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-legend", action="store_true")
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _detect_split(images_path: str) -> str:
    low = images_path.lower()
    if "/train/" in low:
        return "train"
    if "/val/" in low or "/valid/" in low:
        return "val"
    if "/test/" in low:
        return "test"
    return "train"


def _read_yolo_classes(label_path: str) -> set[int]:
    out: set[int] = set()
    if not os.path.isfile(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                out.add(int(float(parts[0])))
            except ValueError:
                continue
    return out


def _augment_image_and_labels(
    image_path: str, label_path: str, out_img: str, out_lbl: str, *, policy: str, seed: int
) -> None:
    random.seed(seed)
    img = Image.open(image_path).convert("RGB")
    lines = []
    if os.path.isfile(label_path):
        lines = [x.strip() for x in Path(label_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    # basic: горизонтальный флип
    if policy in ("basic", "longtail_copy_paste"):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            x, y, w, h = map(float, parts[1:5])
            x = 1.0 - x
            flipped.append(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
        lines = flipped
    # robust: плюс яркость/контраст
    if policy == "robust":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        flipped = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            x, y, w, h = map(float, parts[1:5])
            x = 1.0 - x
            flipped.append(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
        lines = flipped

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    img.save(out_img)
    Path(out_lbl).write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _write_data_yaml(out_dir: str, names: list[str]) -> None:
    p = Path(out_dir) / "data.yaml"
    p.write_text(
        "train: ./train/images\nval: ./val/images\ntest: ./test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def _interactive_fill(args, dataset_names: list[str], classes: list[str]) -> None:
    print("[INFO] Интерактивный режим augment")
    print("[INFO] Доступные датасеты:")
    for n in dataset_names:
        print(f"  - {n}")
    print("[INFO] Доступные классы:")
    for c in classes:
        print(f"  - {c}")
    args.dataset = prompt("Датасет: ", completer=WordCompleter(dataset_names, ignore_case=True)).strip()
    args.output_name = prompt("Имя выходного датасета (пусто=авто): ", default=(args.output_name or "")).strip() or None
    args.policy = prompt(
        "Policy (basic/robust/longtail_copy_paste): ",
        default=args.policy,
        completer=WordCompleter(["basic", "robust", "longtail_copy_paste"], ignore_case=True),
    ).strip() or args.policy
    args.multiplier = int(prompt("Multiplier: ", default=str(args.multiplier)).strip() or str(args.multiplier))
    args.splits = prompt("Splits CSV (train,val,test): ", default=args.splits).strip() or args.splits
    args.classes = (
        prompt(
            "Классы CSV (пусто=все): ",
            default="",
            completer=WordCompleter(classes, ignore_case=True),
            complete_while_typing=True,
        ).strip()
        or None
    )
    args.dry_run = (prompt("Dry-run? [y/N]: ", default="n").strip().lower() in ("y", "yes", "1", "true"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_augment_arg_parser().parse_args(argv)
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] Не найдено datasets_info.json или он пуст.")
        return

    if args.dataset is None and not argv:
        # на всякий случай, но у нас подкоманда вызывается с аргументами от cli
        pass
    if args.dataset is None and sys.stdin.isatty():
        all_classes = sorted({k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()})
        _interactive_fill(args, sorted(catalog.keys()), all_classes)

    if not args.dataset:
        print("[ERROR] Укажите --dataset или используйте интерактивный режим.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Неизвестный датасет: {args.dataset}")
        return

    entry = catalog[args.dataset]
    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    class_map = entry.get("classes", {})
    names_by_id = {int(v): str(k) for k, v in class_map.items()} if isinstance(class_map, dict) else {}
    allowed_classes = None
    if args.classes:
        allowed_classes = {x.strip() for x in args.classes.split(",") if x.strip()}
    split_filter = {SPLIT_ALIASES.get(x.strip().lower(), x.strip().lower()) for x in args.splits.split(",") if x.strip()}
    out_base = args.output_name or f"{args.dataset}_aug"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    buckets = iter_image_label_buckets(
        src_root,
        str(entry.get("structure", "split")),
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )
    copied = 0
    augmented = 0
    if not args.dry_run:
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        for file_name in os.listdir(images_path):
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img_src = os.path.join(images_path, file_name)
            lbl_src = os.path.join(labels_path, f"{stem}.txt")
            classes_in_image = _read_yolo_classes(lbl_src)
            class_names = {names_by_id.get(i, f"id_{i}") for i in classes_in_image}
            if allowed_classes and class_names.isdisjoint(allowed_classes):
                continue
            if not args.dry_run:
                dst_img = os.path.join(out_dir, split, "images", file_name)
                dst_lbl = os.path.join(out_dir, split, "labels", f"{stem}.txt")
                shutil.copy2(img_src, dst_img)
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, dst_lbl)
                else:
                    Path(dst_lbl).write_text("", encoding="utf-8")
            copied += 1
            if split not in split_filter:
                continue
            for i in range(args.multiplier):
                aug_stem = f"{stem}__aug{i+1}"
                if not args.dry_run:
                    out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                    out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
                    _augment_image_and_labels(
                        img_src,
                        lbl_src,
                        out_img,
                        out_lbl,
                        policy=args.policy,
                        seed=args.seed + i,
                    )
                augmented += 1

    if args.dry_run:
        print(f"[OK] dry-run: copied={copied}, augmented={augmented}, output={out_name}")
        return

    all_names = [str(x) for _, x in sorted(names_by_id.items())]
    _write_data_yaml(out_dir, all_names)
    out_hash = calculate_dataset_hash(out_dir)
    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command="augment",
        source_datasets=[
            {
                "name": args.dataset,
                "path": src_root,
                "dataset_hash": entry.get("dataset_hash"),
            }
        ],
        parameters=vars(args),
        transformations=[{"policy": args.policy, "multiplier": args.multiplier, "splits": sorted(split_filter)}],
        random_seed=args.seed,
        stats_before={"copied_input_images": copied},
        stats_after={"copied_images": copied, "augmented_images": augmented, "output_hash": out_hash},
    )
    print(f"[OK] Создан датасет: {out_dir}")
    print(f"[OK] Passport: {passport_path}")


if __name__ == "__main__":
    main()

