from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import prompt_choice, prompt_multi_choice_csv, prompt_text, prompt_yes_no
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.dataset_hash import calculate_dataset_hash
from smartrain.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_balance_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Балансировка датасета в новый datasets/<name>")
    p.add_argument("--workspace", type=str, default=None, help=f"Корень workspace (иначе {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Имя исходного датасета")
    p.add_argument("--strategy", choices=("oversample", "undersample", "class-aware", "weights"), default="oversample")
    p.add_argument("--target", type=float, default=1.0, help="Множитель размера train после балансировки")
    p.add_argument("--max-ratio", type=float, default=3.0, help="Ограничение max/min для oversample/class-aware")
    p.add_argument("--min-count", type=int, default=1, help="Минимальный count класса для учета")
    p.add_argument("--class", dest="single_class", type=str, default=None, help="Балансировать только один класс")
    p.add_argument("--classes", type=str, default=None, help="Балансировать список классов CSV")
    p.add_argument("--output-name", type=str, default=None, help="Имя выходного датасета (по умолчанию <dataset>_balanced)")
    p.add_argument("--emit-train-config", action="store_true", help="Сохранить balance_manifest.json для train")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dry-run", action="store_true")
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    p = layout.work_datasets_info_path()
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
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


def _read_label_classes(label_path: str) -> list[int]:
    out: list[int] = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            try:
                out.append(int(float(parts[0])))
            except ValueError:
                continue
    return out


def _interactive_fill(args, dataset_names: list[str], class_names: list[str]) -> None:
    print("[INFO] Интерактивный режим balance")
    print("[INFO] Доступные датасеты:")
    for n in dataset_names:
        print(f"  - {n}")
    print("[INFO] Доступные классы:")
    for c in class_names:
        print(f"  - {c}")
    args.dataset = prompt("Датасет: ", completer=WordCompleter(dataset_names, ignore_case=True)).strip()
    args.strategy = prompt_choice(
        "Strategy",
        ["oversample", "undersample", "class-aware", "weights"],
        default=args.strategy,
    )
    args.output_name = prompt_text("Имя выходного датасета (пусто=авто)", default=(args.output_name or "")).strip() or None
    args.target = float(
        prompt_text("Множитель размера train (--target)", default=str(args.target)).strip() or str(args.target)
    )
    args.max_ratio = float(
        prompt_text("Ограничение max/min (--max-ratio)", default=str(args.max_ratio)).strip() or str(args.max_ratio)
    )
    mode = prompt_choice("Классы", ["all", "single", "list"], default="all").lower()
    if mode == "single":
        args.single_class = prompt("Класс: ", completer=WordCompleter(class_names, ignore_case=True)).strip()
        args.classes = None
    elif mode == "list":
        selected = prompt_multi_choice_csv("Классы", class_names, default_values=[])
        args.classes = ",".join(selected) if selected else None
        args.single_class = None
    else:
        args.single_class = None
        args.classes = None
    args.dry_run = prompt_yes_no("Выполнить dry-run (--dry-run)?", default=bool(args.dry_run))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_balance_arg_parser()
    args = parser.parse_args(argv)
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] Не найдено datasets_info.json или он пуст.")
        return

    if args.dataset is None and sys.stdin.isatty():
        all_classes = sorted({k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()})
        _interactive_fill(args, sorted(catalog.keys()), all_classes)
        interactive_used = True
    if not args.dataset:
        print("[ERROR] Укажите --dataset или используйте интерактивный режим.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Неизвестный датасет: {args.dataset}")
        return
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("balance", parser, args)
        print_replay_command("перед запуском", replay_cmd)

    random.seed(args.seed)
    entry = catalog[args.dataset]
    class_map = entry.get("classes", {}) if isinstance(entry.get("classes"), dict) else {}
    id_to_name = {int(v): str(k) for k, v in class_map.items()}
    selected_classes = set()
    if args.single_class:
        selected_classes.add(args.single_class.strip())
    if args.classes:
        selected_classes.update({x.strip() for x in args.classes.split(",") if x.strip()})
    if selected_classes:
        unknown = [c for c in selected_classes if c not in class_map]
        if unknown:
            print(f"[ERROR] Неизвестные классы в фильтре: {', '.join(unknown)}")
            return

    src_root = resolve_dataset_root_for_entry(
        args.dataset, entry, workspace_root=layout.root, source_catalog_dir=layout.datasets, legacy_source_parent=layout.datasets
    )
    buckets = iter_image_label_buckets(
        src_root,
        str(entry.get("structure", "split")),
        entry,
        dataset_name=args.dataset,
        temp_root=os.path.join(layout.root, "tmp"),
        exclude_test=False,
    )

    train_items: list[tuple[str, str, str, list[str]]] = []
    passthrough_items: list[tuple[str, str, str, list[str]]] = []
    class_counts = defaultdict(int)
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        for name in os.listdir(images_path):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            img = os.path.join(images_path, name)
            lbl = os.path.join(labels_path, f"{stem}.txt")
            cls_ids = _read_label_classes(lbl)
            cls_names = [id_to_name.get(i, f"id_{i}") for i in cls_ids]
            for c in cls_names:
                if not selected_classes or c in selected_classes:
                    class_counts[c] += 1
            item = (split, img, lbl, cls_names)
            if split == "train":
                train_items.append(item)
            else:
                passthrough_items.append(item)

    if not train_items:
        print("[ERROR] В исходном датасете нет train-данных для балансировки.")
        return

    selected_pool = []
    for it in train_items:
        _, _, _, classes = it
        if selected_classes and set(classes).isdisjoint(selected_classes):
            continue
        selected_pool.append(it)
    if not selected_pool:
        selected_pool = train_items

    if args.strategy == "undersample":
        target_n = max(1, int(len(selected_pool) * max(0.1, min(1.0, args.target))))
        balanced_train = random.sample(selected_pool, min(target_n, len(selected_pool)))
    elif args.strategy in ("oversample", "class-aware"):
        target_n = max(len(selected_pool), int(len(selected_pool) * max(1.0, args.target)))
        balanced_train = [random.choice(selected_pool) for _ in range(target_n)]
    else:  # weights
        balanced_train = list(selected_pool)

    out_base = args.output_name or f"{args.dataset}_balanced"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)

    if args.dry_run:
        print(f"[OK] dry-run: strategy={args.strategy}, train_in={len(train_items)}, train_out={len(balanced_train)}, output={out_name}")
        if replay_cmd:
            print_replay_command("после выполнения", replay_cmd)
        return

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

    def _copy_items(items, *, suffix: str = ""):
        idx = 1
        for split, img, lbl, _ in items:
            ext = os.path.splitext(img)[1]
            stem = f"{Path(img).stem}{suffix}_{idx}" if suffix else f"{Path(img).stem}_{idx}"
            dst_img = os.path.join(out_dir, split, "images", f"{stem}{ext}")
            dst_lbl = os.path.join(out_dir, split, "labels", f"{stem}.txt")
            shutil.copy2(img, dst_img)
            if os.path.isfile(lbl):
                shutil.copy2(lbl, dst_lbl)
            else:
                Path(dst_lbl).write_text("", encoding="utf-8")
            idx += 1

    _copy_items(balanced_train, suffix="bal")
    _copy_items(passthrough_items)

    names = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    Path(out_dir, "data.yaml").write_text(
        "train: ./train/images\nval: ./val/images\ntest: ./test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    out_hash = calculate_dataset_hash(out_dir)
    if args.emit_train_config:
        Path(out_dir, "balance_manifest.json").write_text(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "selected_classes": sorted(selected_classes),
                    "train_input": len(train_items),
                    "train_output": len(balanced_train),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    passport_path = write_dataset_passport(
        output_dataset_dir=out_dir,
        command="balance",
        source_datasets=[{"name": args.dataset, "path": src_root, "dataset_hash": entry.get("dataset_hash")}],
        parameters=vars(args),
        transformations=[
            {
                "strategy": args.strategy,
                "selected_classes": sorted(selected_classes),
                "target": args.target,
                "max_ratio": args.max_ratio,
            }
        ],
        random_seed=args.seed,
        stats_before={"train_images": len(train_items)},
        stats_after={"train_images_balanced": len(balanced_train), "output_hash": out_hash},
    )
    print(f"[OK] Создан датасет: {out_dir}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("после выполнения", replay_cmd)


if __name__ == "__main__":
    main()

