from __future__ import annotations

import argparse
import json
import math
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
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BALANCE_PRESETS: dict[str, dict[str, object]] = {
    # Conservative weighted balancing for most datasets.
    "weights-safe": {
        "strategy": "weights",
        "weight_mode": "effective",
        "beta": 0.9999,
        "image_weight_agg": "mean",
        "weight_clip_min": 0.5,
        "weight_clip_max": 2.0,
        "replacement": "auto",
        "target": 1.2,
        "max_repeat_per_image": 3,
    },
    # LVIS-like repeat-factor sampling, more aggressive on tail classes.
    "rfs-aggressive": {
        "strategy": "rfs",
        "rfs_thresh": 0.002,
        "rfs_power": 0.5,
        "target": 1.5,
        "max_repeat_per_image": 6,
    },
    # Recommended default: moderate RFS + weighted sampling.
    "hybrid-default": {
        "strategy": "hybrid",
        "weight_mode": "effective",
        "beta": 0.9999,
        "image_weight_agg": "max",
        "weight_clip_min": 0.4,
        "weight_clip_max": 3.0,
        "replacement": "auto",
        "rfs_thresh": 0.001,
        "rfs_power": 0.5,
        "target": 1.3,
        "max_repeat_per_image": 5,
    },
}


def build_balance_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Balancing the dataset into a new datasets/<name>")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Source dataset name")
    p.add_argument(
        "--preset",
        choices=tuple(BALANCE_PRESETS.keys()),
        default=None,
        help=(
            "Preset with tuned balancing parameters. "
            "weights-safe: conservative; "
            "rfs-aggressive: stronger tail upsampling; "
            "hybrid-default: recommended general-purpose balance."
        ),
    )
    p.add_argument(
        "--strategy",
        choices=("copy", "oversample", "undersample", "class-aware", "weights", "rfs", "hybrid"),
        default="oversample",
    )
    p.add_argument("--target", type=float, default=1.0, help="Train size multiplier after balancing")
    p.add_argument("--max-ratio", type=float, default=3.0, help="max/min limit for oversample/class-aware")
    p.add_argument("--min-count", type=int, default=1, help="Minimum class count for accounting")
    p.add_argument("--weight-mode", choices=("effective", "inverse", "sqrt-inverse"), default="effective")
    p.add_argument("--beta", type=float, default=0.9999, help="Beta for effective-number weighting")
    p.add_argument("--image-weight-agg", choices=("max", "mean", "sum"), default="max")
    p.add_argument("--weight-clip-min", type=float, default=0.2)
    p.add_argument("--weight-clip-max", type=float, default=5.0)
    p.add_argument("--replacement", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--max-repeat-per-image", type=int, default=5)
    p.add_argument("--rfs-thresh", type=float, default=0.001)
    p.add_argument("--rfs-power", type=float, default=0.5)
    p.add_argument("--class", dest="single_class", type=str, default=None, help="Balance only one class")
    p.add_argument("--classes", type=str, default=None, help="Balance CSV class list")
    p.add_argument("--output-name", type=str, default=None, help="Name of output dataset (default <dataset>_balanced)")
    p.add_argument("--emit-train-config", action="store_true", help="Save balance_manifest.json for train")
    p.add_argument("--emit-balance-report", action="store_true", help="Write expanded balance report to manifest")
    p.add_argument(
        "--eval-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-adjust balanced train split to keep eval splits non-empty and improve class coverage "
            "(enabled by default; disable with --no-eval-coverage)."
        ),
    )
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
    print("[INFO] Interactive balance mode")
    print("[INFO] Available classes:")
    for c in class_names:
        print(f"  - {c}")
    args.dataset = prompt_choice("Dataset", dataset_names, default=dataset_names[0])
    args.strategy = prompt_choice(
        "Strategy",
        ["copy", "oversample", "undersample", "class-aware", "weights", "rfs", "hybrid"],
        default=args.strategy,
    )
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")).strip() or None
    args.target = float(
        prompt_text("train size multiplier (--target)", default=str(args.target)).strip() or str(args.target)
    )
    args.max_ratio = float(
        prompt_text("Limit max/min (--max-ratio)", default=str(args.max_ratio)).strip() or str(args.max_ratio)
    )
    mode = prompt_choice("Classes", ["all", "single", "list"], default="all").lower()
    if mode == "single":
        args.single_class = prompt("Class: ", completer=WordCompleter(class_names, ignore_case=True)).strip()
        args.classes = None
    elif mode == "list":
        selected = prompt_multi_choice_csv("Classes", class_names, default_values=[])
        args.classes = ",".join(selected) if selected else None
        args.single_class = None
    else:
        args.single_class = None
        args.classes = None
    args.emit_balance_report = prompt_yes_no(
        "Write balance report manifest (--emit-balance-report)?",
        default=bool(args.emit_balance_report),
    )
    args.emit_train_config = prompt_yes_no(
        "Write train config manifest (--emit-train-config)?",
        default=bool(args.emit_train_config),
    )
    args.eval_coverage = prompt_yes_no(
        "Auto-fix eval split coverage (--eval-coverage)?",
        default=bool(args.eval_coverage),
    )
    args.dry_run = prompt_yes_no("Do dry-run (--dry-run)?", default=bool(args.dry_run))


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    class_map: dict[str, int],
    target_dir: str,
    output_hash: str,
) -> None:
    os.makedirs(layout.datasets, exist_ok=True)
    rel = os.path.relpath(os.path.abspath(target_dir), layout.root)
    entry = {
        "classes": {str(k): int(v) for k, v in sorted(class_map.items(), key=lambda kv: int(kv[1]))},
        "structure": "split",
        "elements_count": None,
        "data_path": rel,
        "dataset_hash": output_hash,
        "modified": False,
    }
    info_path = layout.work_datasets_info_path()
    previous: dict = {}
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            previous = loaded
    previous[output_key] = entry
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(previous, f, ensure_ascii=False, indent=4)

    cn_path = layout.work_class_names_path()
    class_names_out: dict[str, str] = {}
    if os.path.isfile(cn_path):
        with open(cn_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            class_names_out = {str(k): str(v) for k, v in loaded.items()}
    for c in class_map.keys():
        class_names_out[str(c)] = str(c)
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(class_names_out, f, ensure_ascii=False, indent=4)


def _provided_flags(argv: list[str]) -> set[str]:
    out: set[str] = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            out.add(flag)
        i += 1
    return out


def _apply_preset_defaults(args: argparse.Namespace, provided_flags: set[str]) -> None:
    if not getattr(args, "preset", None):
        return
    preset_cfg = BALANCE_PRESETS.get(str(args.preset), {})
    flag_for_attr = {
        "strategy": "--strategy",
        "weight_mode": "--weight-mode",
        "beta": "--beta",
        "image_weight_agg": "--image-weight-agg",
        "weight_clip_min": "--weight-clip-min",
        "weight_clip_max": "--weight-clip-max",
        "replacement": "--replacement",
        "rfs_thresh": "--rfs-thresh",
        "rfs_power": "--rfs-power",
        "target": "--target",
        "max_repeat_per_image": "--max-repeat-per-image",
    }
    for attr, value in preset_cfg.items():
        flag = flag_for_attr.get(attr)
        if flag and flag in provided_flags:
            continue
        setattr(args, attr, value)


def _build_balancing_stats(
    selected_pool: list[tuple[str, str, str, list[str]]],
    selected_classes: set[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, set[str]]]:
    bbox_count: dict[str, int] = defaultdict(int)
    image_presence: dict[str, int] = defaultdict(int)
    img_to_classes: dict[str, set[str]] = {}
    for _split, img, _lbl, cls_names in selected_pool:
        classes = {c for c in cls_names if not selected_classes or c in selected_classes}
        if not classes and selected_classes:
            continue
        if not classes:
            classes = set(cls_names)
        img_to_classes[img] = classes
        for c in classes:
            image_presence[c] += 1
        for c in cls_names:
            if not selected_classes or c in selected_classes:
                bbox_count[c] += 1
    return bbox_count, image_presence, img_to_classes


def _class_weights(
    bbox_count: dict[str, int],
    *,
    mode: str,
    beta: float,
    min_count: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, n_raw in bbox_count.items():
        n = max(int(n_raw), int(min_count))
        if mode == "inverse":
            w = 1.0 / max(1, n)
        elif mode == "sqrt-inverse":
            w = 1.0 / math.sqrt(max(1, n))
        else:
            b = min(max(float(beta), 0.0), 0.999999)
            eff = (1.0 - (b**n)) / max(1e-12, 1.0 - b)
            w = 1.0 / max(eff, 1e-12)
        out[c] = w
    if not out:
        return out
    mean_w = sum(out.values()) / len(out)
    if mean_w > 0:
        out = {k: v / mean_w for k, v in out.items()}
    return out


def _image_weights(
    selected_pool: list[tuple[str, str, str, list[str]]],
    class_weights: dict[str, float],
    *,
    agg: str,
    clip_min: float,
    clip_max: float,
    selected_classes: set[str],
) -> list[float]:
    weights: list[float] = []
    for _split, _img, _lbl, cls_names in selected_pool:
        classes = [c for c in cls_names if (not selected_classes or c in selected_classes)]
        vals = [class_weights.get(c, 1.0) for c in classes] or [1.0]
        if agg == "sum":
            w = sum(vals)
        elif agg == "mean":
            w = sum(vals) / len(vals)
        else:
            w = max(vals)
        w = max(float(clip_min), min(float(clip_max), float(w)))
        weights.append(w)
    return weights


def _weighted_sample_items(
    pool: list[tuple[str, str, str, list[str]]],
    weights: list[float],
    target_n: int,
    *,
    replacement: str,
    max_repeat_per_image: int,
    rng: random.Random,
) -> list[tuple[str, str, str, list[str]]]:
    if not pool:
        return []
    if replacement == "on":
        use_repl = True
    elif replacement == "off":
        use_repl = False
    else:
        use_repl = target_n > len(pool)
    target_n = max(1, int(target_n))
    out: list[tuple[str, str, str, list[str]]] = []
    if not use_repl:
        idxs = list(range(len(pool)))
        pick_n = min(target_n, len(pool))
        chosen = rng.choices(idxs, weights=weights, k=pick_n * 2)
        seen = set()
        for i in chosen:
            if i in seen:
                continue
            seen.add(i)
            out.append(pool[i])
            if len(out) >= pick_n:
                break
        if len(out) < pick_n:
            for i in idxs:
                if i in seen:
                    continue
                out.append(pool[i])
                if len(out) >= pick_n:
                    break
        return out
    counts_by_img: dict[str, int] = defaultdict(int)
    idxs = list(range(len(pool)))
    while len(out) < target_n:
        i = rng.choices(idxs, weights=weights, k=1)[0]
        img_key = pool[i][1]
        if counts_by_img[img_key] >= max(1, int(max_repeat_per_image)):
            continue
        counts_by_img[img_key] += 1
        out.append(pool[i])
    return out


def _rfs_expand_pool(
    pool: list[tuple[str, str, str, list[str]]],
    img_to_classes: dict[str, set[str]],
    image_presence: dict[str, int],
    *,
    rfs_thresh: float,
    rfs_power: float,
    max_repeat_per_image: int,
    rng: random.Random,
) -> list[tuple[str, str, str, list[str]]]:
    n_images = max(1, len({img for _s, img, _l, _c in pool}))
    class_repeat: dict[str, float] = {}
    for c, n_img in image_presence.items():
        f_c = max(1e-12, float(n_img) / float(n_images))
        class_repeat[c] = max(1.0, (float(rfs_thresh) / f_c) ** float(rfs_power))
    out: list[tuple[str, str, str, list[str]]] = []
    for item in pool:
        img = item[1]
        classes = img_to_classes.get(img, set())
        r_i = max([class_repeat.get(c, 1.0) for c in classes] or [1.0])
        base = int(math.floor(r_i))
        frac = r_i - base
        repeats = base + (1 if rng.random() < frac else 0)
        repeats = min(max(1, repeats), max(1, int(max_repeat_per_image)))
        for _ in range(repeats):
            out.append(item)
    return out


def _ensure_non_empty_eval_splits(
    balanced_train: list[tuple[str, str, str, list[str]]],
    passthrough_items: list[tuple[str, str, str, list[str]]],
    *,
    seed: int,
) -> list[tuple[str, str, str, list[str]]]:
    """
    Ensure val/test are not empty in output when possible.
    We keep the current logic as-is unless one of eval splits is empty.
    If needed, move a deterministic subset of balanced-train items to val/test
    targeting roughly 80/10/10 split.
    """
    out_train = list(balanced_train)
    val_count = sum(1 for s, *_ in passthrough_items if s == "val")
    test_count = sum(1 for s, *_ in passthrough_items if s == "test")
    total = len(out_train) + len(passthrough_items)
    if total < 3:
        return out_train
    have_non_empty_eval = val_count > 0 and test_count > 0

    target_val = max(1, int(round(total * 0.1)))
    target_test = max(1, int(round(total * 0.1)))
    need_val = max(0, target_val - val_count)
    need_test = max(0, target_test - test_count)
    # Keep at least one item in train.
    can_move = max(0, len(out_train) - 1)
    if need_val + need_test > can_move:
        # Prioritize making both splits non-empty first.
        min_need_val = 1 if val_count == 0 and can_move > 0 else 0
        min_need_test = 1 if test_count == 0 and can_move > min_need_val else 0
        left = max(0, can_move - min_need_val - min_need_test)
        need_val = min_need_val
        need_test = min_need_test
        # Distribute remaining budget approximately evenly.
        add_val = min(left // 2 + left % 2, max(0, target_val - val_count - need_val))
        need_val += add_val
        left -= add_val
        add_test = min(left, max(0, target_test - test_count - need_test))
        need_test += add_test

    if (not have_non_empty_eval) and (need_val > 0 or need_test > 0):
        rng = random.Random(seed)
        idxs = list(range(len(out_train)))
        rng.shuffle(idxs)

        pos = 0
        for _ in range(need_val):
            i = idxs[pos]
            pos += 1
            _s, img, lbl, cls = out_train[i]
            out_train[i] = ("val", img, lbl, cls)
        for _ in range(need_test):
            i = idxs[pos]
            pos += 1
            _s, img, lbl, cls = out_train[i]
            out_train[i] = ("test", img, lbl, cls)

    # Optional class-coverage enrichment for eval splits:
    # if a class exists globally but is absent in val/test, move a minimal
    # number of train items containing that class to the target split.
    global_classes = {
        c
        for _s, _img, _lbl, cls_names in (out_train + passthrough_items)
        for c in cls_names
    }
    if not global_classes:
        return out_train

    rng_cov = random.Random(seed + 17)

    def present_classes(split_name: str) -> set[str]:
        out = set()
        for s, _img, _lbl, cls_names in out_train:
            if s == split_name:
                out.update(cls_names)
        for s, _img, _lbl, cls_names in passthrough_items:
            if s == split_name:
                out.update(cls_names)
        return out

    def train_count() -> int:
        return sum(1 for s, *_ in out_train if s == "train")

    for target_split in ("val", "test"):
        missing = set(global_classes) - present_classes(target_split)
        # keep at least one sample in train
        while missing and train_count() > 1:
            candidates: list[tuple[int, int]] = []
            for i, (s, _img, _lbl, cls_names) in enumerate(out_train):
                if s != "train":
                    continue
                cover = len(set(cls_names) & missing)
                if cover > 0:
                    candidates.append((cover, i))
            if not candidates:
                break
            # maximize coverage; tie-break with deterministic random jitter.
            max_cover = max(c for c, _ in candidates)
            best = [i for c, i in candidates if c == max_cover]
            chosen_idx = best[rng_cov.randrange(len(best))]
            s, img, lbl, cls_names = out_train[chosen_idx]
            out_train[chosen_idx] = (target_split, img, lbl, cls_names)
            missing -= set(cls_names)
    return out_train


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_balance_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    _apply_preset_defaults(args, _provided_flags(argv))
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] datasets_info.json was not found or is empty.")
        return

    if args.dataset is None and interactive_allowed and sys.stdin.isatty():
        all_classes = sorted({k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()})
        _interactive_fill(args, sorted(catalog.keys()), all_classes)
        interactive_used = True
    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("balance", parser, args)
        print_replay_command("before launch", replay_cmd)

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
            print(f"[ERROR] Unknown classes in the filter: {', '.join(unknown)}")
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
        print("[ERROR] There is no train data for balancing in the source dataset.")
        return

    selected_pool = []
    for it in train_items:
        _, _, _, classes = it
        if selected_classes and set(classes).isdisjoint(selected_classes):
            continue
        selected_pool.append(it)
    if not selected_pool:
        selected_pool = train_items

    if args.strategy == "copy":
        balanced_train = list(selected_pool)
    elif args.strategy == "undersample":
        target_n = max(1, int(len(selected_pool) * max(0.1, min(1.0, args.target))))
        balanced_train = random.sample(selected_pool, min(target_n, len(selected_pool)))
    elif args.strategy == "oversample":
        target_n = max(len(selected_pool), int(len(selected_pool) * max(1.0, args.target)))
        balanced_train = [random.choice(selected_pool) for _ in range(target_n)]
    else:
        rng = random.Random(args.seed)
        bbox_count, image_presence, img_to_classes = _build_balancing_stats(selected_pool, selected_classes)
        if args.strategy == "rfs":
            expanded = _rfs_expand_pool(
                selected_pool,
                img_to_classes,
                image_presence,
                rfs_thresh=args.rfs_thresh,
                rfs_power=args.rfs_power,
                max_repeat_per_image=args.max_repeat_per_image,
                rng=rng,
            )
            target_n = max(1, int(len(selected_pool) * max(0.1, args.target)))
            if target_n <= len(expanded):
                balanced_train = expanded[:target_n]
            else:
                balanced_train = expanded + [rng.choice(expanded) for _ in range(target_n - len(expanded))]
        else:
            class_w = _class_weights(
                bbox_count,
                mode=args.weight_mode,
                beta=args.beta,
                min_count=args.min_count,
            )
            pool_for_weights = selected_pool
            if args.strategy == "hybrid":
                pool_for_weights = _rfs_expand_pool(
                    selected_pool,
                    img_to_classes,
                    image_presence,
                    rfs_thresh=args.rfs_thresh,
                    rfs_power=args.rfs_power,
                    max_repeat_per_image=args.max_repeat_per_image,
                    rng=rng,
                )
            img_w = _image_weights(
                pool_for_weights,
                class_w,
                agg=args.image_weight_agg,
                clip_min=args.weight_clip_min,
                clip_max=args.weight_clip_max,
                selected_classes=selected_classes,
            )
            target_n = max(1, int(len(selected_pool) * max(0.1, args.target)))
            balanced_train = _weighted_sample_items(
                pool_for_weights,
                img_w,
                target_n,
                replacement=args.replacement,
                max_repeat_per_image=args.max_repeat_per_image,
                rng=rng,
            )

    out_base = args.output_name or f"{args.dataset}_balanced"
    out_name = next_dataset_name(layout.datasets, out_base)
    out_dir = os.path.join(layout.datasets, out_name)
    if bool(getattr(args, "eval_coverage", True)):
        balanced_train = _ensure_non_empty_eval_splits(
            balanced_train,
            passthrough_items,
            seed=int(args.seed),
        )

    if args.dry_run:
        print(f"[OK] dry-run: strategy={args.strategy}, train_in={len(train_items)}, train_out={len(balanced_train)}, output={out_name}")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
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
        "train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(
        layout,
        out_name,
        {str(k): int(v) for k, v in class_map.items()},
        out_dir,
        out_hash,
    )
    if args.emit_train_config or args.emit_balance_report:
        counts_after: dict[str, int] = defaultdict(int)
        for _split, _img, _lbl, cls_names in balanced_train:
            for c in cls_names:
                if not selected_classes or c in selected_classes:
                    counts_after[c] += 1
        Path(out_dir, "balance_manifest.json").write_text(
            json.dumps(
                {
                    "strategy": args.strategy,
                    "selected_classes": sorted(selected_classes),
                    "train_input": len(train_items),
                    "train_output": len(balanced_train),
                    "seed": args.seed,
                    "target": args.target,
                    "replacement": args.replacement,
                    "weight_mode": args.weight_mode,
                    "beta": args.beta,
                    "image_weight_agg": args.image_weight_agg,
                    "weight_clip_min": args.weight_clip_min,
                    "weight_clip_max": args.weight_clip_max,
                    "rfs_thresh": args.rfs_thresh,
                    "rfs_power": args.rfs_power,
                    "max_repeat_per_image": args.max_repeat_per_image,
                    "class_counts_before_bbox": dict(class_counts),
                    "class_counts_after_bbox": dict(counts_after),
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
        workspace_root=layout.root,
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
    print(f"[OK] Dataset created: {out_dir}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)


if __name__ == "__main__":
    main()

