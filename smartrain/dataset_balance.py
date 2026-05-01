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
    p.add_argument(
        "--class-weight-multiplier",
        type=str,
        default="",
        help=(
            "Per-class weight multipliers CSV, e.g. "
            '"other:0.6,tear_up:1.1". Multipliers are applied after base class weights.'
        ),
    )
    p.add_argument("--image-weight-agg", choices=("max", "mean", "sum"), default="max")
    p.add_argument("--weight-clip-min", type=float, default=0.2)
    p.add_argument("--weight-clip-max", type=float, default=5.0)
    p.add_argument("--replacement", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--max-repeat-per-image", type=int, default=5)
    p.add_argument("--rfs-thresh", type=float, default=0.001)
    p.add_argument("--rfs-power", type=float, default=0.5)
    p.add_argument(
        "--auto-head-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-calculate dampening multipliers for overrepresented head classes "
            "to reduce dominance in weighted sampling (enabled by default; disable with --no-auto-head-cap)."
        ),
    )
    p.add_argument(
        "--auto-head-cap-quantile",
        type=float,
        default=0.85,
        help="Quantile used to define head classes for auto cap (0..1).",
    )
    p.add_argument(
        "--auto-head-cap-min-mult",
        type=float,
        default=0.35,
        help="Minimum multiplier for auto head-cap dampening.",
    )
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
            "without source-image leakage between train/val/test; if unique images are insufficient, "
            "eval splits may stay partially unfilled (enabled by default; disable with --no-eval-coverage)."
        ),
    )
    p.add_argument(
        "--eval-min-class-count",
        type=int,
        default=0,
        help=(
            "Optional target minimum bbox count per class in eval splits (val/test). "
            "When > 0 and eval coverage is enabled, balance may move split-safe train source keys "
            "to val/test to increase tail coverage."
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


def _source_image_key(image_path: str) -> str:
    # Stable key for split-uniqueness checks across copied/linked paths.
    return os.path.normcase(os.path.realpath(image_path))


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
        default=True,
    )
    args.emit_train_config = prompt_yes_no(
        "Write train config manifest (--emit-train-config)?",
        default=True,
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


def _parse_class_weight_multiplier(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    text = (raw or "").strip()
    if not text:
        return out
    for token in text.split(","):
        part = token.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid class multiplier token: '{part}'. Expected <class>:<multiplier>.")
        name, value = part.split(":", 1)
        cls_name = name.strip()
        if not cls_name:
            raise ValueError(f"Invalid class multiplier token: '{part}'. Class name is empty.")
        try:
            mult = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid multiplier for class '{cls_name}': '{value.strip()}'.") from exc
        if not math.isfinite(mult) or mult <= 0:
            raise ValueError(f"Multiplier for class '{cls_name}' must be a positive finite number.")
        out[cls_name] = mult
    return out


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    q_clamped = min(max(float(q), 0.0), 1.0)
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * q_clamped
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(data[lo])
    frac = pos - lo
    return float(data[lo] * (1.0 - frac) + data[hi] * frac)


def _auto_head_cap_multipliers(
    bbox_count: dict[str, int],
    *,
    quantile: float,
    min_mult: float,
) -> dict[str, float]:
    if len(bbox_count) < 3:
        return {}
    counts = [int(v) for v in bbox_count.values() if int(v) > 0]
    if not counts:
        return {}
    threshold = _quantile(counts, quantile)
    if threshold <= 0:
        return {}
    median_count = _quantile(counts, 0.5)
    if median_count <= 0:
        return {}
    out: dict[str, float] = {}
    floor_mult = max(0.01, min(1.0, float(min_mult)))
    for cls_name, n_raw in bbox_count.items():
        n = max(1, int(n_raw))
        if float(n) <= threshold:
            continue
        # Smoothly reduce weights for head classes above quantile threshold.
        mult = math.sqrt(float(median_count) / float(n))
        out[cls_name] = max(floor_mult, min(1.0, mult))
    return out


def _apply_class_weight_multipliers(
    class_weights: dict[str, float],
    multipliers: dict[str, float],
) -> dict[str, float]:
    if not class_weights:
        return class_weights
    if not multipliers:
        return class_weights
    out: dict[str, float] = {}
    for cls_name, w in class_weights.items():
        out[cls_name] = float(w) * float(multipliers.get(cls_name, 1.0))
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
    eval_min_class_count: int = 0,
) -> list[tuple[str, str, str, list[str]]]:
    """
    Ensure val/test are not empty in output when possible.
    We keep the current logic as-is unless one of eval splits is empty.
    If needed, move a deterministic subset of balanced-train items to val/test
    targeting roughly 80/10/10 split.
    """
    out_train = list(balanced_train)
    passthrough_keys = {_source_image_key(img) for _s, img, _lbl, _cls in passthrough_items}

    def train_groups() -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, (s, img, _lbl, _cls) in enumerate(out_train):
            if s != "train":
                continue
            groups[_source_image_key(img)].append(i)
        return groups

    def movable_train_keys() -> list[str]:
        groups = train_groups()
        return [key for key in groups.keys() if key not in passthrough_keys]

    def move_key_to_split(key: str, target_split: str) -> int:
        moved = 0
        for i in train_groups().get(key, []):
            s, img, lbl, cls = out_train[i]
            if s != target_split:
                out_train[i] = (target_split, img, lbl, cls)
                moved += 1
        return moved

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
    # Keep at least one source key in train and move only split-safe keys.
    can_move = max(0, len(movable_train_keys()) - 1)
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
        keys = movable_train_keys()
        rng.shuffle(keys)

        pos = 0
        for _ in range(need_val):
            if pos >= len(keys):
                break
            key = keys[pos]
            pos += 1
            move_key_to_split(key, "val")
        for _ in range(need_test):
            if pos >= len(keys):
                break
            key = keys[pos]
            pos += 1
            move_key_to_split(key, "test")

        cur_val = sum(1 for s, *_ in out_train if s == "val") + val_count
        cur_test = sum(1 for s, *_ in out_train if s == "test") + test_count
        if val_count == 0 and cur_val == 0:
            print(
                "[WARN] eval_coverage: val is empty and could not be seeded; "
                "no split-safe train source keys are available."
            )
        if test_count == 0 and cur_test == 0:
            print(
                "[WARN] eval_coverage: test is empty and could not be seeded; "
                "no split-safe train source keys are available."
            )

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
            groups = train_groups()
            movable = [k for k in groups.keys() if k not in passthrough_keys]
            candidates: list[tuple[int, str]] = []
            for key in movable:
                idxs = groups.get(key, [])
                group_classes: set[str] = set()
                for i in idxs:
                    s, _img, _lbl, cls_names = out_train[i]
                    if s != "train":
                        continue
                    group_classes.update(cls_names)
                cover = len(group_classes & missing)
                if cover > 0:
                    candidates.append((cover, key))
            if not candidates:
                missing_preview = ", ".join(sorted(missing)[:8])
                movable_count = len(movable)
                print(
                    f"[WARN] eval_coverage: cannot cover missing classes in '{target_split}' "
                    f"without cross-split duplicate risk. Missing ({len(missing)}): {missing_preview}"
                    f"{' ...' if len(missing) > 8 else ''}; movable source keys: {movable_count}."
                )
                break
            # maximize coverage; tie-break with deterministic random jitter.
            max_cover = max(c for c, _ in candidates)
            best_keys = [k for c, k in candidates if c == max_cover]
            chosen_key = best_keys[rng_cov.randrange(len(best_keys))]
            covered: set[str] = set()
            for i in train_groups().get(chosen_key, []):
                s, _img, _lbl, cls_names = out_train[i]
                if s != "train":
                    continue
                covered.update(cls_names)
            move_key_to_split(chosen_key, target_split)
            missing -= covered

    # Optional eval-tail strengthening:
    # raise per-class bbox minimum in val/test by moving split-safe train groups.
    target_min = max(0, int(eval_min_class_count))
    if target_min <= 0:
        return out_train

    def split_bbox_counts(split_name: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for s, _img, _lbl, cls_names in out_train:
            if s != split_name:
                continue
            for c in cls_names:
                counts[c] += 1
        for s, _img, _lbl, cls_names in passthrough_items:
            if s != split_name:
                continue
            for c in cls_names:
                counts[c] += 1
        return counts

    for target_split in ("val", "test"):
        while train_count() > 1:
            current = split_bbox_counts(target_split)
            shortage = {c: target_min - int(current.get(c, 0)) for c in global_classes if int(current.get(c, 0)) < target_min}
            if not shortage:
                break
            groups = train_groups()
            movable = [k for k in groups.keys() if k not in passthrough_keys]
            best_key = None
            best_gain = 0
            for key in movable:
                idxs = groups.get(key, [])
                key_box_counts: dict[str, int] = defaultdict(int)
                for i in idxs:
                    s, _img, _lbl, cls_names = out_train[i]
                    if s != "train":
                        continue
                    for c in cls_names:
                        key_box_counts[c] += 1
                gain = 0
                for c, miss in shortage.items():
                    gain += min(int(miss), int(key_box_counts.get(c, 0)))
                if gain > best_gain:
                    best_gain = gain
                    best_key = key
            if best_key is None or best_gain <= 0:
                break
            move_key_to_split(best_key, target_split)
    return out_train


def _enforce_no_cross_split_duplicates(
    balanced_train: list[tuple[str, str, str, list[str]]],
    passthrough_items: list[tuple[str, str, str, list[str]]],
) -> tuple[list[tuple[str, str, str, list[str]]], list[tuple[str, str, str, list[str]]], int]:
    """
    Force each source image key to belong to exactly one split globally.
    Priority: train > val > test.
    """
    split_priority = {"train": 0, "val": 1, "test": 2}
    grouped: dict[str, list[str]] = defaultdict(list)
    for s, img, _lbl, _cls in balanced_train:
        grouped[_source_image_key(img)].append(s)
    for s, img, _lbl, _cls in passthrough_items:
        grouped[_source_image_key(img)].append(s)

    winner_by_key: dict[str, str] = {}
    for key, splits in grouped.items():
        winner_by_key[key] = min(splits, key=lambda s: split_priority.get(s, 99))

    changed = 0
    new_balanced: list[tuple[str, str, str, list[str]]] = []
    for s, img, lbl, cls in balanced_train:
        win = winner_by_key[_source_image_key(img)]
        if s != win:
            changed += 1
        new_balanced.append((win, img, lbl, cls))

    new_passthrough: list[tuple[str, str, str, list[str]]] = []
    for s, img, lbl, cls in passthrough_items:
        win = winner_by_key[_source_image_key(img)]
        if s != win:
            changed += 1
        new_passthrough.append((win, img, lbl, cls))

    return new_balanced, new_passthrough, changed


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
    try:
        manual_class_multipliers = _parse_class_weight_multiplier(args.class_weight_multiplier)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    unknown_multiplier_classes = [c for c in manual_class_multipliers if c not in class_map]
    if unknown_multiplier_classes:
        print(
            "[ERROR] Unknown classes in --class-weight-multiplier: "
            + ", ".join(sorted(unknown_multiplier_classes))
        )
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

    auto_class_multipliers: dict[str, float] = {}
    effective_class_multipliers: dict[str, float] = {}

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
            auto_class_multipliers: dict[str, float] = {}
            if bool(getattr(args, "auto_head_cap", True)):
                auto_class_multipliers = _auto_head_cap_multipliers(
                    bbox_count,
                    quantile=float(args.auto_head_cap_quantile),
                    min_mult=float(args.auto_head_cap_min_mult),
                )
            effective_class_multipliers: dict[str, float] = dict(auto_class_multipliers)
            for cls_name, mult in manual_class_multipliers.items():
                effective_class_multipliers[cls_name] = (
                    float(effective_class_multipliers.get(cls_name, 1.0)) * float(mult)
                )
            class_w = _apply_class_weight_multipliers(class_w, effective_class_multipliers)
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
            eval_min_class_count=int(getattr(args, "eval_min_class_count", 0)),
        )
    balanced_train, passthrough_items, forced_reassignments = _enforce_no_cross_split_duplicates(
        balanced_train,
        passthrough_items,
    )
    if forced_reassignments > 0:
        print(
            f"[WARN] balance: reassigned {forced_reassignments} items to prevent "
            "cross-split duplicates (priority train > val > test)."
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
                    "preset": args.preset,
                    "strategy": args.strategy,
                    "selected_classes": sorted(selected_classes),
                    "single_class": args.single_class,
                    "classes": args.classes,
                    "train_input": len(train_items),
                    "train_output": len(balanced_train),
                    "seed": args.seed,
                    "target": args.target,
                    "max_ratio": args.max_ratio,
                    "min_count": args.min_count,
                    "replacement": args.replacement,
                    "weight_mode": args.weight_mode,
                    "beta": args.beta,
                    "class_weight_multiplier": args.class_weight_multiplier,
                    "image_weight_agg": args.image_weight_agg,
                    "weight_clip_min": args.weight_clip_min,
                    "weight_clip_max": args.weight_clip_max,
                    "auto_head_cap": bool(args.auto_head_cap),
                    "auto_head_cap_quantile": args.auto_head_cap_quantile,
                    "auto_head_cap_min_mult": args.auto_head_cap_min_mult,
                    "applied_auto_head_cap_multipliers": dict(sorted(auto_class_multipliers.items())),
                    "applied_manual_class_multipliers": dict(sorted(manual_class_multipliers.items())),
                    "applied_effective_class_multipliers": dict(sorted(effective_class_multipliers.items())),
                    "rfs_thresh": args.rfs_thresh,
                    "rfs_power": args.rfs_power,
                    "max_repeat_per_image": args.max_repeat_per_image,
                    "eval_coverage": bool(args.eval_coverage),
                    "eval_min_class_count": int(args.eval_min_class_count),
                    "emit_train_config": bool(args.emit_train_config),
                    "emit_balance_report": bool(args.emit_balance_report),
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

