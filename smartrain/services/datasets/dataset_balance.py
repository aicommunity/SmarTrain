from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice, prompt_multi_choice_csv, prompt_text, prompt_yes_no
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    sorted_class_names_union_from_catalog,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_cli_common import (
    detect_split_from_path,
    update_datasets_sidecar,
)
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

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
    # Default for hybrid-aug: constrained growth + tail-first budget + head trim.
    "hybrid-aug-tail-budget": {
        "strategy": "hybrid-aug",
        "aug_class_aware_geo": True,
        "aug_total_bbox_cap_mult": 1.10,
        "aug_budget_tail_first": True,
        "aug_budget_tail_gamma": 1.0,
        "train_head_bbox_undersample": "median-factor",
        "train_head_bbox_cap_mult": 5.0,
        "eval_head_bbox_undersample": "median-factor",
        "eval_head_bbox_cap_mult": 8.0,
        "eval_head_bbox_min_count": 30,
        "eval_head_bbox_max_remove_frac": 0.35,
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
            "hybrid-default: recommended general-purpose balance; "
            "hybrid-aug-tail-budget: constrained-growth hybrid-aug with tail-first budget."
        ),
    )
    p.add_argument(
        "--strategy",
        choices=("copy", "oversample", "undersample", "class-aware", "weights", "rfs", "hybrid", "hybrid-aug"),
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
    p.add_argument(
        "--aug-preset",
        choices=("geo-photo", "conveyor-lite"),
        default="geo-photo",
        help="For strategy hybrid-aug: augment preset after hybrid (train split only).",
    )
    p.add_argument(
        "--aug-enable-bbox-copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For hybrid-aug: add bbox-copy to augment (default off).",
    )
    p.add_argument(
        "--keep-hybrid-intermediate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For hybrid-aug: keep the intermediate hybrid-only dataset after augment (default: delete).",
    )
    p.add_argument(
        "--aug-class-aware-geo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For hybrid-aug: scale flip / photometric / conveyor attempt rates by inverse class frequency "
            "(passes through to augment). Disable with --no-aug-class-aware-geo."
        ),
    )
    p.add_argument(
        "--aug-total-bbox-cap-mult",
        type=float,
        default=0.0,
        help="For hybrid-aug: if >0, cap train bbox total after augment (see augment --aug-total-bbox-cap-mult).",
    )
    p.add_argument(
        "--aug-budget-tail-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For hybrid-aug with bbox cap: pass --aug-budget-tail-first / --no-aug-budget-tail-first to augment.",
    )
    p.add_argument(
        "--aug-budget-tail-gamma",
        type=float,
        default=1.0,
        help="For hybrid-aug with bbox cap: tail-priority exponent passed to augment (default 1.0).",
    )
    p.add_argument(
        "--train-head-bbox-undersample",
        choices=("off", "median-factor"),
        default="off",
        help="Optional: trim head-class bbox counts toward median*cap after hybrid sampling.",
    )
    p.add_argument(
        "--train-head-bbox-cap-mult",
        type=float,
        default=5.0,
        help="With --train-head-bbox-undersample median-factor: cap = floor(mult * median bbox count per class).",
    )
    p.add_argument(
        "--eval-head-bbox-undersample",
        choices=("off", "median-factor"),
        default="off",
        help="Optional: conservative head-class bbox trimming on val/test splits after split coverage adjustments.",
    )
    p.add_argument(
        "--eval-head-bbox-cap-mult",
        type=float,
        default=8.0,
        help="With --eval-head-bbox-undersample median-factor: cap = floor(mult * median bbox count per class) per eval split.",
    )
    p.add_argument(
        "--eval-head-bbox-min-count",
        type=int,
        default=30,
        help="Do not trim eval classes with fewer than this bbox count in the split.",
    )
    p.add_argument(
        "--eval-head-bbox-max-remove-frac",
        type=float,
        default=0.35,
        help="Maximum removable fraction [0..1] per class in eval split when head trimming is enabled.",
    )
    return p


def _detect_split(images_path: str) -> str:
    return detect_split_from_path(images_path, prefer_valid_name=False)


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
        ["copy", "oversample", "undersample", "class-aware", "weights", "rfs", "hybrid", "hybrid-aug"],
        default=args.strategy,
    )
    if args.strategy == "hybrid-aug":
        # Apply product defaults in interactive mode as a starting point.
        _apply_hybrid_aug_default_mode(args, set())
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
    if args.strategy == "hybrid-aug":
        args.eval_min_class_count = int(
            prompt_text(
                "Eval minimum bbox per class (--eval-min-class-count, 0=off)",
                default=str(getattr(args, "eval_min_class_count", 0)),
            ).strip()
            or str(getattr(args, "eval_min_class_count", 0))
        )
        args.aug_preset = prompt_choice(
            "Augment preset (--aug-preset)",
            ["geo-photo", "conveyor-lite"],
            default=str(getattr(args, "aug_preset", "geo-photo")),
        )
        args.aug_class_aware_geo = prompt_yes_no(
            "Class-aware geo/photo augment (--aug-class-aware-geo)?",
            default=bool(getattr(args, "aug_class_aware_geo", True)),
        )
        args.aug_total_bbox_cap_mult = float(
            prompt_text(
                "Total train bbox cap multiplier (--aug-total-bbox-cap-mult, 0=off)",
                default=str(getattr(args, "aug_total_bbox_cap_mult", 0.0)),
            ).strip()
            or str(getattr(args, "aug_total_bbox_cap_mult", 0.0))
        )
        if float(args.aug_total_bbox_cap_mult) > 0:
            args.aug_budget_tail_first = prompt_yes_no(
                "Tail-first budget ordering (--aug-budget-tail-first)?",
                default=bool(getattr(args, "aug_budget_tail_first", True)),
            )
            args.aug_budget_tail_gamma = float(
                prompt_text(
                    "Tail priority gamma (--aug-budget-tail-gamma)",
                    default=str(getattr(args, "aug_budget_tail_gamma", 1.0)),
                ).strip()
                or str(getattr(args, "aug_budget_tail_gamma", 1.0))
            )
        args.train_head_bbox_undersample = prompt_choice(
            "Head bbox undersample (--train-head-bbox-undersample)",
            ["off", "median-factor"],
            default=str(getattr(args, "train_head_bbox_undersample", "off")),
        )
        if str(args.train_head_bbox_undersample) == "median-factor":
            args.train_head_bbox_cap_mult = float(
                prompt_text(
                    "Head bbox cap multiplier (--train-head-bbox-cap-mult)",
                    default=str(getattr(args, "train_head_bbox_cap_mult", 5.0)),
                ).strip()
                or str(getattr(args, "train_head_bbox_cap_mult", 5.0))
            )
        args.eval_head_bbox_undersample = prompt_choice(
            "Eval head bbox undersample (--eval-head-bbox-undersample)",
            ["off", "median-factor"],
            default=str(getattr(args, "eval_head_bbox_undersample", "median-factor")),
        )
        if str(args.eval_head_bbox_undersample) == "median-factor":
            args.eval_head_bbox_cap_mult = float(
                prompt_text(
                    "Eval head bbox cap multiplier (--eval-head-bbox-cap-mult)",
                    default=str(getattr(args, "eval_head_bbox_cap_mult", 8.0)),
                ).strip()
                or str(getattr(args, "eval_head_bbox_cap_mult", 8.0))
            )
            args.eval_head_bbox_min_count = int(
                prompt_text(
                    "Eval minimum class bbox before trimming (--eval-head-bbox-min-count)",
                    default=str(getattr(args, "eval_head_bbox_min_count", 30)),
                ).strip()
                or str(getattr(args, "eval_head_bbox_min_count", 30))
            )
            args.eval_head_bbox_max_remove_frac = float(
                prompt_text(
                    "Eval maximum removable fraction (--eval-head-bbox-max-remove-frac)",
                    default=str(getattr(args, "eval_head_bbox_max_remove_frac", 0.35)),
                ).strip()
                or str(getattr(args, "eval_head_bbox_max_remove_frac", 0.35))
            )
        args.aug_enable_bbox_copy = prompt_yes_no(
            "Enable bbox copy-paste (--aug-enable-bbox-copy)?",
            default=bool(getattr(args, "aug_enable_bbox_copy", False)),
        )
        args.keep_hybrid_intermediate = prompt_yes_no(
            "Keep intermediate hybrid dataset (--keep-hybrid-intermediate)?",
            default=bool(getattr(args, "keep_hybrid_intermediate", False)),
        )
    args.dry_run = prompt_yes_no("Do dry-run (--dry-run)?", default=bool(args.dry_run))


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    class_map: dict[str, int],
    target_dir: str,
    output_hash: str,
) -> None:
    update_datasets_sidecar(
        layout=layout,
        output_key=output_key,
        class_map=class_map,
        target_dir=target_dir,
        output_hash=output_hash,
    )


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
        "aug_class_aware_geo": "--aug-class-aware-geo",
        "aug_total_bbox_cap_mult": "--aug-total-bbox-cap-mult",
        "aug_budget_tail_first": "--aug-budget-tail-first",
        "aug_budget_tail_gamma": "--aug-budget-tail-gamma",
        "train_head_bbox_undersample": "--train-head-bbox-undersample",
        "train_head_bbox_cap_mult": "--train-head-bbox-cap-mult",
        "eval_head_bbox_undersample": "--eval-head-bbox-undersample",
        "eval_head_bbox_cap_mult": "--eval-head-bbox-cap-mult",
        "eval_head_bbox_min_count": "--eval-head-bbox-min-count",
        "eval_head_bbox_max_remove_frac": "--eval-head-bbox-max-remove-frac",
    }
    for attr, value in preset_cfg.items():
        flag = flag_for_attr.get(attr)
        if flag and flag in provided_flags:
            continue
        setattr(args, attr, value)


def _apply_hybrid_aug_default_mode(args: argparse.Namespace, provided_flags: set[str]) -> None:
    """Default mode for hybrid-aug unless explicitly overridden by CLI flags."""
    if str(getattr(args, "strategy", "")) != "hybrid-aug":
        return
    mode_defaults: dict[str, object] = BALANCE_PRESETS["hybrid-aug-tail-budget"]
    flags_by_attr: dict[str, set[str]] = {
        "aug_class_aware_geo": {"--aug-class-aware-geo", "--no-aug-class-aware-geo"},
        "aug_total_bbox_cap_mult": {"--aug-total-bbox-cap-mult"},
        "aug_budget_tail_first": {"--aug-budget-tail-first", "--no-aug-budget-tail-first"},
        "aug_budget_tail_gamma": {"--aug-budget-tail-gamma"},
        "train_head_bbox_undersample": {"--train-head-bbox-undersample"},
        "train_head_bbox_cap_mult": {"--train-head-bbox-cap-mult"},
        "eval_head_bbox_undersample": {"--eval-head-bbox-undersample"},
        "eval_head_bbox_cap_mult": {"--eval-head-bbox-cap-mult"},
        "eval_head_bbox_min_count": {"--eval-head-bbox-min-count"},
        "eval_head_bbox_max_remove_frac": {"--eval-head-bbox-max-remove-frac"},
    }
    for attr, value in mode_defaults.items():
        if attr == "strategy":
            continue
        flags = flags_by_attr.get(attr, set())
        if flags and any(f in provided_flags for f in flags):
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


def _remove_dataset_from_workspace_catalog(layout: WorkspaceLayout, dataset_key: str) -> None:
    info_path = layout.work_datasets_info_path()
    if not os.path.isfile(info_path):
        return
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or dataset_key not in data:
        return
    del data[dataset_key]
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _build_hybrid_aug_augment_argv(
    *,
    workspace: str,
    intermediate_dataset: str,
    final_base: str,
    seed: int,
    preset: str,
    aug_enable_bbox_copy: bool,
    aug_class_aware_geo: bool,
    aug_total_bbox_cap_mult: float,
    aug_budget_tail_first: bool,
    aug_budget_tail_gamma: float,
) -> list[str]:
    argv = [
        "--workspace",
        workspace,
        "--dataset",
        intermediate_dataset,
        "--output-name",
        final_base,
        "--seed",
        str(seed),
        "--splits",
        "train",
        "--enable-flip",
        "--enable-photometric",
        "--enable-center-rotate",
        "--center-rotate-anchor",
        "center",
        "--center-rotate-deg",
        "5",
        "--rotate-copies",
        "1",
    ]
    if preset == "conveyor-lite":
        argv.append("--enable-conveyor")
    if aug_enable_bbox_copy:
        argv.append("--enable-bbox-copy")
    if aug_class_aware_geo:
        argv.append("--aug-class-aware-geo")
    else:
        argv.append("--no-aug-class-aware-geo")
    if float(aug_total_bbox_cap_mult) > 0:
        argv.extend(["--aug-total-bbox-cap-mult", str(float(aug_total_bbox_cap_mult))])
        if aug_budget_tail_first:
            argv.append("--aug-budget-tail-first")
        else:
            argv.append("--no-aug-budget-tail-first")
        argv.extend(["--aug-budget-tail-gamma", str(float(aug_budget_tail_gamma))])
    return argv


def _read_label_text_lines(lbl_path: str) -> list[str]:
    if not os.path.isfile(lbl_path):
        return []
    with open(lbl_path, "r", encoding="utf-8") as f:
        return f.readlines()


def _line_class_id(line: str) -> int | None:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def _head_bbox_undersample_balanced_train(
    balanced_train: list[tuple[str, str, str, list[str]]],
    *,
    id_to_name: dict[int, str],
    cap_mult: float,
    seed: int,
    selected_classes: set[str],
) -> tuple[list[tuple[str, str, str, list[str]]], dict[str, object], list[set[int]]]:
    """Stratified removal of excess bbox lines for head classes (plan §7)."""
    empty_skips = [set() for _ in balanced_train]
    counts: Counter[str] = Counter()
    for _s, _img, _lbl, cls_names in balanced_train:
        for c in cls_names:
            if selected_classes and c not in selected_classes:
                continue
            counts[c] += 1
    if not counts:
        return balanced_train, {}, empty_skips

    vals_sorted = sorted(int(v) for v in counts.values())
    median_bbox = int(_quantile(vals_sorted, 0.5))
    if median_bbox <= 0:
        return balanced_train, {}, empty_skips

    cap_target = max(0, int(math.floor(float(cap_mult) * float(median_bbox))))
    omit: dict[int, set[int]] = {}

    for cls_name, n_raw in counts.items():
        if int(n_raw) <= cap_target:
            continue
        excess = int(n_raw) - cap_target
        pool: list[tuple[int, int]] = []
        for ti, (_sp, img, lbl, _cn) in enumerate(balanced_train):
            lines = _read_label_text_lines(lbl)
            for li, raw in enumerate(lines):
                cid = _line_class_id(raw)
                if cid is None:
                    continue
                name = id_to_name.get(cid, f"id_{cid}")
                if selected_classes and name not in selected_classes:
                    continue
                if name != cls_name:
                    continue
                pool.append((ti, li))
        pool.sort(key=lambda p: (Path(balanced_train[p[0]][1]).stem, p[0], p[1]))
        if not pool or excess <= 0:
            continue
        g_sz = min(32, max(1, len(pool)))
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ti, li in pool:
            stem = Path(balanced_train[ti][1]).stem
            h = int(hashlib.md5(f"{seed}:{stem}".encode()).hexdigest(), 16)
            groups[h % g_sz].append((ti, li))
        removed_pairs: set[tuple[int, int]] = set()
        while len(removed_pairs) < excess and groups:
            best_g: int | None = None
            best_sz = -1
            for g in sorted(groups.keys()):
                sz = sum(1 for pair in groups[g] if pair not in removed_pairs)
                if sz > best_sz:
                    best_sz = sz
                    best_g = g
            if best_g is None or best_sz <= 0:
                break
            for pair in groups[best_g]:
                if pair not in removed_pairs:
                    removed_pairs.add(pair)
                    ti, li = pair
                    omit.setdefault(ti, set()).add(li)
                    break

    dropped_indices = set()
    for ti, it in enumerate(balanced_train):
        lines = _read_label_text_lines(it[2])
        kept_is = [j for j in range(len(lines)) if j not in omit.get(ti, set())]
        if not kept_is:
            dropped_indices.add(ti)

    after_counts: Counter[str] = Counter()
    for ti, (_s, _img, lbl, _cn) in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(lbl)
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            name = id_to_name.get(cid, f"id_{cid}")
            if selected_classes and name not in selected_classes:
                continue
            after_counts[name] += 1

    per_class: dict[str, dict[str, int]] = {}
    for cls_name, before in counts.items():
        after = int(after_counts.get(cls_name, 0))
        per_class[str(cls_name)] = {"before": int(before), "after": after, "removed": max(0, int(before) - after)}

    new_train: list[tuple[str, str, str, list[str]]] = []
    for ti, it in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(it[2])
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        new_cls: list[str] = []
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            new_cls.append(id_to_name.get(cid, f"id_{cid}"))
        new_train.append((it[0], it[1], it[2], new_cls))

    stats: dict[str, object] = {
        "mode": "median-factor",
        "cap_mult": float(cap_mult),
        "median_bbox_per_class": median_bbox,
        "cap_target": cap_target,
        "per_class": per_class,
    }
    label_skips: list[set[int]] = []
    for ti, _it in enumerate(balanced_train):
        if ti in dropped_indices:
            continue
        label_skips.append(set(omit.get(ti, set())))
    return new_train, stats, label_skips


def _head_bbox_undersample_items(
    items: list[tuple[str, str, str, list[str]]],
    *,
    id_to_name: dict[int, str],
    cap_mult: float,
    seed: int,
    selected_classes: set[str],
    min_class_count: int = 0,
    max_remove_frac: float = 1.0,
) -> tuple[list[tuple[str, str, str, list[str]]], dict[str, object], list[set[int]]]:
    """Generalized bbox head-trimming with conservative guards for eval/train subsets."""
    empty_skips = [set() for _ in items]
    counts: Counter[str] = Counter()
    for _s, _img, _lbl, cls_names in items:
        for c in cls_names:
            if selected_classes and c not in selected_classes:
                continue
            counts[c] += 1
    if not counts:
        return items, {}, empty_skips

    vals_sorted = sorted(int(v) for v in counts.values())
    median_bbox = int(_quantile(vals_sorted, 0.5))
    if median_bbox <= 0:
        return items, {}, empty_skips

    cap_target = max(0, int(math.floor(float(cap_mult) * float(median_bbox))))
    omit: dict[int, set[int]] = {}
    min_cls = max(0, int(min_class_count))
    max_frac = max(0.0, min(1.0, float(max_remove_frac)))

    for cls_name, n_raw in counts.items():
        before = int(n_raw)
        if before <= cap_target or before <= min_cls:
            continue
        excess = before - cap_target
        # Conservative guard: do not remove more than configured fraction.
        excess = min(excess, int(math.floor(float(before) * max_frac)))
        # Conservative guard: keep at least min_cls instances in split.
        excess = min(excess, max(0, before - min_cls))
        if excess <= 0:
            continue

        pool: list[tuple[int, int]] = []
        for ti, (_sp, img, lbl, _cn) in enumerate(items):
            lines = _read_label_text_lines(lbl)
            for li, raw in enumerate(lines):
                cid = _line_class_id(raw)
                if cid is None:
                    continue
                name = id_to_name.get(cid, f"id_{cid}")
                if selected_classes and name not in selected_classes:
                    continue
                if name != cls_name:
                    continue
                pool.append((ti, li))
        pool.sort(key=lambda p: (Path(items[p[0]][1]).stem, p[0], p[1]))
        if not pool:
            continue
        g_sz = min(32, max(1, len(pool)))
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ti, li in pool:
            stem = Path(items[ti][1]).stem
            h = int(hashlib.md5(f"{seed}:{stem}".encode()).hexdigest(), 16)
            groups[h % g_sz].append((ti, li))
        removed_pairs: set[tuple[int, int]] = set()
        while len(removed_pairs) < excess and groups:
            best_g: int | None = None
            best_sz = -1
            for g in sorted(groups.keys()):
                sz = sum(1 for pair in groups[g] if pair not in removed_pairs)
                if sz > best_sz:
                    best_sz = sz
                    best_g = g
            if best_g is None or best_sz <= 0:
                break
            for pair in groups[best_g]:
                if pair not in removed_pairs:
                    removed_pairs.add(pair)
                    ti, li = pair
                    omit.setdefault(ti, set()).add(li)
                    break

    dropped_indices = set()
    for ti, it in enumerate(items):
        lines = _read_label_text_lines(it[2])
        kept_is = [j for j in range(len(lines)) if j not in omit.get(ti, set())]
        if not kept_is:
            dropped_indices.add(ti)

    after_counts: Counter[str] = Counter()
    for ti, (_s, _img, lbl, _cn) in enumerate(items):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(lbl)
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            name = id_to_name.get(cid, f"id_{cid}")
            if selected_classes and name not in selected_classes:
                continue
            after_counts[name] += 1

    per_class: dict[str, dict[str, int]] = {}
    for cls_name, before in counts.items():
        after = int(after_counts.get(cls_name, 0))
        per_class[str(cls_name)] = {"before": int(before), "after": after, "removed": max(0, int(before) - after)}

    new_items: list[tuple[str, str, str, list[str]]] = []
    for ti, it in enumerate(items):
        if ti in dropped_indices:
            continue
        lines = _read_label_text_lines(it[2])
        kept_lines = [lines[j] for j in range(len(lines)) if j not in omit.get(ti, set())]
        new_cls: list[str] = []
        for raw in kept_lines:
            cid = _line_class_id(raw)
            if cid is None:
                continue
            new_cls.append(id_to_name.get(cid, f"id_{cid}"))
        new_items.append((it[0], it[1], it[2], new_cls))

    stats: dict[str, object] = {
        "mode": "median-factor",
        "cap_mult": float(cap_mult),
        "median_bbox_per_class": median_bbox,
        "cap_target": cap_target,
        "min_class_count": min_cls,
        "max_remove_frac": max_frac,
        "per_class": per_class,
    }
    label_skips: list[set[int]] = []
    for ti, _it in enumerate(items):
        if ti in dropped_indices:
            continue
        label_skips.append(set(omit.get(ti, set())))
    return new_items, stats, label_skips


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_balance_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    provided_flags = _provided_flags(argv)
    _apply_preset_defaults(args, provided_flags)
    _apply_hybrid_aug_default_mode(args, provided_flags)
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    catalog = load_datasets_catalog(layout)
    if not catalog:
        print(EMPTY_DATASETS_INFO_MESSAGE)
        return

    interactive_used = try_prompt_dataset_interactive(
        args=args,
        argv=argv,
        fill=lambda: _interactive_fill(
            args,
            sorted(catalog.keys()),
            sorted_class_names_union_from_catalog(catalog),
        ),
    )
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
            if args.strategy in ("hybrid", "hybrid-aug"):
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

    is_hybrid_aug = args.strategy == "hybrid-aug"
    if is_hybrid_aug:
        final_base = args.output_name or f"{args.dataset}_balanced_aug"
        intermediate_base = f"{final_base}__hybrid"
        out_name = next_dataset_name(layout.datasets, intermediate_base)
    else:
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

    head_bbox_stats: dict[str, object] | None = None
    train_label_skips: list[set[int]] | None = None
    if getattr(args, "train_head_bbox_undersample", "off") == "median-factor":
        balanced_train, head_bbox_stats, skips_list = _head_bbox_undersample_balanced_train(
            balanced_train,
            id_to_name=id_to_name,
            cap_mult=float(getattr(args, "train_head_bbox_cap_mult", 5.0)),
            seed=int(args.seed),
            selected_classes=selected_classes,
        )
        train_label_skips = skips_list
    eval_head_bbox_stats: dict[str, object] | None = None
    passthrough_label_skips: list[set[int]] | None = None
    if getattr(args, "eval_head_bbox_undersample", "off") == "median-factor":
        grouped_eval: dict[str, list[tuple[str, str, str, list[str]]]] = defaultdict(list)
        for it in passthrough_items:
            grouped_eval[it[0]].append(it)
        new_passthrough: list[tuple[str, str, str, list[str]]] = []
        new_passthrough_skips: list[set[int]] = []
        split_stats: dict[str, dict[str, object]] = {}
        for split in ("val", "test"):
            split_items = grouped_eval.get(split, [])
            if not split_items:
                continue
            split_new, split_stat, split_skips = _head_bbox_undersample_items(
                split_items,
                id_to_name=id_to_name,
                cap_mult=float(getattr(args, "eval_head_bbox_cap_mult", 8.0)),
                seed=int(args.seed) + (17 if split == "val" else 29),
                selected_classes=selected_classes,
                min_class_count=int(getattr(args, "eval_head_bbox_min_count", 30)),
                max_remove_frac=float(getattr(args, "eval_head_bbox_max_remove_frac", 0.35)),
            )
            if split_stat:
                split_stats[split] = split_stat
            new_passthrough.extend(split_new)
            new_passthrough_skips.extend(split_skips)
        for split in sorted(grouped_eval.keys()):
            if split in ("val", "test"):
                continue
            split_items = grouped_eval.get(split, [])
            new_passthrough.extend(split_items)
            new_passthrough_skips.extend([set() for _ in split_items])
        passthrough_items = new_passthrough
        passthrough_label_skips = new_passthrough_skips
        eval_head_bbox_stats = {
            "mode": "median-factor",
            "cap_mult": float(getattr(args, "eval_head_bbox_cap_mult", 8.0)),
            "min_class_count": int(getattr(args, "eval_head_bbox_min_count", 30)),
            "max_remove_frac": float(getattr(args, "eval_head_bbox_max_remove_frac", 0.35)),
            "splits": split_stats,
        }

    if args.dry_run:
        print(f"[OK] dry-run: strategy={args.strategy}, train_in={len(train_items)}, train_out={len(balanced_train)}, output={out_name}")
        if is_hybrid_aug:
            print("[INFO] hybrid-aug: skipping augment (dry-run)")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)

    def _copy_items(
        items: list[tuple[str, str, str, list[str]]],
        *,
        suffix: str = "",
        dest_root: str | None = None,
        label_skips: list[set[int]] | None = None,
    ) -> None:
        root_dir = dest_root or out_dir
        idx = 1
        for item_idx, (split, img, lbl, _) in enumerate(items):
            ext = os.path.splitext(img)[1]
            stem = f"{Path(img).stem}{suffix}_{idx}" if suffix else f"{Path(img).stem}_{idx}"
            dst_img = os.path.join(root_dir, split, "images", f"{stem}{ext}")
            dst_lbl = os.path.join(root_dir, split, "labels", f"{stem}.txt")
            shutil.copy2(img, dst_img)
            use_skip = (
                label_skips is not None
                and item_idx < len(label_skips)
                and len(label_skips[item_idx]) > 0
            )
            if use_skip:
                lines = _read_label_text_lines(lbl)
                kept = "".join(lines[j] for j in range(len(lines)) if j not in label_skips[item_idx])
                Path(dst_lbl).write_text(kept, encoding="utf-8")
            elif os.path.isfile(lbl):
                shutil.copy2(lbl, dst_lbl)
            else:
                Path(dst_lbl).write_text("", encoding="utf-8")
            idx += 1

    _copy_items(balanced_train, suffix="bal", label_skips=train_label_skips)
    _copy_items(passthrough_items, label_skips=passthrough_label_skips)

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

    hybrid_intermediate_name_for_manifest: str | None = None
    aug_argv_list: list[str] | None = None
    bbox_before_augment: int | None = None
    bbox_after_augment: int | None = None
    if is_hybrid_aug:
        from smartrain.services.datasets.dataset_augment import main as augment_main, sum_train_bbox_disk

        saved_intermediate_name = out_name
        final_base = args.output_name or f"{args.dataset}_balanced_aug"
        aug_argv_list = _build_hybrid_aug_augment_argv(
            workspace=root,
            intermediate_dataset=saved_intermediate_name,
            final_base=final_base,
            seed=int(args.seed),
            preset=str(args.aug_preset),
            aug_enable_bbox_copy=bool(getattr(args, "aug_enable_bbox_copy", False)),
            aug_class_aware_geo=bool(getattr(args, "aug_class_aware_geo", True)),
            aug_total_bbox_cap_mult=float(getattr(args, "aug_total_bbox_cap_mult", 0.0)),
            aug_budget_tail_first=bool(getattr(args, "aug_budget_tail_first", True)),
            aug_budget_tail_gamma=float(getattr(args, "aug_budget_tail_gamma", 1.0)),
        )
        pred_final = next_dataset_name(layout.datasets, final_base)
        bbox_before_augment = int(sum_train_bbox_disk(out_dir))
        augment_main(aug_argv_list)
        final_out_dir = os.path.join(layout.datasets, pred_final)
        if not os.path.isdir(final_out_dir):
            print(
                "[ERROR] balance: augment did not create the expected output dataset; "
                "intermediate dataset left in place for recovery."
            )
            return
        if not bool(getattr(args, "keep_hybrid_intermediate", False)):
            try:
                shutil.rmtree(out_dir)
            except OSError as exc:
                print(f"[WARN] balance: could not remove intermediate dataset directory: {exc}")
            _remove_dataset_from_workspace_catalog(layout, saved_intermediate_name)
            hybrid_intermediate_name_for_manifest = None
        else:
            hybrid_intermediate_name_for_manifest = saved_intermediate_name
        out_dir = final_out_dir
        out_name = pred_final
        out_hash = calculate_dataset_hash(out_dir)
        bbox_after_augment = int(sum_train_bbox_disk(out_dir))
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
                    "aug_preset": getattr(args, "aug_preset", "geo-photo"),
                    "aug_enable_bbox_copy": bool(getattr(args, "aug_enable_bbox_copy", False)),
                    "keep_hybrid_intermediate": bool(getattr(args, "keep_hybrid_intermediate", False)),
                    "train_head_bbox_undersample": getattr(args, "train_head_bbox_undersample", "off"),
                    "train_head_bbox_cap_mult": float(getattr(args, "train_head_bbox_cap_mult", 5.0)),
                    "eval_head_bbox_undersample": getattr(args, "eval_head_bbox_undersample", "off"),
                    "eval_head_bbox_cap_mult": float(getattr(args, "eval_head_bbox_cap_mult", 8.0)),
                    "eval_head_bbox_min_count": int(getattr(args, "eval_head_bbox_min_count", 30)),
                    "eval_head_bbox_max_remove_frac": float(getattr(args, "eval_head_bbox_max_remove_frac", 0.35)),
                    "hybrid_intermediate_name": hybrid_intermediate_name_for_manifest,
                    "output_dataset_name": out_name,
                    "post_augment": (
                        {
                            "preset": str(args.aug_preset),
                            "aug_enable_bbox_copy": bool(getattr(args, "aug_enable_bbox_copy", False)),
                            "class_aware_geo": bool(getattr(args, "aug_class_aware_geo", True)),
                            "total_bbox_cap_mult": float(getattr(args, "aug_total_bbox_cap_mult", 0.0)),
                            "budget_tail_first": bool(getattr(args, "aug_budget_tail_first", True)),
                            "budget_tail_gamma": float(getattr(args, "aug_budget_tail_gamma", 1.0)),
                            "train_bbox_sum_before_augment": bbox_before_augment,
                            "train_bbox_sum_after_augment": bbox_after_augment,
                            "argv_summary": aug_argv_list or [],
                        }
                        if is_hybrid_aug
                        else None
                    ),
                    "head_bbox_undersample": head_bbox_stats,
                    "eval_head_bbox_undersample_stats": eval_head_bbox_stats,
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

