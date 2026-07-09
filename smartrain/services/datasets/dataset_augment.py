from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import tempfile
from collections import defaultdict
import random
import shutil
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import (
    prompt_choice,
    prompt_multi_choice_csv,
    prompt_prefilled_text,
    prompt_text,
    prompt_yes_no,
)
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_hash import calculate_dataset_hash
from smartrain.services.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.services.datasets.noise_augment import build_conveyor_noise_transform, parse_noise_types
from smartrain.services.datasets.yolo_image_rotate import apply_orthogonal_rotate
from smartrain.services.datasets.yolo_augment_geom import (
    apply_albumentations_to_labels,
    count_label_instances,
    infer_label_kind,
    labels_to_legacy_tuples,
    legacy_tuples_to_serialized,
    read_augment_label_file,
    resolve_label_kind,
    rotate_labels_with_matrix,
    write_augment_label_file,
)
from smartrain.services.datasets.yolo_labels import YoloLabel, read_yolo_labels, serialize_yolo_labels
from smartrain.services.datasets.dataset_cli_catalog import (
    EMPTY_DATASETS_INFO_MESSAGE,
    load_datasets_catalog,
    sorted_class_names_for_dataset,
    try_prompt_dataset_interactive,
)
from smartrain.services.datasets.dataset_cli_common import (
    detect_split_from_path,
    update_datasets_sidecar,
)
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "valid", "test": "test"}
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_ROI_MODEL_CACHE: dict[str, YOLO] = {}

AUGMENT_PRESETS: dict[str, dict[str, object]] = {
    "augment-tail-safe": {
        "aug_class_aware_geo": True,
        "aug_total_bbox_cap_mult": 1.10,
        "aug_budget_tail_first": True,
        "aug_budget_tail_gamma": 1.0,
    },
}


def build_augment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Offline augmentation of a dataset into a new datasets/<name>")
    # --- source ---
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Name of source dataset from datasets_info.json")
    p.add_argument(
        "--label-type",
        choices=("auto", "bbox", "segment"),
        default="auto",
        help="Label interpretation for augment: auto-detect, force bbox, or force segment (polygons).",
    )
    p.add_argument("--output-name", type=str, default=None, help="Name of output dataset (default <dataset>_aug)")
    p.add_argument("--splits", type=str, default="train", help="CSV: train,val,test")
    p.add_argument("--classes", type=str, default=None, help="Limit augmentation to CSV classes")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--preset",
        choices=tuple(AUGMENT_PRESETS.keys()),
        default=None,
        help=(
            "Preset with tuned augment balancing. "
            "augment-tail-safe: class-aware geo, bbox cap 1.10× baseline, tail-first budget "
            "(aligned with hybrid-aug-tail-budget defaults; standalone augment defaults stay unchanged)."
        ),
    )
    # --- flip ---
    p.add_argument("--enable-flip", action="store_true", help="Enable flip augmentation")
    p.add_argument(
        "--flip",
        choices=("horizontal", "vertical", "both", "h-and-v", "none"),
        default="horizontal",
        help="Flip mode: horizontal|vertical|both (combined H+V)|h-and-v (separate files)|none",
    )
    p.add_argument(
        "--flip-sampling",
        choices=("probabilistic", "exhaustive"),
        default="probabilistic",
        help=(
            "probabilistic: at most one flip per frame via --flip-prob; exhaustive: always all variants for --flip. "
            "With --aug-class-aware-geo, exhaustive bypasses per-frame probability scaling (see startup WARN)."
        ),
    )
    p.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        help="Probability of creating a flip variant per frame [0..1] (ignored when --flip-sampling exhaustive)",
    )
    # --- orthogonal ±90° ---
    p.add_argument("--enable-orthogonal-rotate", action="store_true", help="Enable ±90° orthogonal rotation augmentation")
    p.add_argument(
        "--orthogonal-rotate-sampling",
        choices=("probabilistic", "exhaustive"),
        default="probabilistic",
        help=(
            "probabilistic: at most one ±90° variant; exhaustive: always both +90° and -90°. "
            "With --aug-class-aware-geo, exhaustive bypasses per-frame probability scaling (see startup WARN)."
        ),
    )
    p.add_argument(
        "--orthogonal-rotate-prob",
        type=float,
        default=0.5,
        help="Probability of orthogonal rotate variant [0..1] (ignored when exhaustive)",
    )
    p.add_argument(
        "--orthogonal-rotate-direction",
        choices=("random", "cw", "ccw"),
        default="random",
        help="Direction for probabilistic orthogonal rotate: random|cw (+90°)|ccw (-90°)",
    )
    # --- center-rotate ---
    p.add_argument("--enable-center-rotate", action="store_true", dest="enable_center_rotate", help="Enable frame rotation around the center")
    p.add_argument("--disable-center-rotate", action="store_false", dest="enable_center_rotate", help="Disable frame rotation around center")
    p.add_argument("--center-rotate-deg", type=float, default=5.0, help="Maximum rotation angle in both directions")
    p.add_argument("--rotate-copies", type=int, default=1, help="Number of rotate options per frame")
    p.add_argument(
        "--center-rotate-anchor",
        choices=("center", "bbox", "detector"),
        default="center",
        help="Rotation center source: frame center, bbox markup or ROI detector",
    )
    # --- photometric ---
    p.add_argument("--enable-photometric", action="store_true", help="Enable brightness/contrast")
    p.add_argument("--brightness-limit", type=float, default=0.1, help="For policy=basic: brightness range")
    p.add_argument("--contrast-limit", type=float, default=0.1, help="For policy=basic: range contrast")
    # --- conveyor ---
    p.add_argument(
        "--enable-conveyor",
        action="store_true",
        help="Enable all conveyor effects (rotate, scale, blur, shift, noise); alias for all --enable-conveyor-* flags",
    )
    p.add_argument("--enable-conveyor-rotate", action="store_true", help="Conveyor Affine rotate (±5°)")
    p.add_argument("--enable-conveyor-scale", action="store_true", help="Conveyor Affine scale (0.95–1.05)")
    p.add_argument("--enable-conveyor-blur", action="store_true", help="Conveyor motion blur")
    p.add_argument("--enable-conveyor-shift", action="store_true", help="Conveyor Affine shift (±3%% translate)")
    p.add_argument(
        "--enable-conveyor-noise",
        action="store_true",
        help="Conveyor sensor noise (see --conveyor-noise-types)",
    )
    p.add_argument(
        "--conveyor-noise-types",
        type=str,
        default="iso,shot,gaussian",
        help="CSV noise types: gaussian,iso,shot,poisson-gaussian,multiplicative,impulse",
    )
    p.add_argument(
        "--conveyor-noise-intensity",
        type=float,
        default=0.35,
        help="Overall noise strength [0..1]",
    )
    p.add_argument(
        "--conveyor-noise-selection",
        choices=("random", "stack"),
        default="random",
        help="random: one noise type per frame; stack: apply all selected types",
    )
    # --- bbox copy ---
    p.add_argument("--enable-bbox-copy", action="store_true", help="Enable bbox-copy augmentation")
    p.add_argument("--bbox-copy-copies", type=int, default=1, help="Number of bbox_copy options per frame")
    p.add_argument("--copy-paste-rotation", type=float, default=0.0, help="For policy=bbox_copy: max |angle| to rotate the insert")
    p.add_argument("--copy-paste-scale-min", type=float, default=1.0, help="For policy=bbox_copy: min scale (inner)")
    p.add_argument("--copy-paste-scale-max", type=float, default=1.0, help="For policy=bbox_copy: max scale (inner)")
    p.add_argument("--copy-paste-max-iou", type=float, default=0.0, help="Max IoU insertions with existing bboxes")
    p.add_argument("--copy-paste-tries", type=int, default=25, help="Number of attempts to find a valid placement")
    p.add_argument(
        "--copy-paste-min-center-dist",
        type=float,
        default=0.15,
        help="Minimum distance between the centers of new inserts (fraction of the frame diagonal)",
    )
    p.add_argument(
        "--copy-paste-placement-style",
        choices=("random", "uniform-grid"),
        default="random",
        help="Insertion position selection style bbox_copy",
    )
    p.add_argument(
        "--class-balance",
        choices=("on", "off"),
        default="on",
        help="For bbox_copy: balance the selection of donors by class",
    )
    p.add_argument(
        "--color-match",
        choices=("meanstd", "off"),
        default="meanstd",
        help="For bbox_copy: align the brightness/contrast of the patch relative to the target area",
    )
    p.add_argument(
        "--blend-feather",
        type=float,
        default=0.16,
        help="For bbox_copy: seam smoothing strength [0..0.5], 0=no feather",
    )
    p.add_argument("--copy-paste-count", type=int, default=1, help="For policy=bbox_copy: number of inserts per image")
    # --- placement / ROI ---
    p.add_argument(
        "--placement-mode",
        choices=("none", "bbox", "detector"),
        default="none",
        help="ROI placement mode for bbox_copy and rotation pivot: none|bbox|detector (default none)",
    )
    p.add_argument("--placement-roi", action="store_true", help="Legacy: same as --placement-mode bbox")
    p.add_argument("--roi-model", type=str, default="yolo11n.pt", help="ROI detector model for --placement-mode detector")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all)")
    p.add_argument("--side-tolerance-px", type=float, default=3.0, help="Tolerance in px for ROI side classification")
    # --- balancing / budget ---
    p.add_argument("--imbalance-mode", choices=("off", "soft"), default="soft", help="Balancing according to scarce classes")
    p.add_argument("--imbalance-strength", type=float, default=1.0, help="Balancing strength >=0")
    p.add_argument(
        "--aug-class-aware-geo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale flip / photometric / conveyor attempt rate by inverse class frequency on the frame "
            "(uses --imbalance-mode soft and _image_soft_weight). Off keeps prior per-image behavior."
        ),
    )
    p.add_argument(
        "--aug-total-bbox-cap-mult",
        type=float,
        default=0.0,
        help=(
            "If >0: after augment, sum bbox on train ≤ ceil(mult × baseline B₀); baseline copies stay; "
            "only extra augmented frames consume slack (slack = ceil(mult×B₀) − B₀). mult=1.0 forbids extra bbox."
        ),
    )
    p.add_argument(
        "--aug-budget-tail-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --aug-total-bbox-cap-mult >0: process train images in split by descending tail priority "
            "(max_c (n_max/n_c)^γ over classes on the frame) so scarce classes consume slack first. "
            "Disable with --no-aug-budget-tail-first (dataset iteration order)."
        ),
    )
    p.add_argument(
        "--aug-budget-tail-gamma",
        type=float,
        default=1.0,
        help="Exponent γ for tail priority when --aug-budget-tail-first (default 1.0).",
    )
    p.add_argument(
        "--aug-per-class-bbox-cap-mult",
        type=float,
        default=0.0,
        help=(
            "If >0: per-class train bbox cap = ceil(mult × baseline n_c); extra variants that would exceed "
            "class slack are skipped. Complements --aug-total-bbox-cap-mult and tail-first ordering."
        ),
    )
    p.add_argument("--min-diversity-iou", type=float, default=0.97, help="bbox similarity threshold (higher -> almost duplicate)")
    p.add_argument("--min-angle-delta", type=float, default=1.0, help="Minimum angle difference between rotate options")
    # --- service ---
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-legend", action="store_true")
    p.set_defaults(enable_center_rotate=True)
    return p


def _detect_split(images_path: str) -> str:
    return detect_split_from_path(images_path, prefer_valid_name=True)


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


def _parse_yolo_labels(label_path: str) -> list[tuple[int, float, float, float, float]]:
    return labels_to_legacy_tuples(read_augment_label_file(label_path))


def _serialize_yolo_labels(labels: list[tuple[int, float, float, float, float]]) -> str:
    return legacy_tuples_to_serialized(labels)


@dataclass(frozen=True)
class FlipSpec:
    mode: Literal["horizontal", "vertical", "both"]
    tag: str


@dataclass(frozen=True)
class OrthogonalSpec:
    direction: Literal["cw", "ccw"]
    tag: str


def _flip_specs_for_mode(flip: str) -> list[FlipSpec]:
    if flip == "horizontal":
        return [FlipSpec("horizontal", "h")]
    if flip == "vertical":
        return [FlipSpec("vertical", "v")]
    if flip == "both":
        return [FlipSpec("both", "b")]
    if flip == "h-and-v":
        return [FlipSpec("horizontal", "h"), FlipSpec("vertical", "v")]
    return []


def _iter_flip_variants(args, rng: random.Random, *, flip_prob: float | None = None) -> list[FlipSpec]:
    if not args.enable_flip or str(args.flip) == "none":
        return []
    specs = _flip_specs_for_mode(str(args.flip))
    if not specs:
        return []
    if str(getattr(args, "flip_sampling", "probabilistic")) == "exhaustive":
        return specs
    p = float(flip_prob if flip_prob is not None else getattr(args, "flip_prob", 0.5))
    if rng.random() > p:
        return []
    if len(specs) == 1:
        return specs
    return [rng.choice(specs)]


def _iter_orthogonal_variants(args, rng: random.Random, *, orth_prob: float | None = None) -> list[OrthogonalSpec]:
    if not bool(getattr(args, "enable_orthogonal_rotate", False)):
        return []
    if str(getattr(args, "orthogonal_rotate_sampling", "probabilistic")) == "exhaustive":
        return [OrthogonalSpec("cw", "c"), OrthogonalSpec("ccw", "a")]
    p = float(orth_prob if orth_prob is not None else getattr(args, "orthogonal_rotate_prob", 0.5))
    if rng.random() > p:
        return []
    d = str(getattr(args, "orthogonal_rotate_direction", "random"))
    if d == "cw":
        return [OrthogonalSpec("cw", "c")]
    if d == "ccw":
        return [OrthogonalSpec("ccw", "a")]
    direction = rng.choice(["cw", "ccw"])
    return [OrthogonalSpec(direction, "c" if direction == "cw" else "a")]


def _normalize_augment_args(args, *, argv: list[str] | None = None) -> str | None:
    if bool(getattr(args, "placement_roi", False)):
        args.placement_mode = "bbox"

    if bool(getattr(args, "enable_flip", False)) and str(getattr(args, "flip", "horizontal")) == "none":
        return "[ERROR] --enable-flip conflicts with --flip none."
    if not 0.0 <= float(getattr(args, "flip_prob", 0.5)) <= 1.0:
        return "[ERROR] --flip-prob must be in [0, 1]."
    if not 0.0 <= float(getattr(args, "orthogonal_rotate_prob", 0.5)) <= 1.0:
        return "[ERROR] --orthogonal-rotate-prob must be in [0, 1]."
    try:
        parse_noise_types(getattr(args, "conveyor_noise_types", None))
    except ValueError as exc:
        return f"[ERROR] {exc}"
    if not 0.0 <= float(getattr(args, "conveyor_noise_intensity", 0.35)) <= 1.0:
        return "[ERROR] --conveyor-noise-intensity must be in [0, 1]."

    anchor_explicit = argv is not None and "--center-rotate-anchor" in argv
    placement_explicit = argv is not None and ("--placement-mode" in argv or "--placement-roi" in argv)
    rotate_on = bool(getattr(args, "enable_center_rotate", False))
    copy_on = bool(getattr(args, "enable_bbox_copy", False))

    if rotate_on and copy_on:
        if placement_explicit and not anchor_explicit:
            args.center_rotate_anchor = {
                "none": "center",
                "bbox": "bbox",
                "detector": "detector",
            }[str(getattr(args, "placement_mode", "none"))]
        elif anchor_explicit and not placement_explicit:
            args.placement_mode = {
                "center": "none",
                "bbox": "bbox",
                "detector": "detector",
            }[str(getattr(args, "center_rotate_anchor", "center"))]

    return None


def _conveyor_any(args) -> bool:
    return any(
        bool(getattr(args, name, False))
        for name in (
            "enable_conveyor_rotate",
            "enable_conveyor_scale",
            "enable_conveyor_blur",
            "enable_conveyor_shift",
            "enable_conveyor_noise",
        )
    )


def _sync_conveyor_flags(args, *, argv: list[str] | None = None) -> None:
    if argv is not None and "--enable-conveyor" in argv:
        args.enable_conveyor_rotate = True
        args.enable_conveyor_scale = True
        args.enable_conveyor_blur = True
        args.enable_conveyor_shift = True
        args.enable_conveyor_noise = True
    args.enable_conveyor = _conveyor_any(args)


def _set_conveyor_enabled(args, enabled: bool) -> None:
    args.enable_conveyor = bool(enabled)
    if not enabled:
        args.enable_conveyor_rotate = False
        args.enable_conveyor_scale = False
        args.enable_conveyor_blur = False
        args.enable_conveyor_shift = False
        args.enable_conveyor_noise = False


def _append_conveyor_transforms(t: list[A.BasicTransform], args) -> None:
    geo = (
        bool(getattr(args, "enable_conveyor_rotate", False))
        or bool(getattr(args, "enable_conveyor_scale", False))
        or bool(getattr(args, "enable_conveyor_shift", False))
    )
    if geo:
        rotate = (-5, 5) if bool(getattr(args, "enable_conveyor_rotate", False)) else 0
        translate = (
            {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}
            if bool(getattr(args, "enable_conveyor_shift", False))
            else {"x": (0, 0), "y": (0, 0)}
        )
        scale = (0.95, 1.05) if bool(getattr(args, "enable_conveyor_scale", False)) else (1.0, 1.0)
        t.append(
            A.Affine(
                translate_percent=translate,
                scale=scale,
                rotate=rotate,
                border_mode=0,
                p=0.8,
            )
        )
    if bool(getattr(args, "enable_conveyor_noise", False)):
        noise_tf = build_conveyor_noise_transform(args)
        if noise_tf is not None:
            t.append(noise_tf)
    if bool(getattr(args, "enable_conveyor_blur", False)):
        t.append(A.MotionBlur(blur_limit=3, p=0.15))


def _compose_for_basic(
    args,
    *,
    flip_mode: str | None = None,
    with_bboxes: bool = True,
    with_keypoints: bool = False,
) -> A.Compose:
    t: list[A.BasicTransform] = []
    flip = str(flip_mode if flip_mode is not None else getattr(args, "flip", "horizontal"))
    apply_flip = bool(getattr(args, "enable_flip", False)) or flip_mode is not None
    if apply_flip:
        if flip == "horizontal":
            t.append(A.HorizontalFlip(p=1.0))
        elif flip == "vertical":
            t.append(A.VerticalFlip(p=1.0))
        elif flip == "both":
            t.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
    if _conveyor_any(args):
        _append_conveyor_transforms(t, args)
    if args.enable_photometric:
        t.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(args.brightness_limit),
                contrast_limit=float(args.contrast_limit),
                p=1.0,
            )
        )
    if args.enable_center_rotate:
        t.append(A.Affine(rotate=(-float(args.center_rotate_deg), float(args.center_rotate_deg)), p=1.0))
    if not t:
        return A.Compose([])
    params: dict = {}
    if with_bboxes:
        params["bbox_params"] = A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True)
    if with_keypoints:
        params["keypoint_params"] = A.KeypointParams(format="xy", remove_invisible=False)
    return A.Compose(t, **params)


def _sanitize_yolo_box(
    label: tuple[int, float, float, float, float],
    *,
    min_size: float = 1e-6,
) -> tuple[int, float, float, float, float] | None:
    cls_id, cx, cy, bw, bh = label
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    nx = (x1 + x2) / 2.0
    ny = (y1 + y2) / 2.0
    nw = x2 - x1
    nh = y2 - y1
    return (cls_id, nx, ny, nw, nh)


def _base36(num: int) -> str:
    if num <= 0:
        return "0"
    x = num
    out = []
    while x:
        x, r = divmod(x, 36)
        out.append(_BASE36_ALPHABET[r])
    return "".join(reversed(out))


def _variant_code(args) -> str:
    if _conveyor_any(args):
        return "s"
    if args.enable_photometric:
        return "l"
    return "n"


def _aug_stem(
    stem: str,
    args,
    idx: int,
    mode: str,
    *,
    flip_tag: str = "n",
    orth_tag: str = "",
) -> str:
    extra = ""
    if mode == "o":
        extra = orth_tag
    elif mode == "f":
        extra = flip_tag
    return f"{stem}__a-{mode}{extra}{_variant_code(args)}{_base36(idx)}"


def _labels_signature_iou(
    a: list[tuple[int, float, float, float, float]],
    b: list[tuple[int, float, float, float, float]],
    w: int,
    h: int,
) -> float:
    if not a or not b:
        return 0.0
    a_xy = [_to_xyxy(x, w, h) for x in a]
    b_xy = [_to_xyxy(x, w, h) for x in b]
    sims: list[float] = []
    for aa in a_xy:
        sims.append(max((_iou(aa, bb) for bb in b_xy), default=0.0))
    return float(sum(sims) / len(sims)) if sims else 0.0


def _collect_class_freq(items: list[dict], *, train_only: bool = True) -> dict[int, int]:
    freq: dict[int, int] = {}
    for it in items:
        if train_only:
            split_norm = SPLIT_ALIASES.get(str(it.get("split", "")).strip().lower(), str(it.get("split", "")).strip().lower())
            if split_norm != "train":
                continue
        for cls, *_ in _parse_yolo_labels(it["lbl"]):
            freq[int(cls)] = freq.get(int(cls), 0) + 1
    return freq


def _label_class_counts(lbl_path: str) -> dict[int, int]:
    ctr: dict[int, int] = defaultdict(int)
    for cls, *_ in _parse_yolo_labels(lbl_path):
        ctr[int(cls)] += 1
    return dict(ctr)


def _labels_class_counts(labels: list[tuple[int, float, float, float, float]]) -> dict[int, int]:
    ctr: dict[int, int] = defaultdict(int)
    for cls, *_ in labels:
        ctr[int(cls)] += 1
    return dict(ctr)


def _inserted_class_delta(
    original: dict[int, int],
    new_labels: list[tuple[int, float, float, float, float]],
) -> tuple[int, dict[int, int]]:
    new_counts = _labels_class_counts(new_labels)
    delta_by_class: dict[int, int] = {}
    for c in set(original) | set(new_counts):
        d = int(new_counts.get(c, 0)) - int(original.get(c, 0))
        if d > 0:
            delta_by_class[c] = d
    return sum(delta_by_class.values()), delta_by_class


def _image_soft_weight(class_ids: set[int], class_freq: dict[int, int], alpha: float) -> float:
    if not class_ids:
        return 1.0
    vals: list[float] = []
    for c in class_ids:
        f = max(1, int(class_freq.get(int(c), 1)))
        vals.append((1.0 / float(f)) ** float(alpha))
    if not vals:
        return 1.0
    return float(sum(vals) / len(vals))


def _scaled_copies(base: int, image_weight: float, args) -> int:
    base = max(0, int(base))
    if str(getattr(args, "imbalance_mode", "soft")) != "soft" or base == 0:
        return base
    strength = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    scaled = int(round(base * (1.0 + image_weight * strength)))
    return max(0, scaled)


def count_yolo_bbox_lines(lbl_path: str) -> int:
    """Number of label instances (bbox or polygon lines) in a YOLO label file."""
    return count_label_instances(lbl_path)


def _train_split_class_bbox_counts(items: list[dict[str, str]]) -> dict[int, int]:
    """Per-class bbox line counts on train split before augment."""
    ctr: dict[int, int] = defaultdict(int)
    for it in items:
        sp = str(it.get("split", "")).strip().lower()
        if sp == "valid":
            sp = "val"
        if sp != "train":
            continue
        for lb in _parse_yolo_labels(it["lbl"]):
            ctr[int(lb[0])] += 1
    return dict(ctr)


def _train_split_bbox_sum(items: list[dict[str, str]]) -> int:
    """Sum YOLO bbox lines on train split (budget baseline B₀ before augment)."""
    total = 0
    for it in items:
        sp = str(it.get("split", "")).strip().lower()
        if sp == "valid":
            sp = "val"
        if sp != "train":
            continue
        total += count_yolo_bbox_lines(it["lbl"])
    return total


def _image_tail_priority_score(lbl_path: str, cls_counts: dict[int, int], gamma: float) -> float:
    """Plan MVP: max over classes in image of (n_max / n_c)^γ (higher => more tail / process first)."""
    if not cls_counts:
        return 0.0
    n_max = max(int(v) for v in cls_counts.values())
    if n_max <= 0:
        return 0.0
    g = float(gamma)
    best = 0.0
    for c in _read_yolo_classes(lbl_path):
        nc = max(1, int(cls_counts.get(int(c), 0)))
        s = (float(n_max) / float(nc)) ** g
        if s > best:
            best = s
    return float(best)


def _reorder_items_for_bbox_budget(
    items: list[dict[str, str]],
    *,
    split_filter: set[str],
    cls_counts: dict[int, int],
    extra_budget: int | None,
    tail_first: bool,
    gamma: float,
) -> list[dict[str, str]]:
    if extra_budget is None or not tail_first:
        return items

    def sort_key(i: int) -> tuple[int, float, int]:
        it = items[i]
        split = it["split"]
        split_norm = SPLIT_ALIASES.get(str(split).strip().lower(), str(split).strip().lower())
        if split_norm != "train" or split not in split_filter:
            return (1, 0.0, i)
        pr = _image_tail_priority_score(it["lbl"], cls_counts, gamma)
        return (0, -pr, i)

    order = sorted(range(len(items)), key=sort_key)
    return [items[j] for j in order]


def _class_aware_enabled(args) -> bool:
    return bool(getattr(args, "aug_class_aware_geo", False)) and str(getattr(args, "imbalance_mode", "soft")) == "soft"


def _class_aware_trigger_prob(args, image_weight: float, base_prob: float) -> float:
    """Unified class-aware trigger probability for flip / photo / orthogonal geo branches."""
    base = float(base_prob)
    if not _class_aware_enabled(args):
        return base
    strength = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    w = max(1e-9, float(image_weight))
    scale = math.sqrt(w) * max(0.35, min(2.5, strength))
    p = base * scale
    return float(min(1.0, max(0.02, p)))


def _effective_orthogonal_prob_geo(args, image_weight: float) -> float:
    return _class_aware_trigger_prob(args, image_weight, float(getattr(args, "orthogonal_rotate_prob", 0.5)))


def _effective_flip_prob_geo(args, image_weight: float) -> float:
    return _class_aware_trigger_prob(args, image_weight, float(getattr(args, "flip_prob", 0.5)))


def _geo_photo_trigger(args, image_weight: float, rng: random.Random) -> bool:
    """Whether to emit photometric/conveyor variant for this frame (class-aware)."""
    if not _class_aware_enabled(args):
        return True
    p = _class_aware_trigger_prob(args, image_weight, 1.0)
    return bool(rng.random() < p)


def _aug_extra_budget_allow(extra_used: int, delta: int, extra_budget: int | None) -> bool:
    """Extra augment bbox must stay within extra_budget (cap_total − baseline B₀)."""
    if extra_budget is None:
        return True
    return extra_used + delta <= extra_budget


def sum_train_bbox_disk(dataset_root: str) -> int:
    """Sum bbox lines across train labels (split layout) or root labels/ (flat layout)."""
    for rel in ("train/labels", "labels"):
        lbl_dir = os.path.join(dataset_root, rel)
        if not os.path.isdir(lbl_dir):
            continue
        s = 0
        for name in os.listdir(lbl_dir):
            if not name.endswith(".txt"):
                continue
            s += count_yolo_bbox_lines(os.path.join(lbl_dir, name))
        return s
    return 0


def sum_train_class_bbox_disk(dataset_root: str) -> dict[int, int]:
    """Per-class bbox counts on train split (or flat labels/)."""
    ctr: dict[int, int] = defaultdict(int)
    for rel in ("train/labels", "labels"):
        lbl_dir = os.path.join(dataset_root, rel)
        if not os.path.isdir(lbl_dir):
            continue
        for name in os.listdir(lbl_dir):
            if not name.endswith(".txt"):
                continue
            for cls, *_ in _parse_yolo_labels(os.path.join(lbl_dir, name)):
                ctr[int(cls)] += 1
        return dict(ctr)
    return {}


def _provided_augment_flags(argv: list[str]) -> set[str]:
    out: set[str] = set()
    for tok in argv:
        if tok.startswith("--"):
            out.add(tok.split("=", 1)[0])
    return out


def _apply_augment_preset_defaults(args: argparse.Namespace, provided_flags: set[str]) -> None:
    if not getattr(args, "preset", None):
        return
    preset_cfg = AUGMENT_PRESETS.get(str(args.preset), {})
    flag_for_attr = {
        "aug_class_aware_geo": "--aug-class-aware-geo",
        "aug_total_bbox_cap_mult": "--aug-total-bbox-cap-mult",
        "aug_budget_tail_first": "--aug-budget-tail-first",
        "aug_budget_tail_gamma": "--aug-budget-tail-gamma",
    }
    for attr, value in preset_cfg.items():
        flag = flag_for_attr.get(attr)
        if flag and flag in provided_flags:
            continue
        setattr(args, attr, value)


def _warn_exhaustive_class_aware(args) -> None:
    if not _class_aware_enabled(args):
        return
    flip_ex = bool(getattr(args, "enable_flip", False)) and str(getattr(args, "flip_sampling", "probabilistic")) == "exhaustive"
    orth_ex = bool(getattr(args, "enable_orthogonal_rotate", False)) and str(
        getattr(args, "orthogonal_rotate_sampling", "probabilistic")
    ) == "exhaustive"
    if flip_ex or orth_ex:
        print(
            "[WARN] augment: --aug-class-aware-geo is active but flip/orthogonal use exhaustive sampling; "
            "per-frame probability scaling is bypassed (all variants are emitted). "
            "Use probabilistic sampling or disable class-aware geo for head-tail control."
        )


def _to_xyxy(box: tuple[int, float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2.0) * w))
    y1 = int(round((cy - bh / 2.0) * h))
    x2 = int(round((cx + bw / 2.0) * w))
    y2 = int(round((cy + bh / 2.0) * h))
    return x1, y1, x2, y2


def _to_yolo(cls_id: int, xyxy: tuple[int, int, int, int], w: int, h: int) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cls_id, cx / w, cy / h, bw / w, bh / h)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pick_uniform_grid_position(
    *,
    rng: random.Random,
    roi: tuple[int, int, int, int] | None,
    placement_mode: str,
    w: int,
    h: int,
    pw: int,
    ph: int,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    if placement_mode in ("bbox", "detector") and roi is not None:
        x_min, y_min, x_max, y_max = roi
    else:
        x_min, y_min, x_max, y_max = 0, 0, w, h
    x_max = max(x_min, x_max - pw)
    y_max = max(y_min, y_max - ph)
    cell = max(8, int(max(pw, ph) * 0.8))
    xs = list(range(x_min, x_max + 1, cell)) or [x_min]
    ys = list(range(y_min, y_max + 1, cell)) or [y_min]
    candidates: list[tuple[float, int, int]] = []
    for gx in xs:
        for gy in ys:
            cand = (gx, gy, gx + pw, gy + ph)
            cx, cy = _center(cand)
            if not occupied:
                score = 1.0
            else:
                d = min(np.hypot(cx - _center(ob)[0], cy - _center(ob)[1]) for ob in occupied)
                score = float(d)
            # A little noise so as not to get stuck in the same pattern.
            score += rng.random() * 0.01
            candidates.append((score, gx, gy))
    candidates.sort(key=lambda t: t[0], reverse=True)
    top_k = min(10, len(candidates))
    _, x, y = candidates[rng.randrange(0, top_k)]
    return int(x), int(y)


def _roi_from_labels(labels: list[tuple[int, float, float, float, float]], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if not labels:
        return None
    boxes = [_to_xyxy(x, img_w, img_h) for x in labels]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _classify_side(roi: tuple[int, int, int, int], box: tuple[int, int, int, int], tol: float) -> str:
    rx1, ry1, rx2, ry2 = roi
    bx1, by1, bx2, by2 = box
    d = {
        "left": abs(bx1 - rx1),
        "right": abs(rx2 - bx2),
        "top": abs(by1 - ry1),
        "bottom": abs(ry2 - by2),
    }
    first = min(d, key=d.get)
    first_v = d[first]
    sorted_vals = sorted(d.items(), key=lambda it: it[1])
    second, second_v = sorted_vals[1]
    if second_v - first_v <= tol:
        pair = sorted([first, second])
        return f"corner:{pair[0]}-{pair[1]}"
    return first


def _inside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2


def _parse_roi_class_ids(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    out: list[int] = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except ValueError:
            continue
    return out or None


def _detect_roi_box(image_path: str, args) -> tuple[int, int, int, int] | None:
    model_path = str(getattr(args, "roi_model", "") or "yolo11n.pt")
    if model_path not in _ROI_MODEL_CACHE:
        _ROI_MODEL_CACHE[model_path] = YOLO(model_path)
    model = _ROI_MODEL_CACHE[model_path]
    class_ids = _parse_roi_class_ids(getattr(args, "roi_class_ids", None))
    proj = getattr(args, "_ultralytics_roi_predict_project", None) or ultralytics_sidecar_dir(
        tempfile.gettempdir(), "smartrain_augment_roi"
    )
    results = model.predict(
        source=image_path,
        conf=float(getattr(args, "roi_conf", 0.25)),
        classes=class_ids,
        verbose=False,
        save=False,
        project=proj,
        name="augment-roi",
        exist_ok=True,
    )
    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
    idx = int(np.argmax(confs)) if confs is not None and len(confs) else 0
    x1, y1, x2, y2 = xyxy[idx].tolist()
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))


def _build_detector_roi_cache(items: list[dict], args) -> dict[str, tuple[int, int, int, int] | None]:
    cache: dict[str, tuple[int, int, int, int] | None] = {}
    progress = tqdm(
        items,
        total=len(items),
        desc="augment:roi-detector",
        unit="img",
        disable=bool(args.no_legend),
    )
    for it in progress:
        img_path = it["img"]
        cache[img_path] = _detect_roi_box(img_path, args)
    progress.close()
    return cache


def _apply_geom_aug(
    image_path: str,
    label_path: str,
    out_img: str,
    out_lbl: str,
    args,
    *,
    enable_flip: bool | None = None,
    enable_photometric: bool | None = None,
    enable_conveyor: bool | None = None,
    enable_center_rotate: bool | None = None,
    flip_mode: str | None = None,
) -> list[tuple[int, float, float, float, float]]:
    image = np.array(Image.open(image_path).convert("RGB"))
    raw_labels = read_augment_label_file(label_path)
    # Locally disable/enable individual blocks without mutating the main args.
    class _LocalArgs:
        pass

    local = _LocalArgs()
    for k, v in vars(args).items():
        setattr(local, k, v)
    if enable_flip is not None:
        local.enable_flip = bool(enable_flip)
    if enable_photometric is not None:
        local.enable_photometric = bool(enable_photometric)
    if enable_conveyor is not None:
        _set_conveyor_enabled(local, bool(enable_conveyor))
    if enable_center_rotate is not None:
        local.enable_center_rotate = bool(enable_center_rotate)

    kind = infer_label_kind(raw_labels)
    pipeline = _compose_for_basic(
        local,
        flip_mode=flip_mode,
        with_bboxes=kind in {"bbox", "mixed"},
        with_keypoints=kind in {"segment", "mixed"},
    )
    if not list(getattr(pipeline, "transforms", []) or []):
        shutil.copy2(image_path, out_img)
        shutil.copy2(label_path, out_lbl)
        return _parse_yolo_labels(out_lbl)
    new_img, new_label_objs = apply_albumentations_to_labels(image, raw_labels, pipeline)
    new_labels = labels_to_legacy_tuples(new_label_objs)
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(new_img).save(out_img)
    write_augment_label_file(out_lbl, new_label_objs)
    return new_labels


def _apply_exact_center_rotate(
    image_path: str,
    label_path: str,
    out_img: str,
    out_lbl: str,
    args,
    angle: float,
    *,
    detector_roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, float, float, float, float]] | None:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    labels = read_augment_label_file(label_path)
    rotate_anchor = str(getattr(args, "center_rotate_anchor", "center"))
    rotate_use_roi = rotate_anchor in ("bbox", "detector")
    roi: tuple[int, int, int, int] | None = None
    if rotate_use_roi and rotate_anchor == "bbox":
        roi = _roi_from_labels(_parse_yolo_labels(label_path), w, h)
    elif rotate_use_roi and rotate_anchor == "detector":
        roi = detector_roi
    if rotate_use_roi and roi is None:
        return None
    if rotate_use_roi and roi is not None:
        cx = (roi[0] + roi[2]) / 2.0
        cy = (roi[1] + roi[3]) / 2.0
    else:
        cx = w / 2.0
        cy = h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
    src = np.array(img.convert("RGB"))
    dst = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_label_objs = rotate_labels_with_matrix(labels, m, w=w, h=h)
    new_labels = labels_to_legacy_tuples(new_label_objs)
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(dst).save(out_img)
    write_augment_label_file(out_lbl, new_label_objs)
    return new_labels


def _build_donor_pool(items: list[dict], args) -> list[dict]:
    donors: list[dict] = []
    progress = tqdm(
        items,
        total=len(items),
        desc="augment:donors",
        unit="img",
        disable=bool(args.no_legend),
    )
    for it in progress:
        img = Image.open(it["img"]).convert("RGB")
        w, h = img.size
        labels = _parse_yolo_labels(it["lbl"])
        roi = _roi_from_labels(labels, w, h)
        for lb in labels:
            cls_id = lb[0]
            xyxy = _to_xyxy(lb, w, h)
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img.crop((x1, y1, x2, y2))
            if roi is None:
                side = "inner"
            else:
                side_key = _classify_side(roi, xyxy, float(args.side_tolerance_px))
                rx1, ry1, rx2, ry2 = roi
                touches = abs(x1 - rx1) <= args.side_tolerance_px or abs(x2 - rx2) <= args.side_tolerance_px or abs(y1 - ry1) <= args.side_tolerance_px or abs(y2 - ry2) <= args.side_tolerance_px
                side = f"edge:{side_key}" if touches else "inner"
            donors.append({"class_id": cls_id, "patch": patch, "w": x2 - x1, "h": y2 - y1, "side": side})
        progress.set_postfix(donors=len(donors), refresh=False)
    progress.close()
    return donors


def _match_patch_to_region(patch_arr: np.ndarray, region_arr: np.ndarray) -> np.ndarray:
    patch = patch_arr.astype(np.float32)
    region = region_arr.astype(np.float32)
    p_mean = patch.mean(axis=(0, 1), keepdims=True)
    r_mean = region.mean(axis=(0, 1), keepdims=True)
    p_std = patch.std(axis=(0, 1), keepdims=True) + 1e-6
    r_std = region.std(axis=(0, 1), keepdims=True) + 1e-6
    matched = (patch - p_mean) * (r_std / p_std) + r_mean
    return np.clip(matched, 0, 255).astype(np.uint8)


def _feather_alpha(width: int, height: int, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return np.ones((height, width, 1), dtype=np.float32)
    strength = min(0.5, max(0.01, float(strength)))
    edge = max(2, int(min(width, height) * strength))
    x = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    fx = np.clip(x / edge, 0.0, 1.0)
    fy = np.clip(y / edge, 0.0, 1.0)
    return np.outer(fy, fx)[..., None]


def _pick_donor_balanced(donors: list[dict], class_usage: dict[int, int], rng: random.Random) -> dict:
    by_class: dict[int, list[dict]] = {}
    for d in donors:
        by_class.setdefault(int(d["class_id"]), []).append(d)
    min_used = min(class_usage.get(c, 0) for c in by_class.keys())
    candidate_classes = [c for c in by_class.keys() if class_usage.get(c, 0) == min_used]
    cls = candidate_classes[rng.randrange(0, len(candidate_classes))]
    pool = by_class[cls]
    return pool[rng.randrange(0, len(pool))]


def _pick_donor_any(donors: list[dict], rng: random.Random) -> dict:
    return donors[rng.randrange(0, len(donors))]


def _pick_donor_soft(
    donors: list[dict],
    class_usage: dict[int, int],
    class_freq: dict[int, int],
    args,
    rng: random.Random,
) -> dict:
    by_class: dict[int, list[dict]] = {}
    for d in donors:
        by_class.setdefault(int(d["class_id"]), []).append(d)
    alpha = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    weights: list[tuple[int, float]] = []
    for c in by_class.keys():
        f = max(1, int(class_freq.get(c, 1)))
        generated = int(class_usage.get(c, 0))
        w = (1.0 / float(f + generated)) ** alpha
        weights.append((c, max(1e-9, w)))
    total = sum(w for _, w in weights)
    pick = rng.random() * total
    acc = 0.0
    chosen = weights[-1][0]
    for c, w in weights:
        acc += w
        if pick <= acc:
            chosen = c
            break
    pool = by_class[chosen]
    return pool[rng.randrange(0, len(pool))]


def _apply_copy_paste(
    image_path: str,
    label_path: str,
    out_img: str,
    out_lbl: str,
    args,
    donors: list[dict],
    class_usage: dict[int, int],
    *,
    variant_seed: int = 0,
    class_freq: dict[int, int] | None = None,
    detector_roi: tuple[int, int, int, int] | None = None,
) -> tuple[bool, list[tuple[int, float, float, float, float]]]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    labels = _parse_yolo_labels(label_path)
    placement_mode = str(getattr(args, "placement_mode", "none"))
    roi: tuple[int, int, int, int] | None = None
    if placement_mode == "bbox":
        roi = _roi_from_labels(labels, w, h)
    elif placement_mode == "detector":
        roi = detector_roi
    if placement_mode in ("bbox", "detector") and roi is None:
        return False, []
    existing = [_to_xyxy(lb, w, h) for lb in labels]
    placed_boxes: list[tuple[int, int, int, int]] = []
    path_seed = zlib.crc32(image_path.encode("utf-8")) & 0xFFFFFFFF
    rng = random.Random(int(args.seed) + int(variant_seed) + int(path_seed))
    total_placed = 0
    for _ in range(int(args.copy_paste_count)):
        if not donors:
            break
        placed = False
        for _try in range(int(args.copy_paste_tries)):
            if str(getattr(args, "imbalance_mode", "soft")) == "soft" and class_freq:
                d = _pick_donor_soft(donors, class_usage, class_freq, args, rng)
            elif str(getattr(args, "class_balance", "on")) == "on":
                d = _pick_donor_balanced(donors, class_usage, rng)
            else:
                d = _pick_donor_any(donors, rng)
            patch = d["patch"].copy()
            edge_mode = d["side"].startswith("edge:")
            if not edge_mode:
                angle = rng.uniform(-float(args.copy_paste_rotation), float(args.copy_paste_rotation))
                patch = patch.rotate(angle, resample=Image.BICUBIC, expand=True)
                s = rng.uniform(float(args.copy_paste_scale_min), float(args.copy_paste_scale_max))
                nw = max(2, int(round(patch.size[0] * s)))
                nh = max(2, int(round(patch.size[1] * s)))
                patch = patch.resize((nw, nh), Image.BICUBIC)
            pw, ph = patch.size
            if pw >= w or ph >= h:
                continue
            if placement_mode in ("bbox", "detector") and roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if edge_mode:
                    side = d["side"].split(":", 1)[1]
                    if side.startswith("corner:"):
                        c = side.split(":", 1)[1]
                        if c == "left-top":
                            x, y = rx1, ry1
                        elif c == "left-bottom":
                            x, y = rx1, max(ry1, ry2 - ph)
                        elif c == "right-top":
                            x, y = max(rx1, rx2 - pw), ry1
                        else:
                            x, y = max(rx1, rx2 - pw), max(ry1, ry2 - ph)
                    elif side == "left":
                        x = rx1
                        y = rng.randint(ry1, max(ry1, ry2 - ph))
                    elif side == "right":
                        x = max(rx1, rx2 - pw)
                        y = rng.randint(ry1, max(ry1, ry2 - ph))
                    elif side == "top":
                        x = rng.randint(rx1, max(rx1, rx2 - pw))
                        y = ry1
                    else:
                        x = rng.randint(rx1, max(rx1, rx2 - pw))
                        y = max(ry1, ry2 - ph)
                else:
                    if str(getattr(args, "copy_paste_placement_style", "random")) == "uniform-grid":
                        x, y = _pick_uniform_grid_position(
                            rng=rng,
                            roi=roi,
                            placement_mode=placement_mode,
                            w=w,
                            h=h,
                            pw=pw,
                            ph=ph,
                            occupied=existing + placed_boxes,
                        )
                    else:
                        x = rng.randint(rx1, max(rx1, rx2 - pw))
                        y = rng.randint(ry1, max(ry1, ry2 - ph))
            else:
                if str(getattr(args, "copy_paste_placement_style", "random")) == "uniform-grid":
                    x, y = _pick_uniform_grid_position(
                        rng=rng,
                        roi=None,
                        placement_mode=placement_mode,
                        w=w,
                        h=h,
                        pw=pw,
                        ph=ph,
                        occupied=existing + placed_boxes,
                    )
                else:
                    x = rng.randint(0, w - pw)
                    y = rng.randint(0, h - ph)
            cand = (x, y, x + pw, y + ph)
            if placement_mode in ("bbox", "detector") and roi is not None and not _inside(cand, roi):
                continue
            if any(_iou(cand, ex) > float(args.copy_paste_max_iou) for ex in existing):
                continue
            # Anti-clustering: avoid too close centers of new inserts in the same frame.
            min_center_dist = max(0.0, float(getattr(args, "copy_paste_min_center_dist", 0.15)))
            if min_center_dist > 0.0 and placed_boxes:
                cx, cy = _center(cand)
                diag = max(1.0, float(np.hypot(w, h)))
                min_px = min_center_dist * diag
                too_close = False
                for pb in placed_boxes:
                    px, py = _center(pb)
                    if np.hypot(cx - px, cy - py) < min_px:
                        too_close = True
                        break
                if too_close:
                    continue
            region = img.crop((x, y, x + pw, y + ph))
            patch_arr = np.array(patch.convert("RGB"))
            region_arr = np.array(region.convert("RGB"))
            if str(getattr(args, "color_match", "meanstd")) == "meanstd":
                patch_arr = _match_patch_to_region(patch_arr, region_arr)
            alpha = _feather_alpha(pw, ph, float(getattr(args, "blend_feather", 0.16)))
            blended = (patch_arr.astype(np.float32) * alpha + region_arr.astype(np.float32) * (1.0 - alpha)).astype(
                np.uint8
            )
            img.paste(Image.fromarray(blended), (x, y))
            existing.append(cand)
            placed_boxes.append(cand)
            labels.append(_to_yolo(int(d["class_id"]), cand, w, h))
            class_usage[int(d["class_id"])] = class_usage.get(int(d["class_id"]), 0) + 1
            placed = True
            total_placed += 1
            break
        if not placed:
            continue
    if total_placed == 0:
        return False, labels
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    img.save(out_img)
    Path(out_lbl).write_text(_serialize_yolo_labels(labels), encoding="utf-8")
    return True, labels


def _split_images_rel(out_dir: str, split: str) -> str | None:
    base = Path(out_dir)
    for rel in (f"{split}/images", f"images/{split}", split):
        p = base / rel
        if p.is_dir() and any(p.iterdir()):
            return rel
    return None


_AUGMENT_FLAT_SOURCE_STRUCTURES = frozenset({"cvat11", "flat"})


def augment_output_structure(source_structure: str) -> str:
    """Map datasets_info structure to on-disk layout produced by augment."""
    s = str(source_structure).strip().lower()
    if s in _AUGMENT_FLAT_SOURCE_STRUCTURES:
        return "flat"
    if s in ("subset_flat", "nested_split", "split"):
        return s
    return "split"


def _swap_images_labels_rel(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/")
    parts = rel.split("/")
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return "/".join(parts)
    return os.path.join("labels", os.path.basename(rel_path))


def _augment_ensure_base_dirs(out_dir: str, structure: str, splits_present: list[str]) -> None:
    base = Path(out_dir)
    if structure in ("flat", "subset_flat"):
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "labels").mkdir(parents=True, exist_ok=True)
        return
    if structure == "nested_split":
        for split in splits_present:
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        return
    for split in splits_present:
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)


def _augment_output_paths(
    out_dir: str,
    structure: str,
    src_root: str,
    img_src: str,
    lbl_src: str,
    stem: str,
    ext: str,
) -> tuple[str, str]:
    if structure == "flat":
        return (
            os.path.join(out_dir, "images", f"{stem}{ext}"),
            os.path.join(out_dir, "labels", f"{stem}.txt"),
        )
    rel_img = os.path.relpath(img_src, src_root)
    rel_img_dir = os.path.dirname(rel_img)
    dst_img = (
        os.path.join(out_dir, rel_img_dir, f"{stem}{ext}")
        if rel_img_dir
        else os.path.join(out_dir, f"{stem}{ext}")
    )
    if structure == "subset_flat":
        rel_lbl = _swap_images_labels_rel(rel_img)
        rel_lbl_dir = os.path.dirname(rel_lbl)
        dst_lbl = (
            os.path.join(out_dir, rel_lbl_dir, f"{stem}.txt")
            if rel_lbl_dir
            else os.path.join(out_dir, f"{stem}.txt")
        )
        return dst_img, dst_lbl
    try:
        rel_lbl = os.path.relpath(lbl_src, src_root)
    except ValueError:
        rel_lbl = _swap_images_labels_rel(rel_img)
    rel_lbl_dir = os.path.dirname(rel_lbl)
    dst_lbl = (
        os.path.join(out_dir, rel_lbl_dir, f"{stem}.txt")
        if rel_lbl_dir
        else os.path.join(out_dir, f"{stem}.txt")
    )
    return dst_img, dst_lbl


def _write_data_yaml(out_dir: str, names: list[str], *, structure: str) -> None:
    out_structure = augment_output_structure(structure)
    if out_structure in ("flat", "subset_flat"):
        train_rel = val_rel = test_rel = "images"
    elif out_structure == "nested_split":
        train_rel = _split_images_rel(out_dir, "train") or "images/train"
        val_rel = _split_images_rel(out_dir, "val") or _split_images_rel(out_dir, "valid") or train_rel
        test_rel = _split_images_rel(out_dir, "test") or val_rel
    else:
        train_rel = _split_images_rel(out_dir, "train") or "train/images"
        val_rel = _split_images_rel(out_dir, "val") or _split_images_rel(out_dir, "valid") or train_rel
        test_rel = _split_images_rel(out_dir, "test") or val_rel
    p = Path(out_dir) / "data.yaml"
    p.write_text(
        f"train: {train_rel}\nval: {val_rel}\ntest: {test_rel}\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )


def _update_datasets_sidecar(
    layout: WorkspaceLayout,
    output_key: str,
    class_map: dict[str, int],
    target_dir: str,
    output_hash: str,
    structure: str,
) -> None:
    update_datasets_sidecar(
        layout=layout,
        output_key=output_key,
        class_map=class_map,
        target_dir=target_dir,
        output_hash=output_hash,
        structure=structure,
    )


def _list_workspace_detector_models(workspace_root: str) -> list[str]:
    exts = {".pt", ".onnx"}
    root = Path(workspace_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p.name)
    return out


def _augment_roi_prompt_label(*, enable_center_rotate: bool, enable_bbox_copy: bool) -> str:
    if enable_center_rotate and enable_bbox_copy:
        return "Placement mode for rotation pivot and bbox_copy paste area (--placement-mode)"
    if enable_center_rotate:
        return "Rotation pivot (--placement-mode)"
    return "Paste area for bbox_copy (--placement-mode)"


def _print_augment_placement_mode_help(*, enable_center_rotate: bool, enable_bbox_copy: bool) -> None:
    if enable_center_rotate and enable_bbox_copy:
        print("[INFO] Shared placement mode for rotation and bbox_copy:")
    elif enable_center_rotate:
        print("[INFO] Where to take the rotation axis from (not always the image center):")
    else:
        print("[INFO] Where bbox_copy may place pasted objects:")
    print("  none     - image center; simple ±rotation, no ROI detector needed")
    print("  bbox     - center of YOLO boxes on the frame")
    print("  detector - center/region from ROI detector (--roi-model), e.g. belt or conveyor zone")


def _default_placement_mode_for_interactive(*, enable_center_rotate: bool, enable_bbox_copy: bool, current: str) -> str:
    _ = enable_center_rotate, enable_bbox_copy
    cur = str(current or "none")
    if cur in {"none", "bbox", "detector"}:
        return cur
    return "none"


def _augment_balancing_block_title(*, enable_center_rotate: bool, enable_bbox_copy: bool) -> str:
    if enable_center_rotate and enable_bbox_copy:
        return "[INFO] Block: Balancing/Variety (center rotation and bbox_copy)"
    if enable_center_rotate:
        return "[INFO] Block: Balancing/Variety (center rotation)"
    return "[INFO] Block: Balancing/Variety (bbox_copy)"


def _interactive_fill(args, dataset_names: list[str], catalog: dict, workspace_root: str) -> None:
    print("[INFO] Interactive augment mode")
    args.dataset = prompt_choice("Dataset", dataset_names, default=dataset_names[0])
    class_names = sorted_class_names_for_dataset(catalog, str(args.dataset))
    if class_names:
        picked = prompt_multi_choice_csv(
            "Classes (--classes; empty=all)",
            class_names,
            default_values=[],
        )
        args.classes = ",".join(picked) if picked else None
    else:
        print("[WARN] No classes in dataset catalog; --classes filter unavailable.")
        args.classes = None
    output_default = str(args.output_name or "").strip()
    if output_default:
        args.output_name = prompt_prefilled_text("Output dataset name", output_default).strip() or None
    else:
        args.output_name = prompt_text("Output dataset name (empty=auto)", default="").strip() or None
    print("[INFO] Block: flip")
    args.enable_flip = prompt_yes_no("Turn on flip?", default=bool(args.enable_flip))
    if args.enable_flip:
        args.flip = prompt_choice(
            "Flip mode (--flip)",
            ["horizontal", "vertical", "both", "h-and-v", "none"],
            default=str(getattr(args, "flip", "horizontal")),
        )
        args.flip_sampling = prompt_choice(
            "Flip sampling (--flip-sampling)",
            ["probabilistic", "exhaustive"],
            default=str(getattr(args, "flip_sampling", "probabilistic")),
        )
        if str(args.flip_sampling) == "probabilistic":
            args.flip_prob = float(
                prompt_text("Probability of flip [0..1] (--flip-prob)", default=str(getattr(args, "flip_prob", 0.5))).strip()
                or str(getattr(args, "flip_prob", 0.5))
            )
    print("[INFO] Block: orthogonal rotate (±90°)")
    args.enable_orthogonal_rotate = prompt_yes_no(
        "Enable orthogonal ±90° rotate?",
        default=bool(getattr(args, "enable_orthogonal_rotate", False)),
    )
    if args.enable_orthogonal_rotate:
        args.orthogonal_rotate_sampling = prompt_choice(
            "Orthogonal sampling (--orthogonal-rotate-sampling)",
            ["probabilistic", "exhaustive"],
            default=str(getattr(args, "orthogonal_rotate_sampling", "probabilistic")),
        )
        if str(args.orthogonal_rotate_sampling) == "probabilistic":
            args.orthogonal_rotate_prob = float(
                prompt_text(
                    "Orthogonal probability [0..1] (--orthogonal-rotate-prob)",
                    default=str(getattr(args, "orthogonal_rotate_prob", 0.5)),
                ).strip()
                or str(getattr(args, "orthogonal_rotate_prob", 0.5))
            )
            args.orthogonal_rotate_direction = prompt_choice(
                "Orthogonal direction (--orthogonal-rotate-direction)",
                ["random", "cw", "ccw"],
                default=str(getattr(args, "orthogonal_rotate_direction", "random")),
            )
    args.enable_center_rotate = prompt_yes_no(
        "Enable frame rotation augmentation (--enable-center-rotate)?",
        default=bool(args.enable_center_rotate),
    )
    if args.enable_center_rotate:
        print("[INFO] Block: center-rotate")
        args.center_rotate_deg = float(
            prompt_text("Rotation angle limit (degrees, +-) (--center-rotate-deg)", default=str(getattr(args, "center_rotate_deg", 5.0))).strip()
            or str(getattr(args, "center_rotate_deg", 5.0))
        )
        args.rotate_copies = int(
            prompt_text("Number of rotate-options per frame (--rotate-copies)", default=str(getattr(args, "rotate_copies", 1))).strip()
            or str(getattr(args, "rotate_copies", 1))
        )
        args.min_angle_delta = float(
            prompt_text(
                (
                    "Min angle between center-rotate variants on one frame (degrees) "
                    "(--min-angle-delta); avoids nearly identical rotations"
                ),
                default=str(getattr(args, "min_angle_delta", 1.0)),
            ).strip()
            or str(getattr(args, "min_angle_delta", 1.0))
        )
    args.enable_bbox_copy = prompt_yes_no(
        "Enable bbox_copy paste augmentation (--enable-bbox-copy)?",
        default=bool(args.enable_bbox_copy),
    )
    if args.enable_bbox_copy:
        print("[INFO] Block: bbox_copy")
        args.class_balance = prompt_choice(
            "Class balance (--class-balance)",
            ["on", "off"],
            default=str(getattr(args, "class_balance", "on")),
        )
        args.color_match = prompt_choice(
            "Color match (--color-match)",
            ["meanstd", "off"],
            default=str(getattr(args, "color_match", "meanstd")),
        )
        args.blend_feather = float(
            prompt_text("Parameter feather [0..0.5] (--blend-feather)", default=str(getattr(args, "blend_feather", 0.16))).strip()
            or str(getattr(args, "blend_feather", 0.16))
        )
        args.copy_paste_count = int(
            prompt_text("Copy-paste count (--copy-paste-count)", default=str(args.copy_paste_count)).strip()
            or str(args.copy_paste_count)
        )
        args.copy_paste_min_center_dist = float(
            prompt_text(
                "Min. distance between pastes [0..1] (--copy-paste-min-center-dist)",
                default=str(getattr(args, "copy_paste_min_center_dist", 0.15)),
            ).strip()
            or str(getattr(args, "copy_paste_min_center_dist", 0.15))
        )
        args.copy_paste_placement_style = prompt_choice(
            "Paste placement style (--copy-paste-placement-style)",
            ["random", "uniform-grid"],
            default=str(getattr(args, "copy_paste_placement_style", "random")),
        )
        args.bbox_copy_copies = int(
            prompt_text("Number of bbox_copy-options per frame (--bbox-copy-copies)", default=str(getattr(args, "bbox_copy_copies", 1))).strip()
            or str(getattr(args, "bbox_copy_copies", 1))
        )
    if args.enable_center_rotate or args.enable_bbox_copy:
        print("[INFO] Block: placement / ROI")
        _print_augment_placement_mode_help(
            enable_center_rotate=bool(args.enable_center_rotate),
            enable_bbox_copy=bool(args.enable_bbox_copy),
        )
        roi_mode_default = _default_placement_mode_for_interactive(
            enable_center_rotate=bool(args.enable_center_rotate),
            enable_bbox_copy=bool(args.enable_bbox_copy),
            current=str(getattr(args, "placement_mode", "none")),
        )
        roi_mode = prompt_choice(
            _augment_roi_prompt_label(
                enable_center_rotate=bool(args.enable_center_rotate),
                enable_bbox_copy=bool(args.enable_bbox_copy),
            ),
            ["none", "bbox", "detector"],
            default=roi_mode_default,
        )
        args.placement_mode = roi_mode
        args.center_rotate_anchor = {"none": "center", "bbox": "bbox", "detector": "detector"}[roi_mode]
        if roi_mode == "detector":
            models = _list_workspace_detector_models(workspace_root)
            if models:
                print("[INFO] ROI detectors in the workspace root:")
                for m in models:
                    print(f"  - {m}")
            args.roi_model = prompt_text(
                "ROI detector model (--roi-model)",
                default=str(getattr(args, "roi_model", "yolo11n.pt")),
                choices=models if models else None,
            ).strip() or str(getattr(args, "roi_model", "yolo11n.pt"))
            args.roi_conf = float(
                prompt_text("ROI threshold conf (--roi-conf)", default=str(getattr(args, "roi_conf", 0.25))).strip()
                or str(getattr(args, "roi_conf", 0.25))
            )
            args.roi_class_ids = (
                prompt_text("ROI class ids CSV (--roi-class-ids, empty=all)", default=str(getattr(args, "roi_class_ids", "") or "")).strip()
                or None
            )
    print("[INFO] Block: photometric/conveyor")
    args.enable_photometric = prompt_yes_no("Enable brightness/contrast?", default=bool(args.enable_photometric))
    args.enable_conveyor_rotate = prompt_yes_no(
        "Enable conveyor rotate (±5°)?",
        default=bool(getattr(args, "enable_conveyor_rotate", False)),
    )
    args.enable_conveyor_scale = prompt_yes_no(
        "Enable conveyor scale (0.95–1.05)?",
        default=bool(getattr(args, "enable_conveyor_scale", False)),
    )
    args.enable_conveyor_blur = prompt_yes_no(
        "Enable conveyor motion blur?",
        default=bool(getattr(args, "enable_conveyor_blur", False)),
    )
    args.enable_conveyor_shift = prompt_yes_no(
        "Enable conveyor shift (±3% translate)?",
        default=bool(getattr(args, "enable_conveyor_shift", False)),
    )
    args.enable_conveyor_noise = prompt_yes_no(
        "Enable conveyor noise?",
        default=False,
    )
    if args.enable_conveyor_noise:
        args.conveyor_noise_types = prompt_text(
            "Noise types CSV (--conveyor-noise-types)",
            default=str(getattr(args, "conveyor_noise_types", "iso,shot,gaussian")),
        ).strip() or "iso,shot,gaussian"
        args.conveyor_noise_intensity = float(
            prompt_text(
                "Noise intensity [0..1] (--conveyor-noise-intensity)",
                default=str(getattr(args, "conveyor_noise_intensity", 0.35)),
            ).strip()
            or str(getattr(args, "conveyor_noise_intensity", 0.35))
        )
        args.conveyor_noise_selection = prompt_choice(
            "Noise selection (--conveyor-noise-selection)",
            ["random", "stack"],
            default=str(getattr(args, "conveyor_noise_selection", "random")),
        )
    args.enable_conveyor = _conveyor_any(args)
    imbalance_soft = str(getattr(args, "imbalance_mode", "soft")) == "soft"
    geo_copies_enabled = bool(args.enable_center_rotate or args.enable_bbox_copy)
    if geo_copies_enabled:
        print(_augment_balancing_block_title(
            enable_center_rotate=bool(args.enable_center_rotate),
            enable_bbox_copy=bool(args.enable_bbox_copy),
        ))
        args.imbalance_mode = prompt_choice(
            (
                "Class balancing (--imbalance-mode): off=equal copies per frame; "
                "soft=extra rotate/bbox_copy on frames with rare classes"
            ),
            ["off", "soft"],
            default=str(getattr(args, "imbalance_mode", "soft")),
        )
        imbalance_soft = str(args.imbalance_mode) == "soft"
        if imbalance_soft:
            args.imbalance_strength = float(
                prompt_text(
                    (
                        "Balancing strength (>=0) (--imbalance-strength): scales extra "
                        "rotate/bbox_copy on rare-class frames (0=none, 1=default, higher=stronger)"
                    ),
                    default=str(getattr(args, "imbalance_strength", 1.0)),
                ).strip()
                or str(getattr(args, "imbalance_strength", 1.0))
            )
        print("[INFO] Sub-block: deduplication (drop near-duplicate variants)")
        args.min_diversity_iou = float(
            prompt_text(
                (
                    "Skip variant if label IoU vs any saved variant >= threshold [0..1] "
                    "(--min-diversity-iou); higher rejects more (0.97≈almost identical)"
                ),
                default=str(getattr(args, "min_diversity_iou", 0.97)),
            ).strip()
            or str(getattr(args, "min_diversity_iou", 0.97))
        )
    print("[INFO] Block: budget / class-aware")
    if geo_copies_enabled and not imbalance_soft:
        print("[INFO] Class-aware geo augment skipped (--imbalance-mode off)")
        args.aug_class_aware_geo = False
    else:
        args.aug_class_aware_geo = prompt_yes_no(
            (
                "Reduce flip/photo/conveyor rate on majority-class frames "
                "(--aug-class-aware-geo; requires --imbalance-mode soft)?"
            ),
            default=bool(getattr(args, "aug_class_aware_geo", False)),
        )
    args.aug_total_bbox_cap_mult = float(
        prompt_text(
            (
                "Train bbox budget: max total bbox = mult × baseline B₀ "
                "(--aug-total-bbox-cap-mult; 0=unlimited, 1.0=no extra bbox beyond baseline)"
            ),
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
    args.splits = prompt_text("Splits separated by commas (train,val,test)", default=args.splits).strip() or args.splits
    args.dry_run = prompt_yes_no("Do dry-run (--dry-run)?", default=bool(args.dry_run))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_augment_arg_parser()
    args = parser.parse_args(argv)
    provided_flags = _provided_augment_flags(argv)
    _apply_augment_preset_defaults(args, provided_flags)
    _sync_conveyor_flags(args, argv=argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    atexit.register(lambda wr=root: best_effort_prune_workspace_runs_detect(wr))
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
            catalog,
            layout.root,
        ),
    )
    _sync_conveyor_flags(args)

    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    norm_err = _normalize_augment_args(args, argv=argv)
    if norm_err:
        print(norm_err)
        return
    _warn_exhaustive_class_aware(args)
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("augment", parser, args)
        print_replay_command("before launch", replay_cmd)

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
    if "val" in split_filter:
        split_filter.add("valid")
    if "valid" in split_filter:
        split_filter.add("val")
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
    source_structure = str(entry.get("structure", "split"))
    output_structure = augment_output_structure(source_structure)
    copied = 0
    augmented = 0
    skipped_roi_missing = 0
    items: list[dict[str, str]] = []
    for images_path, labels_path in buckets:
        split = _detect_split(images_path)
        for file_name in os.listdir(images_path):
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            items.append(
                {
                    "split": split,
                    "stem": stem,
                    "ext": ext,
                    "img": os.path.join(images_path, file_name),
                    "lbl": os.path.join(labels_path, f"{stem}.txt"),
                }
            )
    if not items:
        print("[ERROR] No images found for augment.")
        return
    label_kinds: set[str] = set()
    for it in items:
        kind = resolve_label_kind(it["lbl"], label_type=str(getattr(args, "label_type", "auto")))
        if kind != "empty":
            label_kinds.add(kind)
    if "segment" in label_kinds or "mixed" in label_kinds:
        if bool(args.enable_bbox_copy):
            print("[ERROR] --enable-bbox-copy is not supported for polygon (segmentation) label files.")
            return
        if "mixed" in label_kinds:
            print(
                "[WARN] Mixed bbox and polygon labels detected; geometric augment applies both bbox and keypoint transforms."
            )
    if not args.dry_run:
        splits_present = sorted({str(it["split"]) for it in items}) or ["train", "val", "test"]
        _augment_ensure_base_dirs(out_dir, output_structure, splits_present)
    donors = _build_donor_pool(items, args) if args.enable_bbox_copy else []
    detector_roi_cache: dict[str, tuple[int, int, int, int] | None] = {}
    need_detector_for_rotate = bool(args.enable_center_rotate) and str(
        getattr(args, "center_rotate_anchor", "center")
    ) == "detector"
    need_detector_for_copy = bool(args.enable_bbox_copy) and str(getattr(args, "placement_mode", "none")) == "detector"
    if need_detector_for_rotate or need_detector_for_copy:
        setattr(
            args,
            "_ultralytics_roi_predict_project",
            ultralytics_sidecar_dir(layout.root, ".cache", "ultralytics_augment_roi"),
        )
        detector_roi_cache = _build_detector_roi_cache(items, args)
    class_usage: dict[int, int] = {}
    class_freq = _collect_class_freq(items)
    alpha = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    train_baseline_bbox = _train_split_bbox_sum(items)
    cap_mult = float(getattr(args, "aug_total_bbox_cap_mult", 0.0))
    bbox_cap_total: int | None = None
    if cap_mult > 0 and train_baseline_bbox > 0:
        bbox_cap_total = int(math.ceil(cap_mult * train_baseline_bbox))
    extra_budget: int | None = None
    if bbox_cap_total is not None:
        extra_budget = max(0, bbox_cap_total - train_baseline_bbox)
    per_class_cap_mult = float(getattr(args, "aug_per_class_bbox_cap_mult", 0.0))
    per_class_extra_budget: dict[int, int] | None = None
    train_extra_bbox_by_class: dict[int, int] = defaultdict(int)
    if per_class_cap_mult > 0 and train_cls_counts:
        per_class_extra_budget = {
            c: max(0, int(math.ceil(per_class_cap_mult * int(n))) - int(n)) for c, n in train_cls_counts.items()
        }
    train_extra_bbox_used = 0
    train_cls_counts = _train_split_class_bbox_counts(items)
    items = _reorder_items_for_bbox_budget(
        items,
        split_filter=split_filter,
        cls_counts=train_cls_counts,
        extra_budget=extra_budget,
        tail_first=bool(getattr(args, "aug_budget_tail_first", True)),
        gamma=float(getattr(args, "aug_budget_tail_gamma", 1.0)),
    )
    progress = tqdm(
        items,
        total=len(items),
        desc="augment",
        unit="img",
        disable=bool(args.no_legend),
    )
    for it in progress:
        split = it["split"]
        stem = it["stem"]
        ext = it["ext"]
        img_src = it["img"]
        lbl_src = it["lbl"]
        split_norm = SPLIT_ALIASES.get(str(split).strip().lower(), str(split).strip().lower())
        classes_in_image = _read_yolo_classes(lbl_src)
        class_names = {names_by_id.get(i, f"id_{i}") for i in classes_in_image}
        if allowed_classes and class_names.isdisjoint(allowed_classes):
            continue
        bi = count_yolo_bbox_lines(lbl_src)
        if not args.dry_run:
            dst_img, dst_lbl = _augment_output_paths(
                out_dir, output_structure, src_root, img_src, lbl_src, stem, ext
            )
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
            shutil.copy2(img_src, dst_img)
            if os.path.isfile(lbl_src):
                shutil.copy2(lbl_src, dst_lbl)
            else:
                Path(dst_lbl).write_text("", encoding="utf-8")
        copied += 1
        if split not in split_filter:
            continue

        image_weight = _image_soft_weight(classes_in_image, class_freq, alpha)
        image_class_counts = _label_class_counts(lbl_src)
        seen_labels: list[list[tuple[int, float, float, float, float]]] = []

        def _budget_ok(delta_total: int, delta_by_class: dict[int, int] | None = None) -> bool:
            if split_norm != "train":
                return True
            if args.dry_run:
                return True
            if delta_by_class and per_class_extra_budget is not None:
                for c, d in delta_by_class.items():
                    if d <= 0:
                        continue
                    if train_extra_bbox_by_class.get(c, 0) + d > per_class_extra_budget.get(c, 0):
                        return False
            return _aug_extra_budget_allow(train_extra_bbox_used, delta_total, extra_budget)

        def _consume_extra(delta_total: int, delta_by_class: dict[int, int] | None = None) -> None:
            nonlocal train_extra_bbox_used
            if split_norm != "train" or args.dry_run:
                return
            train_extra_bbox_used += delta_total
            if delta_by_class:
                for c, d in delta_by_class.items():
                    if d > 0:
                        train_extra_bbox_by_class[c] = train_extra_bbox_by_class.get(c, 0) + d

        crc_img = zlib.crc32(img_src.encode("utf-8"))
        flip_rng = random.Random(int(args.seed) + crc_img)
        flip_p = _effective_flip_prob_geo(args, image_weight)

        for fi, flip_spec in enumerate(_iter_flip_variants(args, flip_rng, flip_prob=flip_p), start=1):
            if not _budget_ok(bi, image_class_counts):
                break
            aug_stem = _aug_stem(stem, args, fi, "f", flip_tag=flip_spec.tag)
            if args.dry_run:
                augmented += 1
                continue
            out_img, out_lbl = _augment_output_paths(
                out_dir, output_structure, src_root, img_src, lbl_src, aug_stem, ext
            )
            os.makedirs(os.path.dirname(out_img), exist_ok=True)
            os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
            new_labels = _apply_geom_aug(
                img_src,
                lbl_src,
                out_img,
                out_lbl,
                args,
                enable_flip=True,
                enable_photometric=False,
                enable_conveyor=False,
                enable_center_rotate=False,
                flip_mode=flip_spec.mode,
            )
            if any(
                _labels_signature_iou(prev, new_labels, 1000, 1000)
                >= float(getattr(args, "min_diversity_iou", 0.97))
                for prev in seen_labels
            ):
                try:
                    os.remove(out_img)
                    os.remove(out_lbl)
                except OSError:
                    pass
                continue
            seen_labels.append(new_labels)
            if split_norm == "train":
                _consume_extra(bi, image_class_counts)
            augmented += 1

        photo_rng = random.Random(int(args.seed) + 901 + crc_img)
        if (args.enable_photometric or args.enable_conveyor) and _geo_photo_trigger(args, image_weight, photo_rng):
            geom_mode = "c" if args.enable_conveyor else "b"
            aug_stem = _aug_stem(stem, args, 1, geom_mode)
            if _budget_ok(bi, image_class_counts):
                if not args.dry_run:
                    out_img, out_lbl = _augment_output_paths(
                        out_dir, output_structure, src_root, img_src, lbl_src, aug_stem, ext
                    )
                    os.makedirs(os.path.dirname(out_img), exist_ok=True)
                    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
                    new_labels = _apply_geom_aug(
                        img_src,
                        lbl_src,
                        out_img,
                        out_lbl,
                        args,
                        enable_flip=False,
                        enable_photometric=bool(args.enable_photometric),
                        enable_conveyor=bool(args.enable_conveyor),
                        enable_center_rotate=False,
                    )
                    if any(
                        _labels_signature_iou(prev, new_labels, 1000, 1000)
                        >= float(getattr(args, "min_diversity_iou", 0.97))
                        for prev in seen_labels
                    ):
                        try:
                            os.remove(out_img)
                            os.remove(out_lbl)
                        except OSError:
                            pass
                    else:
                        seen_labels.append(new_labels)
                        if split_norm == "train":
                            _consume_extra(bi, image_class_counts)
                        augmented += 1
                else:
                    augmented += 1

        orth_rng = random.Random(int(args.seed) + 7001 + crc_img)
        orth_p = _effective_orthogonal_prob_geo(args, image_weight)
        for oi, orth_spec in enumerate(_iter_orthogonal_variants(args, orth_rng, orth_prob=orth_p), start=1):
            if not _budget_ok(bi, image_class_counts):
                break
            aug_stem = _aug_stem(stem, args, oi, "o", orth_tag=orth_spec.tag)
            if args.dry_run:
                augmented += 1
                continue
            out_img, out_lbl = _augment_output_paths(
                out_dir, output_structure, src_root, img_src, lbl_src, aug_stem, ext
            )
            new_labels = apply_orthogonal_rotate(
                img_src,
                lbl_src,
                out_img,
                out_lbl,
                direction=orth_spec.direction,
            )
            orth_tot = len(new_labels)
            if not _budget_ok(orth_tot, image_class_counts):
                try:
                    os.remove(out_img)
                    os.remove(out_lbl)
                except OSError:
                    pass
                continue
            if any(
                _labels_signature_iou(prev, new_labels, 1000, 1000)
                >= float(getattr(args, "min_diversity_iou", 0.97))
                for prev in seen_labels
            ):
                try:
                    os.remove(out_img)
                    os.remove(out_lbl)
                except OSError:
                    pass
                continue
            seen_labels.append(new_labels)
            if split_norm == "train":
                _consume_extra(orth_tot, image_class_counts)
            augmented += 1

        if args.enable_center_rotate:
            rot_copies = _scaled_copies(int(getattr(args, "rotate_copies", 1)), image_weight, args)
            path_seed = zlib.crc32(img_src.encode("utf-8")) & 0xFFFFFFFF
            rng_rot = random.Random(int(args.seed) + 5003 + path_seed)
            used_angles: list[float] = []
            rot_saved = 0
            tries_left = max(1, rot_copies * 8)
            while rot_saved < rot_copies and tries_left > 0:
                tries_left -= 1
                angle = rng_rot.uniform(
                    -float(getattr(args, "center_rotate_deg", 5.0)),
                    float(getattr(args, "center_rotate_deg", 5.0)),
                )
                if any(abs(angle - prev) < float(getattr(args, "min_angle_delta", 1.0)) for prev in used_angles):
                    continue
                used_angles.append(angle)
                aug_stem = _aug_stem(stem, args, rot_saved + 1, "r")
                if not args.dry_run:
                    out_img, out_lbl = _augment_output_paths(
                        out_dir, output_structure, src_root, img_src, lbl_src, aug_stem, ext
                    )
                    os.makedirs(os.path.dirname(out_img), exist_ok=True)
                    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
                    new_labels = _apply_exact_center_rotate(
                        img_src,
                        lbl_src,
                        out_img,
                        out_lbl,
                        args,
                        angle,
                        detector_roi=detector_roi_cache.get(img_src),
                    )
                    if new_labels is None:
                        if args.placement_mode in ("bbox", "detector"):
                            skipped_roi_missing += 1
                        continue
                    rot_tot = len(new_labels)
                    if not _budget_ok(rot_tot, image_class_counts):
                        try:
                            os.remove(out_img)
                            os.remove(out_lbl)
                        except OSError:
                            pass
                        continue
                    if any(
                        _labels_signature_iou(prev, new_labels, 1000, 1000)
                        >= float(getattr(args, "min_diversity_iou", 0.97))
                        for prev in seen_labels
                    ):
                        try:
                            os.remove(out_img)
                            os.remove(out_lbl)
                        except OSError:
                            pass
                        continue
                    seen_labels.append(new_labels)
                    if split_norm == "train":
                        _consume_extra(rot_tot, image_class_counts)
                    rot_saved += 1
                    augmented += 1
                else:
                    rot_saved += 1
                    augmented += 1

        if args.enable_bbox_copy:
            cp_copies = _scaled_copies(int(getattr(args, "bbox_copy_copies", 1)), image_weight, args)
            for i in range(cp_copies):
                aug_stem = _aug_stem(stem, args, i + 1, "p")
                if not args.dry_run:
                    out_img, out_lbl = _augment_output_paths(
                        out_dir, output_structure, src_root, img_src, lbl_src, aug_stem, ext
                    )
                    os.makedirs(os.path.dirname(out_img), exist_ok=True)
                    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
                    ok, new_labels = _apply_copy_paste(
                        img_src,
                        lbl_src,
                        out_img,
                        out_lbl,
                        args,
                        donors,
                        class_usage,
                        variant_seed=i + 1,
                        class_freq=class_freq,
                        detector_roi=detector_roi_cache.get(img_src),
                    )
                    if ok:
                        delta_bb, delta_by_class = _inserted_class_delta(image_class_counts, new_labels)
                        if any(
                            _labels_signature_iou(prev, new_labels, 1000, 1000)
                            >= float(getattr(args, "min_diversity_iou", 0.97))
                            for prev in seen_labels
                        ):
                            try:
                                os.remove(out_img)
                                os.remove(out_lbl)
                            except OSError:
                                pass
                            continue
                        if not _budget_ok(delta_bb, delta_by_class):
                            try:
                                os.remove(out_img)
                                os.remove(out_lbl)
                            except OSError:
                                pass
                            continue
                        seen_labels.append(new_labels)
                        if split_norm == "train":
                            _consume_extra(delta_bb, delta_by_class)
                        augmented += 1
                    elif args.placement_mode in ("bbox", "detector"):
                        skipped_roi_missing += 1
                else:
                    augmented += 1
        progress.set_postfix(
            copied=copied,
            augmented=augmented,
            skipped_roi=skipped_roi_missing,
            refresh=False,
        )
    progress.close()

    if args.dry_run:
        print(f"[OK] dry-run: copied={copied}, augmented={augmented}, output={out_name}")
        if replay_cmd:
            print_replay_command("after execution", replay_cmd)
        return

    all_names = [str(x) for _, x in sorted(names_by_id.items())]
    _write_data_yaml(out_dir, all_names, structure=source_structure)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(
        layout,
        out_name,
        class_map if isinstance(class_map, dict) else {},
        out_dir,
        out_hash,
        output_structure,
    )
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
        workspace_root=layout.root,
        transformations=[
            {
                "enable_flip": bool(args.enable_flip),
                "flip_prob": float(getattr(args, "flip_prob", 0.5)),
                "flip": args.flip,
                "flip_sampling": str(getattr(args, "flip_sampling", "probabilistic")),
                "enable_orthogonal_rotate": bool(getattr(args, "enable_orthogonal_rotate", False)),
                "orthogonal_rotate_sampling": str(getattr(args, "orthogonal_rotate_sampling", "probabilistic")),
                "orthogonal_rotate_prob": float(getattr(args, "orthogonal_rotate_prob", 0.5)),
                "orthogonal_rotate_direction": str(getattr(args, "orthogonal_rotate_direction", "random")),
                "enable_photometric": bool(args.enable_photometric),
                "enable_conveyor": bool(args.enable_conveyor),
                "enable_conveyor_rotate": bool(getattr(args, "enable_conveyor_rotate", False)),
                "enable_conveyor_scale": bool(getattr(args, "enable_conveyor_scale", False)),
                "enable_conveyor_blur": bool(getattr(args, "enable_conveyor_blur", False)),
                "enable_conveyor_shift": bool(getattr(args, "enable_conveyor_shift", False)),
                "enable_conveyor_noise": bool(getattr(args, "enable_conveyor_noise", False)),
                "conveyor_noise_types": str(getattr(args, "conveyor_noise_types", "iso,shot,gaussian")),
                "conveyor_noise_intensity": float(getattr(args, "conveyor_noise_intensity", 0.35)),
                "conveyor_noise_selection": str(getattr(args, "conveyor_noise_selection", "random")),
                "enable_center_rotate": bool(args.enable_center_rotate),
                "center_rotate_deg": float(getattr(args, "center_rotate_deg", 5.0)),
                "rotate_copies": int(getattr(args, "rotate_copies", 1)),
                "center_rotate_anchor": str(getattr(args, "center_rotate_anchor", "center")),
                "enable_bbox_copy": bool(args.enable_bbox_copy),
                "bbox_copy_copies": int(getattr(args, "bbox_copy_copies", 1)),
                "copy_paste_min_center_dist": float(getattr(args, "copy_paste_min_center_dist", 0.15)),
                "copy_paste_placement_style": str(getattr(args, "copy_paste_placement_style", "random")),
                "placement_mode": args.placement_mode,
                "imbalance_mode": str(getattr(args, "imbalance_mode", "soft")),
                "imbalance_strength": float(getattr(args, "imbalance_strength", 1.0)),
                "aug_class_aware_geo": bool(getattr(args, "aug_class_aware_geo", False)),
                "aug_total_bbox_cap_mult": float(getattr(args, "aug_total_bbox_cap_mult", 0.0)),
                "aug_budget_tail_first": bool(getattr(args, "aug_budget_tail_first", True)),
                "aug_budget_tail_gamma": float(getattr(args, "aug_budget_tail_gamma", 1.0)),
                "train_bbox_baseline_before_augment": train_baseline_bbox,
                "train_bbox_budget_cap": bbox_cap_total,
                "train_bbox_extra_budget": extra_budget,
                "splits": sorted(split_filter),
            }
        ],
        random_seed=args.seed,
        stats_before={"copied_input_images": copied},
        stats_after={
            "copied_images": copied,
            "augmented_images": augmented,
            "skipped_roi_missing": skipped_roi_missing,
            "train_bbox_extra_used_after_augment": train_extra_bbox_used,
            "train_bbox_total_on_disk": sum_train_bbox_disk(out_dir),
            "output_hash": out_hash,
        },
    )
    print(f"[OK] Dataset created: {out_dir}")
    print(f"[OK] Passport: {passport_path}")
    if replay_cmd:
        print_replay_command("after execution", replay_cmd)


if __name__ == "__main__":
    main()

