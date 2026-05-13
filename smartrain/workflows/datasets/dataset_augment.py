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
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from tqdm import tqdm
from ultralytics import YOLO

from smartrain.cli_support.cli_argparse import CliArgumentParser
from smartrain.cli_support.cli_prompts import prompt_choice, prompt_text, prompt_yes_no
from smartrain.cli_support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.workflows.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.workflows.datasets.dataset_hash import calculate_dataset_hash
from smartrain.workflows.datasets.dataset_passport import next_dataset_name, write_dataset_passport
from smartrain.workflows.datasets.dataset_cli_common import (
    detect_split_from_path,
    load_dataset_catalog,
    update_datasets_sidecar,
)
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_workspace_runs_detect, ultralytics_sidecar_dir
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "valid", "test": "test"}
_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_ROI_MODEL_CACHE: dict[str, YOLO] = {}


def build_augment_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Offline augmentation of a dataset into a new datasets/<name>")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Name of source dataset from datasets_info.json")
    p.add_argument("--output-name", type=str, default=None, help="Name of output dataset (default <dataset>_aug)")
    p.add_argument("--enable-flip", action="store_true", help="Enable flip augmentation")
    p.add_argument("--flip-prob", type=float, default=0.5, help="Probability of creating a flip variant per frame [0..1]")
    p.add_argument("--enable-photometric", action="store_true", help="Enable brightness/contrast")
    p.add_argument("--enable-conveyor", action="store_true", help="Enable pipeline noise/blur/shift/rotate")
    p.add_argument("--enable-center-rotate", action="store_true", dest="enable_center_rotate", help="Enable frame rotation around the center")
    p.add_argument("--disable-center-rotate", action="store_false", dest="enable_center_rotate", help="Disable frame rotation around center")
    p.set_defaults(enable_center_rotate=True)
    p.add_argument("--center-rotate-deg", type=float, default=5.0, help="Maximum rotation angle in both directions")
    p.add_argument("--rotate-copies", type=int, default=1, help="Number of rotate options per frame")
    p.add_argument(
        "--center-rotate-anchor",
        choices=("center", "bbox", "detector"),
        default="center",
        help="Rotation center source: frame center, bbox markup or ROI detector",
    )
    p.add_argument("--enable-bbox-copy", action="store_true", help="Enable bbox-copy augmentation")
    p.add_argument("--bbox-copy-copies", type=int, default=1, help="Number of bbox_copy options per frame")
    p.add_argument("--flip", choices=("horizontal", "vertical", "both", "none"), default="horizontal")
    p.add_argument("--brightness-limit", type=float, default=0.1, help="For policy=basic: brightness range")
    p.add_argument("--contrast-limit", type=float, default=0.1, help="For policy=basic: range contrast")
    p.add_argument("--copy-paste-count", type=int, default=1, help="For policy=bbox_copy: number of inserts per image")
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
    p.add_argument(
        "--placement-mode",
        choices=("none", "bbox", "detector"),
        default="detector",
        help="ROI placement mode for bbox_copy: none|bbox|detector (default detector)",
    )
    p.add_argument("--placement-roi", action="store_true", help="Legacy: same as --placement-mode bbox")
    p.add_argument("--roi-model", type=str, default="yolo11n.pt", help="ROI detector model for --placement-mode detector")
    p.add_argument("--roi-conf", type=float, default=0.25, help="Confidence threshold for ROI detector")
    p.add_argument("--roi-class-ids", type=str, default=None, help="CSV class ids for ROI detector (empty=all)")
    p.add_argument("--side-tolerance-px", type=float, default=3.0, help="Tolerance in px for ROI side classification")
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
    p.add_argument("--min-diversity-iou", type=float, default=0.97, help="bbox similarity threshold (higher -> almost duplicate)")
    p.add_argument("--min-angle-delta", type=float, default=1.0, help="Minimum angle difference between rotate options")
    p.add_argument("--splits", type=str, default="train", help="CSV: train,val,test")
    p.add_argument("--classes", type=str, default=None, help="Limit augmentation to CSV classes")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-legend", action="store_true")
    return p


def _load_catalog(layout: WorkspaceLayout) -> dict:
    return load_dataset_catalog(layout)


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
    out: list[tuple[int, float, float, float, float]] = []
    if not os.path.isfile(label_path):
        return out
    for raw in Path(label_path).read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        except ValueError:
            continue
    return out


def _serialize_yolo_labels(labels: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for cls, x, y, w, h in labels)


def _compose_for_basic(args) -> A.Compose:
    t: list[A.BasicTransform] = []
    if args.enable_flip:
        if args.flip == "horizontal":
            t.append(A.HorizontalFlip(p=1.0))
        elif args.flip == "vertical":
            t.append(A.VerticalFlip(p=1.0))
        elif args.flip == "both":
            t.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))
    if args.enable_conveyor:
        t.extend(
            [
                A.Affine(
                    translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                    scale=(0.95, 1.05),
                    rotate=(-5, 5),
                    border_mode=0,
                    p=0.8,
                ),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=3, p=0.15),
            ]
        )
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
    return A.Compose(
        t,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True),
    )


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
    if args.enable_conveyor:
        return "s"
    if args.enable_photometric:
        return "l"
    return "n"


def _flip_code(args) -> str:
    if not args.enable_flip:
        return "n"
    return {"horizontal": "h", "vertical": "v", "both": "b", "none": "n"}[args.flip]


def _aug_stem(stem: str, args, idx: int, mode: str) -> str:
    return f"{stem}__a-{mode}{_flip_code(args)}{_variant_code(args)}{_base36(idx)}"


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


def _collect_class_freq(items: list[dict]) -> dict[int, int]:
    freq: dict[int, int] = {}
    for it in items:
        for cls, *_ in _parse_yolo_labels(it["lbl"]):
            freq[int(cls)] = freq.get(int(cls), 0) + 1
    return freq


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
    """Number of bbox lines (non-empty) in a YOLO label file."""
    if not os.path.isfile(lbl_path):
        return 0
    n = 0
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().split():
                n += 1
    return n


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


def _effective_flip_prob_geo(args, image_weight: float) -> float:
    """Higher probability for tail-heavy frames when aug-class-aware-geo + imbalance soft."""
    base = float(getattr(args, "flip_prob", 0.5))
    if not bool(getattr(args, "aug_class_aware_geo", False)):
        return base
    if str(getattr(args, "imbalance_mode", "soft")) != "soft":
        return base
    strength = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    w = max(1e-9, float(image_weight))
    scale = math.sqrt(w) * max(0.35, min(2.5, strength))
    p = base * scale
    return float(min(1.0, max(0.02, p)))


def _geo_photo_trigger(args, image_weight: float, rng: random.Random) -> bool:
    """Whether to emit photometric/conveyor variant for this frame (class-aware)."""
    if not bool(getattr(args, "aug_class_aware_geo", False)):
        return True
    if str(getattr(args, "imbalance_mode", "soft")) != "soft":
        return True
    strength = max(0.0, float(getattr(args, "imbalance_strength", 1.0)))
    w = float(image_weight)
    p = min(1.0, max(0.0, (w * strength) / (1.0 + strength)))
    return bool(rng.random() < p)


def _aug_extra_budget_allow(extra_used: int, delta: int, extra_budget: int | None) -> bool:
    """Extra augment bbox must stay within extra_budget (cap_total − baseline B₀)."""
    if extra_budget is None:
        return True
    return extra_used + delta <= extra_budget


def sum_train_bbox_disk(dataset_root: str) -> int:
    """Sum bbox lines across train/labels/*.txt under dataset_root."""
    lbl_dir = os.path.join(dataset_root, "train", "labels")
    if not os.path.isdir(lbl_dir):
        return 0
    s = 0
    for name in os.listdir(lbl_dir):
        if not name.endswith(".txt"):
            continue
        s += count_yolo_bbox_lines(os.path.join(lbl_dir, name))
    return s


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
) -> list[tuple[int, float, float, float, float]]:
    image = np.array(Image.open(image_path).convert("RGB"))
    raw_labels = _parse_yolo_labels(label_path)
    labels = [x for x in (_sanitize_yolo_box(lb) for lb in raw_labels) if x is not None]
    bboxes = [(x, y, w, h) for _, x, y, w, h in labels]
    class_labels = [cls for cls, *_ in labels]
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
        local.enable_conveyor = bool(enable_conveyor)
    if enable_center_rotate is not None:
        local.enable_center_rotate = bool(enable_center_rotate)

    pipeline = _compose_for_basic(local)
    transformed = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    new_img = transformed["image"]
    new_labels_raw = [
        (int(cls), float(x), float(y), float(w), float(h))
        for cls, (x, y, w, h) in zip(transformed["class_labels"], transformed["bboxes"])
    ]
    new_labels = [x for x in (_sanitize_yolo_box(lb) for lb in new_labels_raw) if x is not None]
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(new_img).save(out_img)
    Path(out_lbl).write_text(_serialize_yolo_labels(new_labels), encoding="utf-8")
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
    labels = _parse_yolo_labels(label_path)
    rotate_anchor = str(getattr(args, "center_rotate_anchor", "detector"))
    rotate_use_roi = rotate_anchor in ("bbox", "detector")
    roi: tuple[int, int, int, int] | None = None
    if rotate_use_roi and rotate_anchor == "bbox":
        roi = _roi_from_labels(labels, w, h)
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
    new_labels: list[tuple[int, float, float, float, float]] = []
    for lb in labels:
        cls, *_ = lb
        x1, y1, x2, y2 = _to_xyxy(lb, w, h)
        corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float32)
        tr = (m @ corners.T).T
        nx1 = max(0, min(w, int(np.floor(np.min(tr[:, 0])))))
        ny1 = max(0, min(h, int(np.floor(np.min(tr[:, 1])))))
        nx2 = max(0, min(w, int(np.ceil(np.max(tr[:, 0])))))
        ny2 = max(0, min(h, int(np.ceil(np.max(tr[:, 1])))))
        if nx2 <= nx1 or ny2 <= ny1:
            continue
        new_labels.append(_to_yolo(int(cls), (nx1, ny1, nx2, ny2), w, h))
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)
    Image.fromarray(dst).save(out_img)
    Path(out_lbl).write_text(_serialize_yolo_labels(new_labels), encoding="utf-8")
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
    placement_mode = str(getattr(args, "placement_mode", "detector"))
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


def _write_data_yaml(out_dir: str, names: list[str]) -> None:
    p = Path(out_dir) / "data.yaml"
    val_rel = "valid/images" if (Path(out_dir) / "valid" / "images").is_dir() else "val/images"
    p.write_text(
        f"train: train/images\nval: {val_rel}\ntest: test/images\n\n"
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
) -> None:
    update_datasets_sidecar(
        layout=layout,
        output_key=output_key,
        class_map=class_map,
        target_dir=target_dir,
        output_hash=output_hash,
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


def _interactive_fill(args, dataset_names: list[str], classes: list[str], workspace_root: str) -> None:
    print("[INFO] Interactive augment mode")
    print("[INFO] Available classes:")
    for c in classes:
        print(f"  - {c}")
    args.dataset = prompt_choice("Dataset", dataset_names, default=dataset_names[0])
    args.classes = (
        prompt(
            "Classes separated by commas (empty=all): ",
            default="",
            completer=WordCompleter(classes, ignore_case=True),
            complete_while_typing=True,
        ).strip()
        or None
    )
    args.output_name = prompt_text("Output dataset name (empty=auto)", default=(args.output_name or "")).strip() or None
    print("[INFO] Block: flip")
    args.enable_flip = prompt_yes_no("Turn on flip?", default=bool(args.enable_flip))
    if args.enable_flip:
        args.flip = prompt_choice(
            "Flip mode (--flip)",
            ["horizontal", "vertical", "both", "none"],
            default=args.flip,
        )
        args.flip_prob = float(
            prompt_text("Probability of flip [0..1] (--flip-prob)", default=str(getattr(args, "flip_prob", 0.5))).strip()
            or str(getattr(args, "flip_prob", 0.5))
        )
    print("[INFO] Block: photometric/conveyor")
    args.enable_photometric = prompt_yes_no("Enable brightness/contrast?", default=bool(args.enable_photometric))
    args.enable_conveyor = prompt_yes_no("Enable conveyor noise/blur/shift/rotate?", default=bool(args.enable_conveyor))
    args.enable_center_rotate = prompt_yes_no("Enable frame rotation around the center?", default=bool(args.enable_center_rotate))
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
    args.enable_bbox_copy = prompt_yes_no("Enable bbox_copy?", default=bool(args.enable_bbox_copy))
    if args.enable_center_rotate or args.enable_bbox_copy:
        print("[INFO] Block: ROI source (general)")
        roi_mode_default = "detector" if str(getattr(args, "placement_mode", "detector")) == "detector" else (
            "bbox" if str(getattr(args, "placement_mode", "detector")) == "bbox" else "none"
        )
        roi_mode = prompt_choice(
            "ROI mode to rotate/bbox_copy (--placement-mode)",
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
    # The soft-balance and diversity options are only relevant to rotate/bbox_copy.
    if args.enable_center_rotate or args.enable_bbox_copy:
        print("[INFO] Block: Balancing/Variety")
        args.imbalance_mode = prompt_choice(
            "Class balancing (--imbalance-mode)",
            ["off", "soft"],
            default=str(getattr(args, "imbalance_mode", "soft")),
        )
        args.imbalance_strength = float(
            prompt_text("Balancing strength (>=0) (--imbalance-strength)", default=str(getattr(args, "imbalance_strength", 1.0))).strip()
            or str(getattr(args, "imbalance_strength", 1.0))
        )
        args.min_diversity_iou = float(
            prompt_text("IoU duplicate threshold [0..1] (--min-diversity-iou)", default=str(getattr(args, "min_diversity_iou", 0.97))).strip()
            or str(getattr(args, "min_diversity_iou", 0.97))
        )
    if args.enable_center_rotate:
        args.min_angle_delta = float(
            prompt_text("Min. angle difference (degrees) (--min-angle-delta)", default=str(getattr(args, "min_angle_delta", 1.0))).strip()
            or str(getattr(args, "min_angle_delta", 1.0))
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
    args.splits = prompt_text("Splits separated by commas (train,val,test)", default=args.splits).strip() or args.splits
    args.dry_run = prompt_yes_no("Do dry-run (--dry-run)?", default=bool(args.dry_run))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_augment_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)
    if args.dataset is None and not interactive_allowed:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    interactive_used = False
    root = resolve_workspace_root(args.workspace)
    layout = WorkspaceLayout(root)
    atexit.register(lambda wr=root: best_effort_prune_workspace_runs_detect(wr))
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] datasets_info.json was not found or is empty.")
        return

    if args.dataset is None and not argv:
        # just in case, but our subcommand is called with arguments from the cli
        pass
    if args.dataset is None and interactive_allowed and sys.stdin.isatty():
        all_classes = sorted({k for v in catalog.values() if isinstance(v, dict) for k in (v.get("classes") or {}).keys()})
        _interactive_fill(args, sorted(catalog.keys()), all_classes, layout.root)
        interactive_used = True

    if not args.dataset:
        print("[ERROR] Incomplete arguments: specify --dataset.")
        return
    if args.dataset not in catalog:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        return
    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("augment", parser, args)
        print_replay_command("before launch", replay_cmd)
    if bool(getattr(args, "placement_roi", False)):
        args.placement_mode = "bbox"

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
    if not args.dry_run:
        splits_present = sorted({str(it["split"]) for it in items}) or ["train", "val", "test"]
        for split in splits_present:
            os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, split, "labels"), exist_ok=True)
    donors = _build_donor_pool(items, args) if args.enable_bbox_copy else []
    detector_roi_cache: dict[str, tuple[int, int, int, int] | None] = {}
    need_detector_for_rotate = bool(args.enable_center_rotate) and str(
        getattr(args, "center_rotate_anchor", "detector")
    ) == "detector"
    need_detector_for_copy = bool(args.enable_bbox_copy) and str(getattr(args, "placement_mode", "detector")) == "detector"
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
            dst_img = os.path.join(out_dir, split, "images", f"{stem}{ext}")
            dst_lbl = os.path.join(out_dir, split, "labels", f"{stem}.txt")
            shutil.copy2(img_src, dst_img)
            if os.path.isfile(lbl_src):
                shutil.copy2(lbl_src, dst_lbl)
            else:
                Path(dst_lbl).write_text("", encoding="utf-8")
        copied += 1
        if split not in split_filter:
            continue

        image_weight = _image_soft_weight(classes_in_image, class_freq, alpha)
        seen_labels: list[list[tuple[int, float, float, float, float]]] = []

        def _budget_ok(delta_total: int) -> bool:
            if split_norm != "train":
                return True
            if args.dry_run:
                return True
            return _aug_extra_budget_allow(train_extra_bbox_used, delta_total, extra_budget)

        def _consume_extra(delta_total: int) -> None:
            nonlocal train_extra_bbox_used
            if split_norm != "train" or args.dry_run:
                return
            train_extra_bbox_used += delta_total

        crc_img = zlib.crc32(img_src.encode("utf-8"))
        flip_rng = random.Random(int(args.seed) + crc_img)
        flip_p = _effective_flip_prob_geo(args, image_weight)

        if args.enable_flip and flip_rng.random() <= flip_p and _budget_ok(bi):
            aug_stem = _aug_stem(stem, args, 1, "f")
            if not args.dry_run:
                out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
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
                )
                seen_labels.append(new_labels)
                if split_norm == "train":
                    _consume_extra(count_yolo_bbox_lines(out_lbl))
                augmented += 1
            else:
                augmented += 1

        photo_rng = random.Random(int(args.seed) + 901 + crc_img)
        if (args.enable_photometric or args.enable_conveyor) and _geo_photo_trigger(args, image_weight, photo_rng):
            geom_mode = "c" if args.enable_conveyor else "b"
            aug_stem = _aug_stem(stem, args, 1, geom_mode)
            if _budget_ok(bi):
                if not args.dry_run:
                    out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                    out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
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
                            _consume_extra(count_yolo_bbox_lines(out_lbl))
                        augmented += 1
                else:
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
                    out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                    out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
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
                    if not _budget_ok(rot_tot):
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
                        _consume_extra(rot_tot)
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
                    out_img = os.path.join(out_dir, split, "images", f"{aug_stem}{ext}")
                    out_lbl = os.path.join(out_dir, split, "labels", f"{aug_stem}.txt")
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
                        delta_bb = len(new_labels)
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
                        if not _budget_ok(delta_bb):
                            try:
                                os.remove(out_img)
                                os.remove(out_lbl)
                            except OSError:
                                pass
                            continue
                        seen_labels.append(new_labels)
                        if split_norm == "train":
                            _consume_extra(delta_bb)
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
    _write_data_yaml(out_dir, all_names)
    out_hash = calculate_dataset_hash(out_dir)
    _update_datasets_sidecar(layout, out_name, class_map if isinstance(class_map, dict) else {}, out_dir, out_hash)
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
                "enable_photometric": bool(args.enable_photometric),
                "enable_conveyor": bool(args.enable_conveyor),
                "enable_center_rotate": bool(args.enable_center_rotate),
                "center_rotate_deg": float(getattr(args, "center_rotate_deg", 5.0)),
                "rotate_copies": int(getattr(args, "rotate_copies", 1)),
                "center_rotate_anchor": str(getattr(args, "center_rotate_anchor", "detector")),
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

