from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any

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


