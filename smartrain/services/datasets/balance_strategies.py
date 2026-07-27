from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

def _class_weights(
    bbox_count: dict[str, int],
    *,
    mode: str,
    beta: float,
    min_count: int,
) -> dict[str, float]:
    """Per-class sampling weights.

    Mode ``effective`` uses Class-Balanced effective number of samples
    ``E_n = (1 - beta^n) / (1 - beta)``, ``w ∝ 1/E_n`` (Cui et al.,
    arXiv:1901.05555). Other modes: inverse / sqrt-inverse frequency.
    """
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
    """LVIS-style Repeat Factor Sampling (offline pool expansion).

    Category repeat factor ``r_c = max(1, (t / f_c) ** p)`` with default
    ``t=0.001``, ``p=0.5`` (square-root), matching Gupta et al., LVIS,
    arXiv:1908.03195. Image factor is the max over categories in the image.
    """
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


def _irfs_expand_pool(
    pool: list[tuple[str, str, str, list[str]]],
    instance_count: dict[str, int],
    *,
    rfs_thresh: float,
    rfs_power: float,
    max_repeat_per_image: int,
    rng: random.Random,
    selected_classes: set[str] | None = None,
) -> list[tuple[str, str, str, list[str]]]:
    """Instance-aware Repeat Factor Sampling (IRFS-style offline expansion).

    Category frequency ``f_c`` uses **instance** (bbox) share, not image presence.
    Image repeat is the instance-weighted mean of ``r_c`` over classes on the
    frame: ``r_i = sum_c n_{i,c} r_c / sum_c n_{i,c}``. Inspired by
    instance-aware RFS (arXiv:2305.08069) / object-level resampling
    (arXiv:2104.05702); reduces head inflation vs image-level max-RFS when
    rare and frequent classes co-occur.
    """
    total_instances = max(1, sum(max(0, int(n)) for n in instance_count.values()))
    class_repeat: dict[str, float] = {}
    for c, n_raw in instance_count.items():
        f_c = max(1e-12, float(max(0, int(n_raw))) / float(total_instances))
        class_repeat[c] = max(1.0, (float(rfs_thresh) / f_c) ** float(rfs_power))

    out: list[tuple[str, str, str, list[str]]] = []
    for item in pool:
        cls_names = item[3]
        counts: dict[str, int] = defaultdict(int)
        for c in cls_names:
            if selected_classes and c not in selected_classes:
                continue
            counts[c] += 1
        if not counts:
            repeats = 1
        else:
            denom = float(sum(counts.values()))
            r_i = sum(float(counts[c]) * float(class_repeat.get(c, 1.0)) for c in counts) / denom
            base = int(math.floor(r_i))
            frac = r_i - base
            repeats = base + (1 if rng.random() < frac else 0)
            repeats = min(max(1, repeats), max(1, int(max_repeat_per_image)))
        for _ in range(repeats):
            out.append(item)
    return out


