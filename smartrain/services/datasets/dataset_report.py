"""
Multilingual dataset sample report: diverse crops per class, Markdown + optional PDF/ODT.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
from tqdm import tqdm

from smartrain.cli_entrypoints.support.cli_argparse import CliArgumentParser
from smartrain.cli_entrypoints.support.cli_prompts import prompt_choice, prompt_int, prompt_text
from smartrain.cli_entrypoints.support.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.services.datasets.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.services.datasets.image_label_pairs import collect_label_image_pairs as _collect_label_image_pairs
from smartrain.services.datasets.dataset_stats import (
    SPLITS,
    DatasetStats,
    _classes_from_data_yaml,
    _imbalance_summary,
    _scan_one_dataset,
)
from smartrain.core.runtime.interactive_contract import is_interactive_allowed
from smartrain.core.runtime.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.yolo_labels import YoloBBox, YoloLabel, YoloSegment, read_yolo_labels
from smartrain.services.reporting.document_export import (
    UI,
    _export_odt_builtin_zip,
    _export_odt_odfpy,
    _export_pdf_fpdf2,
    _log,
    _pandoc_executable,
    _pandoc_resource_path,
    _t,
    _try_pandoc_odt,
    _try_pandoc_pdf,
    _xml_attr,
)

DATASETS_REPORTS_SUBDIR = ("analytics", "datasets-reports")

# Letterboxed RGB previews so figures are comparable across classes (fits inside this box).
PREVIEW_CANVAS_W = 200
PREVIEW_CANVAS_H = 200
# If letterbox would upscale more than this, grow the square context crop around the object instead.
REPORT_MAX_LETTERBOX_SCALE = 1.0
# Below this max side (px), outline is drawn outside the true box by one bbox width/height per axis.
TINY_OBJECT_MAX_SIDE_PX = 20



def _tiny_outline_margins_xy(inst: _Instance) -> tuple[float, float]:
    """
    Extra margin left/right (``mx``) and top/bottom (``my``) for drawing only, in image pixels.
    For very small objects (max side ≤ ``TINY_OBJECT_MAX_SIDE_PX``), use one bbox width per
    horizontal side and one bbox height per vertical side so the frame stays visible.
    """
    ax1, ay1 = min(inst.x1, inst.x2), min(inst.y1, inst.y2)
    ax2, ay2 = max(inst.x1, inst.x2), max(inst.y1, inst.y2)
    bw = max(ax2 - ax1, 1e-9)
    bh = max(ay2 - ay1, 1e-9)
    if max(bw, bh) > TINY_OBJECT_MAX_SIDE_PX:
        return 0.0, 0.0
    return bw, bh


def _expand_polygon_radial(
    pts: tuple[tuple[float, float], ...],
    *,
    margin: float,
) -> list[tuple[float, float]]:
    if margin <= 0 or len(pts) < 2:
        return [(float(x), float(y)) for x, y in pts]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    out: list[tuple[float, float]] = []
    for px, py in pts:
        vx, vy = px - cx, py - cy
        L = math.hypot(vx, vy)
        if L < 1e-6:
            out.append((px, py))
            continue
        out.append((px + margin * vx / L, py + margin * vy / L))
    return out


def _markdown_examples_grid(items: list[tuple[Any, str]], lang: str) -> list[str]:
    """
    Two columns via a pipe table and standard ``![alt](url)`` figures.
    Raw ``<img>`` HTML is omitted by Pandoc's LaTeX/PDF writer, so tables keep PDFs working.
    """
    if not items:
        return ["", ""]
    lines: list[str] = ["", "|  |  |", "|:---:|:---:|"]
    for j in range(0, len(items), 2):
        alt_l = f"{_t(lang, 'example_caption')} {j + 1}"
        rel_l = items[j][1]
        cell_l = f"![{alt_l}](../{rel_l})"
        if j + 1 < len(items):
            alt_r = f"{_t(lang, 'example_caption')} {j + 2}"
            rel_r = items[j + 1][1]
            cell_r = f"![{alt_r}](../{rel_r})"
            lines.append(f"| {cell_l} | {cell_r} |")
        else:
            lines.append(f"| {cell_l} |  |")
    lines.append("")
    return lines


def _slug(s: str) -> str:
    raw = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return raw[:80] if raw else "class"


def _dataset_stats_markdown_lines(ds: DatasetStats, lang: str) -> list[str]:
    """Summary aligned with ``smartrain stats`` table row (no duplicate-scan flags)."""
    class_totals = {
        k: sum(v.values()) for k, v in ds.per_class_split_instances.items() if sum(v.values()) > 0
    }
    m = _imbalance_summary(class_totals)
    empty_pct = (100.0 * ds.empty_images / ds.images_total) if ds.images_total else 0.0
    quality_ok = (
        ds.broken_label_lines == 0
        and ds.unknown_class_ids == 0
        and ds.orphan_images == 0
        and ds.orphan_labels == 0
    )
    q_key = "stats_quality_ok" if quality_ok else "stats_quality_warn"
    split_lines: list[str] = []
    for sp in SPLITS:
        ni = int(ds.split_images.get(sp, 0))
        no = int(ds.split_instances.get(sp, 0))
        if ni or no:
            split_lines.append(
                f"  - `{sp}`: {ni} {_t(lang, 'stats_split_images')}, {no} {_t(lang, 'stats_split_objects')}"
            )
    lines: list[str] = [
        f"## {_t(lang, 'stats_heading')}",
        "",
        f"- **{_t(lang, 'stats_classes')}:** {len(class_totals)}",
        f"- **{_t(lang, 'stats_images')}:** {ds.images_total}",
        f"- **{_t(lang, 'stats_labeled')}:** {ds.labeled_images}",
        f"- **{_t(lang, 'stats_empty')}:** {ds.empty_images}",
        f"- **{_t(lang, 'stats_empty_pct')}:** {empty_pct:.2f}%",
        f"- **{_t(lang, 'stats_instances')}:** {ds.instances_total}",
        f"- **{_t(lang, 'stats_imbalance')}:** {m['ratio']:.3f}",
        f"- **{_t(lang, 'stats_gini')}:** {m['gini']:.3f}",
        f"- **{_t(lang, 'stats_quality')}:** {_t(lang, q_key)}",
        "",
        f"### {_t(lang, 'stats_per_split')}",
        "",
    ]
    if split_lines:
        lines.extend(split_lines)
    else:
        lines.append(f"- {_t(lang, 'stats_no_splits')}")
    lines.extend(
        [
            "",
            f"### {_t(lang, 'stats_issues')}",
            "",
            f"- {_t(lang, 'stats_broken_lines')}: **{ds.broken_label_lines}** · "
            f"{_t(lang, 'stats_unknown_ids')}: **{ds.unknown_class_ids}** · "
            f"{_t(lang, 'stats_orphan_images')}: **{ds.orphan_images}** · "
            f"{_t(lang, 'stats_orphan_labels')}: **{ds.orphan_labels}**",
            "",
        ]
    )
    return lines


def _bbox_norm_to_px(lb: YoloBBox, w: int, h: int) -> tuple[float, float, float, float]:
    x1 = (lb.cx - lb.w / 2.0) * w
    y1 = (lb.cy - lb.h / 2.0) * h
    x2 = (lb.cx + lb.w / 2.0) * w
    y2 = (lb.cy + lb.h / 2.0) * h
    return x1, y1, x2, y2


def _segment_bbox_px(lb: YoloSegment, w: int, h: int) -> tuple[float, float, float, float]:
    xs = [p[0] * w for p in lb.points]
    ys = [p[1] * h for p in lb.points]
    return min(xs), min(ys), max(xs), max(ys)


def _segment_points_px(lb: YoloSegment, w: int, h: int) -> list[tuple[float, float]]:
    return [(p[0] * w, p[1] * h) for p in lb.points]


def _padded_roi(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    iw: int,
    ih: int,
    padding_frac: float,
) -> tuple[int, int, int, int]:
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad = padding_frac * max(bw, bh)
    nx1 = int(max(0, x1 - pad))
    ny1 = int(max(0, y1 - pad))
    nx2 = int(min(iw, x2 + pad))
    ny2 = int(min(ih, y2 + pad))
    if nx2 <= nx1:
        nx2 = min(iw, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(ih, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _crop_phash(img: Image.Image) -> int:
    """Average hash (64 bits) of grayscale 8x8 — same spirit as dataset_stats."""
    g = img.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    get_flat = getattr(g, "get_flattened_data", None)
    pixels = list(get_flat() if callable(get_flat) else g.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= mean else 0)
    return bits


# Max side before hashing: speeds up huge ROIs (full-image resize to 8x8 is expensive).
_FPHASH_THUMB_MAX = 128


def _raw_crop_phash(
    opened_rgb: Image.Image,
    inst: _Instance,
    padding_frac: float,
    *,
    canvas_w: int,
    canvas_h: int,
    max_letterbox_scale: float,
) -> int:
    """
    Perceptual hash of the same context ROI as the report preview (no overlay).
    ``opened_rgb`` must be RGB mode and already loaded.
    """
    iw, ih = opened_rgb.size
    roi = _report_context_roi(
        inst,
        iw,
        ih,
        padding_frac=padding_frac,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        max_letterbox_scale=max_letterbox_scale,
    )
    tile = opened_rgb.crop(roi)
    if tile.width > _FPHASH_THUMB_MAX or tile.height > _FPHASH_THUMB_MAX:
        tile = tile.copy()
        tile.thumbnail((_FPHASH_THUMB_MAX, _FPHASH_THUMB_MAX), Image.Resampling.BILINEAR)
    return _crop_phash(tile)


def _fingerprint_hashes_by_class(
    by_class: dict[int, list[_Instance]],
    *,
    padding_frac: float,
    pbar: tqdm | None,
    canvas_w: int,
    canvas_h: int,
    max_letterbox_scale: float,
) -> dict[int, tuple[list[int], list[int]]]:
    """
    For each class id, returns (hashes, valid_instance_indices) for diversity selection.
    Groups work by image path so each file is opened once.
    """
    by_image: dict[str, list[tuple[int, int, _Instance]]] = defaultdict(list)
    for cid, instances in by_class.items():
        for idx, inst in enumerate(instances):
            by_image[inst.image_path].append((cid, idx, inst))

    acc: dict[int, list[tuple[int, int]]] = defaultdict(list)  # cid -> (instance_idx, hash)

    for img_path in sorted(by_image.keys()):
        group = by_image[img_path]
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                for cid, idx, inst in group:
                    try:
                        h = _raw_crop_phash(
                            im,
                            inst,
                            padding_frac,
                            canvas_w=canvas_w,
                            canvas_h=canvas_h,
                            max_letterbox_scale=max_letterbox_scale,
                        )
                        acc[cid].append((idx, h))
                    except Exception:
                        continue
        except Exception:
            pass
        finally:
            if pbar is not None:
                pbar.update(1)

    out: dict[int, tuple[list[int], list[int]]] = {}
    for cid, instances in by_class.items():
        pairs = sorted(acc.get(cid, []), key=lambda t: t[0])
        valid_idx = [p[0] for p in pairs]
        hashes = [p[1] for p in pairs]
        out[cid] = (hashes, valid_idx)
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _farthest_first_indices(hashes: list[int], k: int, rng: random.Random) -> list[int]:
    n = len(hashes)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    first = rng.randrange(n)
    selected = [first]
    while len(selected) < k:
        unused = [i for i in range(n) if i not in selected]
        best_i = max(
            unused,
            key=lambda idx: (min(_hamming(hashes[idx], hashes[j]) for j in selected), rng.random()),
        )
        selected.append(best_i)
    return selected


def _class_color(name: str) -> tuple[int, int, int]:
    h = hash(name) & 0xFFFFFF
    r = 80 + (h >> 16) % 160
    g = 80 + ((h >> 8) & 0xFF) % 160
    b = 80 + (h & 0xFF) % 160
    return r, g, b


@dataclass(frozen=True)
class _Instance:
    cls_id: int
    image_path: str
    label_path: str
    kind: str  # "bbox" | "segment"
    # full-image pixel geometry
    x1: float
    y1: float
    x2: float
    y2: float
    segment_points: tuple[tuple[float, float], ...] | None


def _report_context_roi(
    inst: _Instance,
    iw: int,
    ih: int,
    *,
    padding_frac: float,
    canvas_w: int,
    canvas_h: int,
    max_letterbox_scale: float,
) -> tuple[int, int, int, int]:
    """
    Square crop centered on the instance bbox, at least as large as the padded label ROI,
    and large enough that fitting to ``canvas_w``×``canvas_h`` does not upscale beyond
    ``max_letterbox_scale`` (when the image border allows it).
    """
    base = _padded_roi(inst.x1, inst.y1, inst.x2, inst.y2, iw=iw, ih=ih, padding_frac=padding_frac)
    m = max(float(max_letterbox_scale), 1e-6)
    min_side = float(min(canvas_w, canvas_h)) / m

    cx = 0.5 * (inst.x1 + inst.x2)
    cy = 0.5 * (inst.y1 + inst.y2)
    half_x = max(abs(float(base[0]) - cx), abs(float(base[2]) - cx))
    half_y = max(abs(float(base[1]) - cy), abs(float(base[3]) - cy))
    side_need = 2.0 * max(half_x, half_y, 1e-6)
    s = max(side_need, min_side)

    x1 = int(math.floor(cx - s * 0.5))
    y1 = int(math.floor(cy - s * 0.5))
    x2 = int(math.ceil(cx + s * 0.5))
    y2 = int(math.ceil(cy + s * 0.5))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, max(x2, x1 + 1))
    y2 = min(ih, max(y2, y1 + 1))
    return x1, y1, x2, y2


def _labels_to_instances(image_path: str, label_path: str, labels: Iterable[YoloLabel]) -> list[_Instance]:
    im = Image.open(image_path)
    iw, ih = im.size
    im.close()
    out: list[_Instance] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            x1, y1, x2, y2 = _bbox_norm_to_px(lb, iw, ih)
            out.append(
                _Instance(
                    cls_id=int(lb.cls_id),
                    image_path=image_path,
                    label_path=label_path,
                    kind="bbox",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    segment_points=None,
                )
            )
        elif isinstance(lb, YoloSegment):
            x1, y1, x2, y2 = _segment_bbox_px(lb, iw, ih)
            pts = tuple(_segment_points_px(lb, iw, ih))
            out.append(
                _Instance(
                    cls_id=int(lb.cls_id),
                    image_path=image_path,
                    label_path=label_path,
                    kind="segment",
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    segment_points=pts,
                )
            )
    return out


def _build_report_instance_image(
    inst: _Instance,
    *,
    class_name: str,
    padding_frac: float,
    canvas_w: int,
    canvas_h: int,
    max_letterbox_scale: float,
    bg: tuple[int, int, int] = (246, 246, 246),
) -> Image.Image:
    """
    Context ROI (expanded when the object is small so letterbox does not upscale too much)
    → letterbox to ``canvas_w``×``canvas_h`` → draw bbox/segment on the final pixels.
    """
    im = Image.open(inst.image_path).convert("RGB")
    iw, ih = im.size
    roi = _report_context_roi(
        inst,
        iw,
        ih,
        padding_frac=padding_frac,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        max_letterbox_scale=max_letterbox_scale,
    )
    crop = im.crop(roi)
    w, h = crop.size
    ox1, oy1 = float(roi[0]), float(roi[1])

    if canvas_w < 8 or canvas_h < 8:
        return crop.convert("RGB")
    if w <= 0 or h <= 0:
        return Image.new("RGB", (canvas_w, canvas_h), bg)

    scale = min(canvas_w / w, canvas_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resample = (
        Image.Resampling.LANCZOS if max(w, h) > max(canvas_w, canvas_h) else Image.Resampling.BICUBIC
    )
    resized = crop.resize((nw, nh), resample)
    out = Image.new("RGB", (canvas_w, canvas_h), bg)
    ox = (canvas_w - nw) // 2
    oy = (canvas_h - nh) // 2
    out.paste(resized, (ox, oy))

    def map_xy(lx: float, ly: float) -> tuple[float, float]:
        return ox + lx * nw / w, oy + ly * nh / h

    mx, my = _tiny_outline_margins_xy(inst)
    ax1, ay1 = min(inst.x1, inst.x2), min(inst.y1, inst.y2)
    ax2, ay2 = max(inst.x1, inst.x2), max(inst.y1, inst.y2)
    dr_x1, dr_y1 = ax1 - mx, ay1 - my
    dr_x2, dr_y2 = ax2 + mx, ay2 + my

    col = _class_color(class_name)
    line_w = max(2, int(min(canvas_w, canvas_h) * 0.012))
    draw = ImageDraw.Draw(out)
    if inst.kind == "bbox" and inst.segment_points is None:
        px1, py1 = map_xy(dr_x1 - ox1, dr_y1 - oy1)
        px2, py2 = map_xy(dr_x2 - ox1, dr_y2 - oy1)
        x1, y1, x2, y2 = (
            int(round(min(px1, px2))),
            int(round(min(py1, py2))),
            int(round(max(px1, px2))),
            int(round(max(py1, py2))),
        )
        draw.rectangle([x1, y1, x2, y2], outline=col, width=line_w)
    elif inst.segment_points:
        m_poly = max(mx, my) if (mx > 0 or my > 0) else 0.0
        pts_draw = (
            tuple(_expand_polygon_radial(inst.segment_points, margin=m_poly))
            if m_poly > 0
            else inst.segment_points
        )
        poly = [map_xy(x - ox1, y - oy1) for x, y in pts_draw]
        if len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=col, width=line_w)
        bx1, by1 = map_xy(dr_x1 - ox1, dr_y1 - oy1)
        bx2, by2 = map_xy(dr_x2 - ox1, dr_y2 - oy1)
        rx1, ry1, rx2, ry2 = (
            int(round(min(bx1, bx2))),
            int(round(min(by1, by2))),
            int(round(max(bx1, bx2))),
            int(round(max(by1, by2))),
        )
        draw.rectangle([rx1, ry1, rx2, ry2], outline=col, width=1)
    return out


def _load_catalog(layout: WorkspaceLayout) -> dict[str, Any]:
    p = layout.work_datasets_info_path()
    if not os.path.isfile(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _parse_languages(s: str) -> list[str]:
    parts = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_report_dataset_arg_parser() -> argparse.ArgumentParser:
    p = CliArgumentParser(description="Build a multilingual visual dataset sample report (Markdown + assets + optional PDF/ODT).")
    p.add_argument("--workspace", type=str, default=None, help=f"Workspace root (aka {WORKSPACE_ENV_VAR})")
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from datasets_info.json")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Report directory (default: <workspace>/{'/'.join(DATASETS_REPORTS_SUBDIR)}/<dataset>_<timestamp>/)",
    )
    p.add_argument(
        "--examples-per-class",
        "-n",
        type=int,
        default=4,
        help="Number of diverse examples per class (1–8)",
    )
    p.add_argument(
        "--languages",
        type=str,
        default="en,ru",
        help="Comma-separated language codes for UI strings (default: en,ru)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for diverse sampling ties")
    p.add_argument(
        "--crop-padding",
        type=float,
        default=0.12,
        help="Extra padding around each box as a fraction of max(box width, height)",
    )
    p.add_argument(
        "--report-max-letterbox-scale",
        type=float,
        default=REPORT_MAX_LETTERBOX_SCALE,
        help=(
            "When fitting crops to the preview canvas, do not upscale raw pixels beyond this factor; "
            "if needed, a larger square context around the object is taken instead (default: 1.0 = no upscale)."
        ),
    )
    p.add_argument("--no-pdf", action="store_true", help="Do not build PDF")
    p.add_argument("--no-odt", action="store_true", help="Do not build ODT")
    p.epilog = (
        "PDF/ODT via pandoc requires the export extra: "
        "smartrain deps install or pip install -e \".[export]\". "
        "Bundled pandoc comes from pypandoc-binary. "
        "Env PANDOC=/full/path/to/pandoc overrides discovery."
    )
    return p


def _default_output_dir(layout: WorkspaceLayout, dataset_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(layout.root, *DATASETS_REPORTS_SUBDIR)
    return os.path.join(base, f"{dataset_name}_{ts}")


def _collect_all_instances(
    dataset_root: str,
    structure: str,
    entry: dict[str, Any],
    *,
    dataset_name: str,
    tmp_root: str,
) -> tuple[list[tuple[str, str]], dict[int, str]]:
    if structure == "cvat11":
        _log("[INFO] CVAT 1.1 layout: generating temporary YOLO labels (may take several minutes on large tasks)…")
    else:
        _log("[INFO] Resolving image/label directory buckets…")
    buckets = iter_image_label_buckets(
        dataset_root,
        structure,
        entry,
        dataset_name=dataset_name,
        temp_root=tmp_root,
    )
    _log(f"[INFO] Found {len(buckets)} bucket(s); collecting image–label pairs…")
    pairs: list[tuple[str, str]] = []
    for img_dir, lbl_dir in tqdm(buckets, desc="Buckets", unit="bucket", disable=len(buckets) <= 1):
        chunk = _collect_label_image_pairs(img_dir, lbl_dir)
        pairs.extend(chunk)
    _log(f"[INFO] Collected {len(pairs)} image–label pair(s).")
    classes_by_id = _classes_from_data_yaml(dataset_root)
    return pairs, classes_by_id


def _gather_instances(pairs: list[tuple[str, str]], classes_by_id: dict[int, str]) -> dict[int, list[_Instance]]:
    by_class: dict[int, list[_Instance]] = {}
    _log("[INFO] Reading label files and building instance list…")
    for img_path, lbl_path in tqdm(pairs, desc="Labels", unit="pair", disable=len(pairs) < 200):
        labels = read_yolo_labels(lbl_path)
        if not labels:
            continue
        for inst in _labels_to_instances(img_path, lbl_path, labels):
            by_class.setdefault(inst.cls_id, []).append(inst)
    inst_total = sum(len(v) for v in by_class.values())
    _log(f"[INFO] Instances by class: {len(by_class)} class(es), {inst_total} object(s) total.")
    return by_class




def _run_interactive(args: argparse.Namespace, dataset_names: list[str]) -> None:
    print("[INFO] Interactive dataset report (Enter = defaults).")
    args.dataset = prompt_choice("Dataset", dataset_names, default=args.dataset or dataset_names[0])
    layout = WorkspaceLayout(resolve_workspace_root(args.workspace))
    default_out = _default_output_dir(layout, args.dataset)
    raw = prompt_text("Output directory (--output-dir)", default=default_out).strip()
    args.output_dir = raw or default_out
    args.examples_per_class = prompt_int("Examples per class (1-8)", default=int(args.examples_per_class or 4))
    args.examples_per_class = max(1, min(8, int(args.examples_per_class)))
    langs_raw = prompt_text("Languages (comma, e.g. en,ru)", default=str(args.languages or "en,ru")).strip()
    if langs_raw:
        args.languages = langs_raw


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_report_dataset_arg_parser()
    args = parser.parse_args(argv)
    interactive_allowed = is_interactive_allowed(argv)

    try:
        root = resolve_workspace_root(args.workspace)
    except ValueError:
        if interactive_allowed and sys.stdin.isatty():
            raw = prompt_text("Workspace path", default=os.getcwd()).strip()
            if not raw:
                print("[ERROR] Workspace not set.")
                return
            args.workspace = raw
            root = resolve_workspace_root(args.workspace)
        else:
            print("[ERROR] Workspace not set: use --workspace or set SMART_TRAIN_WORKSPACE.")
            return

    layout = WorkspaceLayout(root)
    catalog = _load_catalog(layout)
    if not catalog:
        print("[ERROR] datasets_info.json missing or empty.")
        return

    names_sorted = sorted(str(k) for k in catalog if isinstance(catalog.get(k), dict))
    interactive_used = False
    if not args.dataset:
        if not interactive_allowed:
            print("[ERROR] Incomplete arguments: specify --dataset.")
            return
        if not sys.stdin.isatty():
            print("[ERROR] Interactive mode requires a terminal (TTY).")
            return
        _run_interactive(args, names_sorted)
        interactive_used = True

    if not args.dataset or args.dataset not in catalog:
        print(f"[ERROR] Unknown or missing dataset: {args.dataset!r}")
        return

    n = int(args.examples_per_class)
    if n < 1 or n > 8:
        print("[ERROR] --examples-per-class must be between 1 and 8.")
        return

    langs = _parse_languages(args.languages or "en,ru")
    if not langs:
        langs = ["en", "ru"]

    entry = catalog[args.dataset]
    if not isinstance(entry, dict):
        print("[ERROR] Invalid catalog entry.")
        return
    structure = str(entry.get("structure") or "flat")

    if not args.output_dir:
        args.output_dir = _default_output_dir(layout, args.dataset)
    out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)

    replay_cmd = None
    if interactive_used:
        replay_cmd = build_non_interactive_command("dataset report", parser, args)
        print_replay_command("before launch", replay_cmd)

    _log("[INFO] --- Generating report (this may take a while) ---")
    _log(f"[INFO] Output directory: {out_dir}")

    tmp_root = os.path.join(layout.root, "tmp", "dataset_report")
    os.makedirs(tmp_root, exist_ok=True)

    _log("[INFO] Resolving dataset root path…")
    src_root = resolve_dataset_root_for_entry(
        args.dataset,
        entry,
        workspace_root=layout.root,
        source_catalog_dir=layout.datasets,
        legacy_source_parent=layout.datasets,
    )
    _log(f"[INFO] Dataset root: {src_root}")
    _log(f"[INFO] Catalog structure: {structure}")

    pairs, classes_by_id = _collect_all_instances(
        src_root,
        structure,
        entry,
        dataset_name=args.dataset,
        tmp_root=tmp_root,
    )
    by_class = _gather_instances(pairs, classes_by_id)
    if not classes_by_id and by_class:
        classes_by_id = {cid: f"id_{cid}" for cid in sorted(by_class.keys())}
    for cid in by_class:
        if cid not in classes_by_id:
            classes_by_id[cid] = f"id_{cid}"
    if not classes_by_id:
        print("[ERROR] No class names found (data.yaml missing or empty names) and no labels scanned.")
        return

    _log("[INFO] Dataset statistics (same scan as `smartrain stats`)…")
    ds_stats = _scan_one_dataset(src_root, args.dataset)

    picks: dict[int, list[tuple[_Instance, str]]] = {cid: [] for cid in sorted(classes_by_id.keys())}
    padding = float(args.crop_padding)
    max_lb_scale = max(0.25, min(float(getattr(args, "report_max_letterbox_scale", 1.0)), 8.0))

    fp_total = sum(len(v) for v in by_class.values())
    uniq_images = len({inst.image_path for v in by_class.values() for inst in v})
    _log(
        f"[INFO] Selecting up to {n} diverse example(s) per class: "
        f"fingerprinting {fp_total} object(s) over {uniq_images} image file(s) (raw crops, one open per image)…"
    )
    if fp_total == 0:
        _log("[INFO] No instances to fingerprint.")
    pbar_fp: tqdm | None = None
    if uniq_images > 0:
        pbar_fp = tqdm(
            total=uniq_images,
            desc="Fingerprint images",
            unit="img",
            disable=uniq_images < 80,
        )
    try:
        fp_result = _fingerprint_hashes_by_class(
            by_class,
            padding_frac=padding,
            pbar=pbar_fp,
            canvas_w=PREVIEW_CANVAS_W,
            canvas_h=PREVIEW_CANVAS_H,
            max_letterbox_scale=max_lb_scale,
        )
        for cid, instances in by_class.items():
            hashes, valid_idx = fp_result.get(cid, ([], []))
            if not hashes:
                picks[cid] = []
                continue
            sel_local = _farthest_first_indices(hashes, n, rng)
            chosen = [instances[valid_idx[j]] for j in sel_local]
            cname = classes_by_id.get(cid, f"id_{cid}")
            sub = _slug(cname)
            picks[cid] = [
                (inst, os.path.join("assets", sub, f"{k}.png").replace(os.sep, "/")) for k, inst in enumerate(chosen)
            ]
    finally:
        if pbar_fp is not None:
            pbar_fp.close()

    png_total = sum(len(v) for v in picks.values())
    _log(f"[INFO] Writing {png_total} final crop PNG(s)…")
    assets_root = os.path.join(out_dir, "assets")
    os.makedirs(assets_root, exist_ok=True)
    for cid, items in tqdm(list(picks.items()), desc="Export crops", unit="class", disable=png_total < 8):
        cname = classes_by_id.get(cid, f"id_{cid}")
        sub = os.path.join(assets_root, _slug(cname))
        os.makedirs(sub, exist_ok=True)
        new_items: list[tuple[_Instance, str]] = []
        for k, (inst, _rel) in enumerate(items):
            rel = os.path.join("assets", _slug(cname), f"{k}.png").replace(os.sep, "/")
            crop = _build_report_instance_image(
                inst,
                class_name=cname,
                padding_frac=padding,
                canvas_w=PREVIEW_CANVAS_W,
                canvas_h=PREVIEW_CANVAS_H,
                max_letterbox_scale=max_lb_scale,
            )
            crop.save(os.path.join(out_dir, rel.replace("/", os.sep)))
            new_items.append((inst, rel))
        picks[cid] = new_items

    _log(f"[INFO] Writing Markdown for {len(langs)} language(s): {', '.join(langs)}…")
    for lang in langs:
        lang_dir = os.path.join(out_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        lines: list[str] = []
        lines.append(f"# {_t(lang, 'doc_title')}")
        lines.append("")
        lines.append(f"- **{_t(lang, 'dataset')}:** `{args.dataset}`")
        lines.append(f"- **{_t(lang, 'generated')}:** {datetime.now().isoformat(timespec='seconds')}")
        lines.append("")
        lines.extend(_dataset_stats_markdown_lines(ds_stats, lang))
        lines.append(f"## {_t(lang, 'footer_classes')}")
        lines.append("")
        for cid0 in sorted(classes_by_id.keys()):
            lines.append(f"- `{classes_by_id[cid0]}` (id={cid0})")
        lines.append("")

        for cid0 in sorted(classes_by_id.keys()):
            cname0 = classes_by_id[cid0]
            lines.append(f"## {_t(lang, 'class_heading')}: `{cname0}` (id={cid0})")
            lines.append("")
            items = picks.get(cid0) or []
            if not items:
                lines.append(_t(lang, "no_instances"))
                lines.append("")
                continue
            lines.append(f"### {_t(lang, 'examples_heading')}")
            lines.extend(_markdown_examples_grid(items, lang))
        with open(os.path.join(lang_dir, "index.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        _log(f"[INFO] Wrote {lang}/index.md")

    _log(f"[OK] Markdown + assets written to: {out_dir}")

    if not args.no_pdf:
        for lang in langs:
            _log(f"[INFO] Building PDF for [{lang}]…")
            if not _try_pandoc_pdf(out_dir, lang):
                _log(f"[INFO] Pandoc PDF failed for [{lang}]; trying fpdf2 fallback…")
                if not _export_pdf_fpdf2(lang, out_dir, args.dataset, picks, classes_by_id):
                    _log(f"[WARNING] PDF not built for {lang} (check pandoc/weasyprint engines and fpdf2 logs above).")
                else:
                    _log(f"[OK] PDF (fpdf2): report-{lang}.pdf")
            else:
                _log(f"[OK] PDF (pandoc): report-{lang}.pdf")

    if not args.no_odt:
        for lang in langs:
            _log(f"[INFO] Building ODT for [{lang}]…")
            if _try_pandoc_odt(out_dir, lang):
                _log(f"[OK] ODT (pandoc): report-{lang}.odt")
                continue
            _log(f"[INFO] Pandoc ODT failed for [{lang}]; trying odfpy…")
            if _export_odt_odfpy(lang, out_dir, args.dataset, picks, classes_by_id):
                _log(f"[OK] ODT (odfpy): report-{lang}.odt")
                continue
            _log(f"[INFO] odfpy failed; built-in ODT zip writer…")
            if _export_odt_builtin_zip(lang, out_dir, args.dataset, picks, classes_by_id):
                _log(f"[OK] ODT (builtin): report-{lang}.odt")
            else:
                _log(f"[WARNING] ODT not built for {lang}.")

    if replay_cmd:
        print_replay_command("after execution", replay_cmd)
