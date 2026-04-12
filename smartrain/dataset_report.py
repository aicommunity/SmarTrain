"""
Multilingual dataset sample report: diverse crops per class, Markdown + optional PDF/ODT.
"""
from __future__ import annotations

import argparse
import html
import json
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

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from smartrain.cli_argparse import CliArgumentParser
from smartrain.cli_prompts import prompt_choice, prompt_int, prompt_text
from smartrain.cli_replay import build_non_interactive_command, print_replay_command
from smartrain.dataset_access import iter_image_label_buckets, resolve_dataset_root_for_entry
from smartrain.dataset_former import _collect_label_image_pairs
from smartrain.dataset_stats import _classes_from_data_yaml
from smartrain.interactive_contract import is_interactive_allowed
from smartrain.workspace_paths import WORKSPACE_ENV_VAR, WorkspaceLayout, resolve_workspace_root
from smartrain.yolo_labels import YoloBBox, YoloLabel, YoloSegment, read_yolo_labels

DATASETS_REPORTS_SUBDIR = ("analytics", "datasets-reports")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _xml_attr(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


UI: dict[str, dict[str, str]] = {
    "en": {
        "doc_title": "Dataset sample report",
        "dataset": "Dataset",
        "generated": "Generated",
        "class_heading": "Class",
        "examples_heading": "Examples",
        "example_caption": "Example",
        "no_instances": "No labeled instances found for this class.",
        "footer_classes": "Classes in data.yaml",
    },
    "ru": {
        "doc_title": "Отчёт по примерам датасета",
        "dataset": "Датасет",
        "generated": "Сформировано",
        "class_heading": "Класс",
        "examples_heading": "Примеры",
        "example_caption": "Пример",
        "no_instances": "Для этого класса нет размеченных экземпляров.",
        "footer_classes": "Классы в data.yaml",
    },
}


def _t(lang: str, key: str) -> str:
    block = UI.get(lang) or UI["en"]
    return block.get(key) or UI["en"][key]


def _slug(s: str) -> str:
    raw = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE).strip("_")
    return raw[:80] if raw else "class"


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
    pixels = list(g.getdata())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= mean else 0)
    return bits


# Max side before hashing: speeds up huge ROIs (full-image resize to 8x8 is expensive).
_FPHASH_THUMB_MAX = 128


def _raw_crop_phash(opened_rgb: Image.Image, inst: _Instance, padding_frac: float) -> int:
    """
    Perceptual hash of the padded ROI without drawing boxes/text (fast path for diversity).
    ``opened_rgb`` must be RGB mode and already loaded.
    """
    iw, ih = opened_rgb.size
    roi = _padded_roi(inst.x1, inst.y1, inst.x2, inst.y2, iw=iw, ih=ih, padding_frac=padding_frac)
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
                        h = _raw_crop_phash(im, inst, padding_frac)
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


def _render_instance_crop(
    inst: _Instance,
    *,
    class_name: str,
    padding_frac: float,
) -> Image.Image:
    im = Image.open(inst.image_path).convert("RGB")
    iw, ih = im.size
    roi = _padded_roi(inst.x1, inst.y1, inst.x2, inst.y2, iw=iw, ih=ih, padding_frac=padding_frac)
    crop = im.crop(roi)
    ox1, oy1 = roi[0], roi[1]
    draw = ImageDraw.Draw(crop)
    col = _class_color(class_name)
    if inst.kind == "bbox" and inst.segment_points is None:
        rx1 = inst.x1 - ox1
        ry1 = inst.y1 - oy1
        rx2 = inst.x2 - ox1
        ry2 = inst.y2 - oy1
        draw.rectangle([rx1, ry1, rx2, ry2], outline=col, width=max(2, int(min(crop.size) * 0.005) + 1))
    elif inst.segment_points:
        poly = [(x - ox1, y - oy1) for x, y in inst.segment_points]
        if len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=col, width=max(2, int(min(crop.size) * 0.005) + 1))
        bx1, by1, bx2, by2 = inst.x1 - ox1, inst.y1 - oy1, inst.x2 - ox1, inst.y2 - oy1
        draw.rectangle([bx1, by1, bx2, by2], outline=col, width=1)

    label = class_name
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, int(min(crop.size) * 0.04)))
    except OSError:
        font = ImageFont.load_default()
    if hasattr(draw, "textbbox"):
        tw, th = draw.textbbox((0, 0), label, font=font)[2:4]
    else:
        tw, th = draw.textsize(label, font=font)
    pad = 4
    tx, ty = 4, 4
    draw.rectangle([tx, ty, tx + tw + pad * 2, ty + th + pad * 2], fill=(0, 0, 0))
    draw.text((tx + pad, ty + pad), label, fill=(255, 255, 255), font=font)
    return crop


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
    p.add_argument("--no-pdf", action="store_true", help="Do not build PDF")
    p.add_argument("--no-odt", action="store_true", help="Do not build ODT")
    p.epilog = (
        "Default install includes bundled pandoc (pypandoc-binary) and WeasyPrint for PDF; "
        "system PDF engines (typst, wkhtmltopdf, TeX) are used when on PATH. "
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


def _pandoc_executable() -> str | None:
    """PANDOC env, then PATH, then bundled pandoc from optional ``pypandoc-binary``."""
    raw = (os.environ.get("PANDOC") or "").strip()
    if raw:
        if os.path.isfile(raw):
            return raw
        w = shutil.which(raw)
        if w:
            return w
    w = shutil.which("pandoc")
    if w:
        return w
    try:
        import pypandoc

        p = pypandoc.get_pandoc_path()
        if p and os.path.isfile(p):
            _log(f"[INFO] Using bundled pandoc: {p}")
            return p
    except Exception as e:
        _log(f"[INFO] pypandoc/bundled pandoc unavailable ({e}); reinstall smartrain or set PANDOC.")
    return None


def _pandoc_resource_path(out_dir: str, lang: str) -> str:
    """Search dirs for images etc. (pandoc resolves relative URIs from the .md location)."""
    parts = [
        out_dir,
        os.path.join(out_dir, lang),
        os.path.join(out_dir, "assets"),
    ]
    return os.pathsep.join(parts)


def _pandoc_pdf_engine_variants() -> list[list[str]]:
    """
    Try engines that avoid a full TeX stack first (typst, weasyprint, wkhtmltopdf).
    Pandoc's default is often pdflatex — often missing on minimal hosts.
    """
    variants: list[list[str]] = []
    if shutil.which("typst"):
        variants.append(["--pdf-engine=typst"])
    if shutil.which("weasyprint"):
        variants.append(["--pdf-engine=weasyprint"])
    if shutil.which("wkhtmltopdf"):
        variants.append(["--pdf-engine=wkhtmltopdf"])
    variants.append([])  # pandoc default (often pdflatex)
    if shutil.which("xelatex"):
        variants.append(["--pdf-engine=xelatex"])
    if shutil.which("lualatex"):
        variants.append(["--pdf-engine=lualatex"])
    if shutil.which("pdflatex"):
        variants.append(["--pdf-engine=pdflatex"])
    # de-dupe while preserving order
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for v in variants:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _try_pandoc_pdf(out_dir: str, lang: str) -> bool:
    exe = _pandoc_executable()
    if not exe:
        _log("[WARNING] pandoc: executable not found (set PANDOC=/full/path if it is outside PATH).")
        return False
    rel_md = f"{lang}/index.md"
    rel_pdf = f"report-{lang}.pdf"
    rp = _pandoc_resource_path(out_dir, lang)
    last_err = ""
    for extra in _pandoc_pdf_engine_variants():
        cmd = [exe, rel_md, "-o", rel_pdf, "--resource-path", rp, *extra]
        try:
            r = subprocess.run(
                cmd,
                cwd=out_dir,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if r.returncode == 0:
                if extra:
                    _log(f"[INFO] pandoc PDF for [{lang}] using {' '.join(extra)}")
                return True
            tail = (r.stderr or r.stdout or "").strip()
            last_err = tail[-4000:] if tail else f"exit {r.returncode}"
            _log(f"[INFO] pandoc PDF attempt failed ({' '.join(extra) or 'default engine'}): {last_err[:800]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            last_err = str(e)
            _log(f"[INFO] pandoc PDF attempt error ({extra}): {e}")
    _log(f"[WARNING] pandoc could not build PDF for [{lang}] after all engines. Last error (truncated):\n{last_err[:2000]}")
    return False


def _try_pandoc_odt(out_dir: str, lang: str) -> bool:
    exe = _pandoc_executable()
    if not exe:
        _log("[WARNING] pandoc: executable not found (set PANDOC=/full/path if it is outside PATH).")
        return False
    rel_md = f"{lang}/index.md"
    rel_odt = f"report-{lang}.odt"
    rp = _pandoc_resource_path(out_dir, lang)
    try:
        r = subprocess.run(
            [exe, rel_md, "-o", rel_odt, "--resource-path", rp],
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode == 0:
            return True
        err = (r.stderr or r.stdout or "").strip()
        _log(f"[WARNING] pandoc ODT for [{lang}] failed (exit {r.returncode}): {err[:2000]}")
    except (OSError, subprocess.TimeoutExpired) as e:
        _log(f"[WARNING] pandoc ODT for [{lang}]: {e}")
    return False


def _fpdf_dejavu_paths() -> tuple[str | None, str | None]:
    """(regular_ttf, bold_ttf). Bold may equal regular if no Bold file found."""
    reg: str | None = None
    try:
        import fpdf

        d = Path(fpdf.__file__).resolve().parent / "font"
        r = d / "DejaVuSans.ttf"
        b = d / "DejaVuSans-Bold.ttf"
        if r.is_file():
            reg = str(r)
            if b.is_file():
                return reg, str(b)
    except Exception:
        pass
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    ):
        if os.path.isfile(p):
            reg = p
            break
    if not reg:
        return None, None
    bold_cand = os.path.join(os.path.dirname(reg), "DejaVuSans-Bold.ttf")
    if os.path.isfile(bold_cand):
        return reg, bold_cand
    return reg, reg


def _export_pdf_fpdf2(lang: str, out_dir: str, dataset_name: str, picks: dict[int, list[tuple[_Instance, str]]], classes_by_id: dict[int, str]) -> bool:
    try:
        from fpdf import FPDF
    except ImportError:
        return False

    font_path, bold_path = _fpdf_dejavu_paths()
    if not font_path:
        _log("[WARNING] fpdf2 PDF: no DejaVuSans.ttf found; Cyrillic/non-ASCII text would break. Install fonts-dejavu or pip fpdf2 (bundled font).")
        return False

    pdf_path = os.path.join(out_dir, f"report-{lang}.pdf")
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        try:
            pdf.add_font("ReportFont", "", font_path)
            pdf.add_font("ReportFont", "B", bold_path or font_path)
        except Exception as e:
            _log(f"[WARNING] fpdf2: add_font failed: {e}")
            return False
        font = "ReportFont"
        pdf.set_font(font, size=14)
        pdf.multi_cell(0, 8, f"{_t(lang, 'doc_title')}: {dataset_name}")
        pdf.ln(4)

        for cid in sorted(classes_by_id.keys()):
            cname = classes_by_id[cid]
            pdf.set_font(font, style="B", size=12)
            pdf.multi_cell(0, 7, f"{_t(lang, 'class_heading')}: {cname}")
            pdf.set_font(font, size=10)
            items = picks.get(cid) or []
            if not items:
                pdf.multi_cell(0, 6, _t(lang, "no_instances"))
                pdf.ln(2)
                continue
            for _inst, rel in items:
                img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
                if os.path.isfile(img_abs):
                    try:
                        pdf.image(img_abs, w=min(160, pdf.w - 24))
                    except Exception as ie:
                        _log(f"[WARNING] fpdf2: skip image {img_abs!r}: {ie}")
                    pdf.ln(4)
        pdf.output(pdf_path)
        return True
    except Exception as e:
        _log(f"[WARNING] fpdf2 PDF failed for [{lang}]: {e}")
        try:
            if os.path.isfile(pdf_path):
                os.remove(pdf_path)
        except OSError:
            pass
        return False


def _export_odt_builtin_zip(
    lang: str,
    out_dir: str,
    dataset_name: str,
    picks: dict[int, list[tuple[_Instance, str]]],
    classes_by_id: dict[int, str],
) -> bool:
    """
    Minimal valid ODT (OpenDocument Text) via zip + XML — no odfpy required.
    """
    odt_path = os.path.join(out_dir, f"report-{lang}.odt")
    title = html.escape(f"{_t(lang, 'doc_title')}: {dataset_name}")
    body_parts: list[str] = [f'<text:p>{title}</text:p>', "<text:p/>"]

    manifest_entries: list[tuple[str, str]] = [
        ("/", "application/vnd.oasis.opendocument.text"),
        ("content.xml", "text/xml"),
        ("styles.xml", "text/xml"),
        ("meta.xml", "text/xml"),
    ]
    pic_index = 0
    picture_internal: list[str] = []

    for cid in sorted(classes_by_id.keys()):
        cname = classes_by_id[cid]
        body_parts.append(f'<text:p>{html.escape(_t(lang, "class_heading"))}: {html.escape(cname)} (id={cid})</text:p>')
        items = picks.get(cid) or []
        if not items:
            body_parts.append(f'<text:p>{html.escape(_t(lang, "no_instances"))}</text:p>')
            continue
        for _inst, rel in items:
            img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
            if not os.path.isfile(img_abs):
                continue
            ext = os.path.splitext(img_abs)[1].lower() or ".png"
            if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
                ext = ".png"
            mt = "image/png" if ext == ".png" else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
            internal = f"Pictures/img{pic_index}{ext}"
            picture_internal.append((internal, img_abs, mt))
            manifest_entries.append((internal, mt))
            href_esc = _xml_attr(internal)
            body_parts.append(
                "<text:p>"
                '<draw:frame text:anchor-type="paragraph" draw:z-index="0" '
                'svg:width="15cm" svg:height="11cm">'
                f'<draw:image xlink:href="{href_esc}" xlink:type="simple" xlink:show="embed" xlink:actuate="onLoad"/>'
                "</draw:frame>"
                "</text:p>"
            )
            pic_index += 1
        body_parts.append("<text:p/>")

    manifest_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">',
    ]
    for full_path, media in manifest_entries:
        manifest_lines.append(
            f'<manifest:file-entry manifest:full-path="{_xml_attr(full_path)}" '
            f'manifest:media-type="{_xml_attr(media)}"/>'
        )
    manifest_lines.append("</manifest:manifest>")
    manifest_xml = "\n".join(manifest_lines)

    content_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
        'office:version="1.2">'
        "<office:body><office:text>"
        + "".join(body_parts)
        + "</office:text></office:body></office:document-content>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-styles xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'office:version="1.2"><office:styles/></office:document-styles>'
    )
    meta_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" office:version="1.2">'
        "<office:meta><meta:generator>smartrain dataset_report</meta:generator></office:meta>"
        "</office:document-meta>"
    )

    try:
        with zipfile.ZipFile(odt_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo("mimetype")
            zi.compress_type = zipfile.ZIP_STORED
            zf.writestr(zi, "application/vnd.oasis.opendocument.text")
            zf.writestr("META-INF/manifest.xml", manifest_xml.encode("utf-8"))
            zf.writestr("content.xml", content_xml.encode("utf-8"))
            zf.writestr("styles.xml", styles_xml.encode("utf-8"))
            zf.writestr("meta.xml", meta_xml.encode("utf-8"))
            for internal, abs_path, _mt in picture_internal:
                with open(abs_path, "rb") as f:
                    zf.writestr(internal, f.read())
        return True
    except Exception as e:
        _log(f"[WARNING] built-in ODT failed for [{lang}]: {e}")
        try:
            if os.path.isfile(odt_path):
                os.remove(odt_path)
        except OSError:
            pass
        return False


def _export_odt_odfpy(lang: str, out_dir: str, dataset_name: str, picks: dict[int, list[tuple[_Instance, str]]], classes_by_id: dict[int, str]) -> bool:
    try:
        from odf.draw import Frame, Image
        from odf.opendocument import OpenDocumentText
        from odf.text import H, P
    except ImportError:
        _log("[INFO] odfpy import failed; using built-in ODT zip writer.")
        return False

    doc = OpenDocumentText()
    doc.text.addElement(H(outlinelevel=1, text=f"{_t(lang, 'doc_title')}: {dataset_name}"))
    doc.text.addElement(P(text=""))

    for cid in sorted(classes_by_id.keys()):
        cname = classes_by_id[cid]
        doc.text.addElement(H(outlinelevel=2, text=f"{_t(lang, 'class_heading')}: {cname}"))
        items = picks.get(cid) or []
        if not items:
            doc.text.addElement(P(text=_t(lang, "no_instances")))
            continue
        for _inst, rel in items:
            img_abs = os.path.join(out_dir, rel.replace("/", os.sep))
            if os.path.isfile(img_abs):
                try:
                    href = doc.addPicture(img_abs)
                    frame = Frame(width="16cm", height="12cm", zindex="0")
                    frame.addElement(Image(href=href, type="simple"))
                    doc.text.addElement(frame)
                except Exception:
                    doc.text.addElement(P(text=img_abs))
            doc.text.addElement(P(text=""))

    odt_path = os.path.join(out_dir, f"report-{lang}.odt")
    try:
        doc.save(odt_path)
        return True
    except Exception as e:
        _log(f"[WARNING] odfpy ODT save failed for [{lang}]: {e}")
        return False


def _run_interactive(args: argparse.Namespace, dataset_names: list[str]) -> None:
    print("[INFO] Interactive report dataset (Enter = defaults).")
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
        replay_cmd = build_non_interactive_command("report dataset", parser, args)
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

    picks: dict[int, list[tuple[_Instance, str]]] = {cid: [] for cid in sorted(classes_by_id.keys())}
    padding = float(args.crop_padding)

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
        fp_result = _fingerprint_hashes_by_class(by_class, padding_frac=padding, pbar=pbar_fp)
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
            crop = _render_instance_crop(inst, class_name=cname, padding_frac=padding)
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
            lines.append("")
            for j, (_inst, rel) in enumerate(items):
                alt = f"{_t(lang, 'example_caption')} {j + 1}"
                lines.append(f"![{alt}](../{rel})")
                lines.append("")
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
