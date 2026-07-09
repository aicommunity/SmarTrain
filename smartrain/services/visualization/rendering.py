from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from smartrain.services.datasets.yolo_labels import YoloBBox, YoloLabel, YoloSegment

REF_VIEWPORT_W = 1920
REF_VIEWPORT_H = 1080
BASE_FONT_PX = 21
BASE_LINE_PX = 3
BASE_PAD_PX = 6

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


@dataclass(frozen=True)
class VisDrawMetrics:
    line_width: int
    font_size: int
    pad: int


@dataclass
class _PlacedLabel:
    rect: tuple[int, int, int, int]


def _annotation_scale(width: int, height: int) -> float:
    """Scale annotations up only when the source image is larger than the 1080p viewport."""
    return max(1.0, float(width) / float(REF_VIEWPORT_W), float(height) / float(REF_VIEWPORT_H))


def _vis_draw_metrics(width: int, height: int) -> VisDrawMetrics:
    scale = _annotation_scale(width, height)
    return VisDrawMetrics(
        line_width=max(1, int(round(BASE_LINE_PX * scale))),
        font_size=max(12, int(round(BASE_FONT_PX * scale))),
        pad=max(2, int(round(BASE_PAD_PX * scale))),
    )


@lru_cache(maxsize=32)
def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_color(label: str, palette: dict[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    return palette.get(label) or palette.get("unknown") or (255, 0, 0)


def _fade_color(rgb: tuple[int, int, int], *, strength: float = 0.62) -> tuple[int, int, int]:
    """
    Return noticeably faded variant of class color.
    strength in [0,1]: higher -> more faded (closer to gray/white).
    """
    s = max(0.0, min(1.0, float(strength)))
    r, g, b = rgb
    gray = int(round((r + g + b) / 3))
    mix_to_gray = 0.7 * s
    mix_to_white = 0.55 * s
    rr = int(round(r * (1.0 - mix_to_gray) + gray * mix_to_gray))
    gg = int(round(g * (1.0 - mix_to_gray) + gray * mix_to_gray))
    bb = int(round(b * (1.0 - mix_to_gray) + gray * mix_to_gray))
    rr = int(round(rr * (1.0 - mix_to_white) + 255 * mix_to_white))
    gg = int(round(gg * (1.0 - mix_to_white) + 255 * mix_to_white))
    bb = int(round(bb * (1.0 - mix_to_white) + 255 * mix_to_white))
    return (max(0, min(255, rr)), max(0, min(255, gg)), max(0, min(255, bb)))


def _class_name(class_names: dict[int, str], class_id: int) -> str:
    return str(class_names.get(int(class_id), f"class_{int(class_id)}"))


def _det_class_id(det: dict[str, Any]) -> int:
    raw = det.get("class_id", det.get("class_index", -1))
    try:
        return int(raw)
    except Exception:
        try:
            return int(float(raw))
        except Exception:
            return -1


def _det_class_name(det: dict[str, Any], class_names: dict[int, str], class_id: int) -> str:
    raw_name = det.get("class_name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    cid = int(class_id)
    if cid < 0:
        cid = _det_class_id(det)
    if cid >= 0 and cid in class_names:
        return str(class_names[cid])
    return _class_name(class_names, cid)


def _is_grayscale_image(image: Image.Image) -> bool:
    if image.mode in {"L", "LA", "1", "I", "I;16"}:
        return True
    if image.mode != "RGB":
        return False
    stat = image.getextrema()
    if len(stat) != 3:
        return False
    return stat[0] == stat[1] == stat[2]


def _prepare_canvas(image_path: Path) -> tuple[Image.Image, str | None]:
    with Image.open(image_path) as src:
        original_format = src.format
        image = src.copy()
    if _is_grayscale_image(image):
        image = image.convert("RGB")
    elif image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image, original_format


def _save_format_for_path(output_path: Path, original_format: str | None) -> str | None:
    if original_format:
        return original_format
    ext = output_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "JPEG"
    if ext == ".png":
        return "PNG"
    if ext == ".webp":
        return "WEBP"
    if ext == ".bmp":
        return "BMP"
    return None


def save_rendered_image(image: Image.Image, output_path: Path, *, original_format: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _save_format_for_path(output_path, original_format)
    save_kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = 92
    if fmt in {"JPEG", "BMP"} and image.mode == "RGBA":
        image = image.convert("RGB")
    if fmt:
        image.save(output_path, format=fmt, **save_kwargs)
    else:
        image.save(output_path, **save_kwargs)


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _text_rect(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, pad: int) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        b = draw.textbbox((0, 0), text, font=font)
        return (int(b[2] - b[0]) + 2 * pad, int(b[3] - b[1]) + 2 * pad)
    w, h = draw.textsize(text, font=font)
    return (int(w) + 2 * pad, int(h) + 2 * pad)


def _place_label_rect(
    *,
    image_w: int,
    image_h: int,
    anchor_box: tuple[int, int, int, int],
    label_size: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
    placed: list[_PlacedLabel],
    pad: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = anchor_box
    lw, lh = label_size
    candidates = [
        (x1, y1 - lh - pad),
        (x2 - lw, y1 - lh - pad),
        (x1, y2 + pad),
        (x2 - lw, y2 + pad),
        (x1 + pad, y1 + pad),
    ]
    for cx, cy in candidates:
        rx1 = max(0, min(int(cx), max(0, image_w - lw)))
        ry1 = max(0, min(int(cy), max(0, image_h - lh)))
        rect = (rx1, ry1, rx1 + lw, ry1 + lh)
        if any(_rect_intersects(rect, f) for f in forbidden):
            continue
        if any(_rect_intersects(rect, p.rect) for p in placed):
            continue
        return rect
    # fallback: top-left clamped
    rx1 = max(0, min(int(x1), max(0, image_w - lw)))
    ry1 = max(0, min(int(max(0, y1 - lh - pad)), max(0, image_h - lh)))
    return (rx1, ry1, rx1 + lw, ry1 + lh)


def _draw_text_label(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    metrics: VisDrawMetrics,
    forbidden: list[tuple[int, int, int, int]],
    placed: list[_PlacedLabel],
    anchor_box: tuple[int, int, int, int],
) -> None:
    font = _load_font(metrics.font_size)
    img_w, img_h = image.size
    pad = metrics.pad
    lw, lh = _text_rect(draw, text, font, pad)
    bbox = _place_label_rect(
        image_w=img_w,
        image_h=img_h,
        anchor_box=anchor_box,
        label_size=(lw, lh),
        forbidden=forbidden,
        placed=placed,
        pad=pad,
    )
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [bbox[0], bbox[1], bbox[2], bbox[3]],
        fill=(0, 0, 0, 120),
        outline=(*color, 255),
        width=max(1, metrics.line_width // 2),
    )
    overlay_draw.text((bbox[0] + pad, bbox[1] + pad), text, fill=(*color, 255), font=font)
    image.alpha_composite(overlay)
    placed.append(_PlacedLabel(rect=bbox))


def _draw_gt_label(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    w: int,
    h: int,
    lb: YoloLabel,
    class_names: dict[int, str],
    metrics: VisDrawMetrics,
    label_colors: dict[str, tuple[int, int, int]],
    faded: bool,
    forbidden: list[tuple[int, int, int, int]],
    placed: list[_PlacedLabel],
) -> None:
    cls_id = int(lb.cls_id)
    name = _class_name(class_names, cls_id)
    base_col = _label_color(name, label_colors)
    col = _fade_color(base_col) if faded else base_col
    if isinstance(lb, YoloBBox):
        x1 = (lb.cx - lb.w / 2.0) * w
        y1 = (lb.cy - lb.h / 2.0) * h
        x2 = (lb.cx + lb.w / 2.0) * w
        y2 = (lb.cy + lb.h / 2.0) * h
        draw.rectangle([x1, y1, x2, y2], outline=col, width=metrics.line_width)
        anchor = (int(x1), int(y1), int(x2), int(y2))
        forbidden.append(anchor)
        _draw_text_label(
            image=image,
            draw=draw,
            xy=(x1 + metrics.pad, y1 + metrics.pad),
            text=f"GT {name}",
            color=col,
            metrics=metrics,
            forbidden=forbidden,
            placed=placed,
            anchor_box=anchor,
        )
        return
    if isinstance(lb, YoloSegment):
        pts = [(float(x) * w, float(y) * h) for x, y in lb.points]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=col, width=metrics.line_width)
            anchor = _bbox_from_points(pts)
            forbidden.append(anchor)
            _draw_text_label(
                image=image,
                draw=draw,
                xy=(pts[0][0] + metrics.pad, pts[0][1] + metrics.pad),
                text=f"GT {name}",
                color=col,
                metrics=metrics,
                forbidden=forbidden,
                placed=placed,
                anchor_box=anchor,
            )


def _extract_bbox_xyxy(det: dict[str, Any]) -> list[float] | None:
    for key in ("bbox_roi_xyxy", "bbox_xyxy"):
        raw = det.get(key)
        if isinstance(raw, list) and len(raw) >= 4:
            try:
                return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
            except Exception:
                return None
    return None


def _extract_polygon(det: dict[str, Any], w: int, h: int) -> list[tuple[float, float]]:
    poly = det.get("polygon_roi_xy") or det.get("polygon_xy")
    if not isinstance(poly, list):
        return []
    pts: list[tuple[float, float]] = []
    for item in poly:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append((float(item[0]) * w, float(item[1]) * h))
            except Exception:
                continue
    return pts


def render_gt_overlay(
    image_path: Path,
    gt_labels: list[YoloLabel],
    class_names: dict[int, str],
    label_colors: dict[str, tuple[int, int, int]] | None = None,
    gt_faded: bool = False,
) -> Image.Image:
    image, _fmt = _prepare_canvas(image_path)
    w, h = image.size
    metrics = _vis_draw_metrics(w, h)
    draw = ImageDraw.Draw(image)
    palette = dict(label_colors or {})
    forbidden: list[tuple[int, int, int, int]] = []
    placed: list[_PlacedLabel] = []
    for lb in gt_labels:
        _draw_gt_label(image, draw, w, h, lb, class_names, metrics, palette, gt_faded, forbidden, placed)
    return image


def render_pred_overlay(
    image: Image.Image,
    pred_rows: list[dict[str, Any]],
    class_names: dict[int, str],
    label_colors: dict[str, tuple[int, int, int]] | None = None,
) -> Image.Image:
    out = image.copy()
    w, h = out.size
    metrics = _vis_draw_metrics(w, h)
    draw = ImageDraw.Draw(out)
    palette = dict(label_colors or {})
    forbidden: list[tuple[int, int, int, int]] = []
    placed: list[_PlacedLabel] = []
    for det in pred_rows:
        cls_id = _det_class_id(det)
        name = _det_class_name(det, class_names, cls_id)
        col = _label_color(name, palette)
        conf = det.get("confidence")
        label = f"PRED {name}" if conf is None else f"PRED {name} {float(conf):.2f}"
        bbox = _extract_bbox_xyxy(det)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=col, width=metrics.line_width)
            anchor = (int(x1), int(y1), int(x2), int(y2))
            forbidden.append(anchor)
            _draw_text_label(
                image=out,
                draw=draw,
                xy=(x1 + metrics.pad, max(metrics.pad, y1 - metrics.font_size - metrics.pad * 2)),
                text=label,
                color=col,
                metrics=metrics,
                forbidden=forbidden,
                placed=placed,
                anchor_box=anchor,
            )
            continue
        pts = _extract_polygon(det, w, h)
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=col, width=metrics.line_width)
            anchor = _bbox_from_points(pts)
            forbidden.append(anchor)
            _draw_text_label(
                image=out,
                draw=draw,
                xy=(pts[0][0] + metrics.pad, pts[0][1] + metrics.pad),
                text=label,
                color=col,
                metrics=metrics,
                forbidden=forbidden,
                placed=placed,
                anchor_box=anchor,
            )
    return out


def render_combined_overlay(
    image_path: Path,
    gt_labels: list[YoloLabel],
    pred_rows: list[dict[str, Any]],
    class_names: dict[int, str],
    label_colors: dict[str, tuple[int, int, int]] | None = None,
    gt_faded: bool = False,
) -> tuple[Image.Image, str | None]:
    image, original_format = _prepare_canvas(image_path)
    w, h = image.size
    metrics = _vis_draw_metrics(w, h)
    draw = ImageDraw.Draw(image)
    palette = dict(label_colors or {})
    forbidden: list[tuple[int, int, int, int]] = []
    placed: list[_PlacedLabel] = []
    for lb in gt_labels:
        _draw_gt_label(image, draw, w, h, lb, class_names, metrics, palette, gt_faded, forbidden, placed)
    for det in pred_rows:
        cls_id = _det_class_id(det)
        name = _det_class_name(det, class_names, cls_id)
        col = _label_color(name, palette)
        conf = det.get("confidence")
        label = f"PRED {name}" if conf is None else f"PRED {name} {float(conf):.2f}"
        bbox = _extract_bbox_xyxy(det)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=col, width=metrics.line_width)
            anchor = (int(x1), int(y1), int(x2), int(y2))
            forbidden.append(anchor)
            _draw_text_label(
                image=image,
                draw=draw,
                xy=(x1 + metrics.pad, max(metrics.pad, y1 - metrics.font_size - metrics.pad * 2)),
                text=label,
                color=col,
                metrics=metrics,
                forbidden=forbidden,
                placed=placed,
                anchor_box=anchor,
            )
            continue
        pts = _extract_polygon(det, w, h)
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=col, width=metrics.line_width)
            anchor = _bbox_from_points(pts)
            forbidden.append(anchor)
            _draw_text_label(
                image=image,
                draw=draw,
                xy=(pts[0][0] + metrics.pad, pts[0][1] + metrics.pad),
                text=label,
                color=col,
                metrics=metrics,
                forbidden=forbidden,
                placed=placed,
                anchor_box=anchor,
            )
    return image, original_format
