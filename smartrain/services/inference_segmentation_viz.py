from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


def _class_color(class_name: str) -> tuple[int, int, int]:
    h = abs(hash(str(class_name))) % 360
    return (
        int(80 + (h * 3) % 176),
        int(80 + (h * 5) % 176),
        int(80 + (h * 7) % 176),
    )


def render_segment_overlay(
    image_path: str,
    segments: list[dict],
    *,
    out_path: str,
    line_width: int = 2,
) -> str:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        cls = str(seg.get("class_name") or seg.get("class_id") or "obj")
        conf = seg.get("confidence")
        poly = seg.get("polygon_roi_xy") or seg.get("polygon_xy")
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        pts: list[tuple[float, float]] = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]) * w, float(p[1]) * h))
        if len(pts) < 3:
            continue
        col = _class_color(cls)
        draw.line(pts + [pts[0]], fill=col, width=line_width)
        label = f"{cls}" if conf is None else f"{cls} {float(conf):.2f}"
        draw.text((pts[0][0] + 2, pts[0][1] + 2), label, fill=col)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="JPEG", quality=92)
    return str(out)


def save_inference_segment_overlays(
    report_path: str,
    *,
    overlay_dir: str | None = None,
    max_images: int = 50,
) -> list[str]:
    report = Path(report_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        return []
    base_dir = overlay_dir or str(report.parent / "overlays")
    saved: list[str] = []
    for idx, item in enumerate(images[: max(0, int(max_images))]):
        if not isinstance(item, dict):
            continue
        src = str(item.get("image_path") or item.get("source_path") or "")
        if not src or not os.path.isfile(src):
            continue
        outputs = item.get("task_outputs") if isinstance(item.get("task_outputs"), dict) else {}
        segments = outputs.get("segments") if isinstance(outputs, dict) else None
        if not isinstance(segments, list) or not segments:
            continue
        stem = Path(src).stem
        out_path = os.path.join(base_dir, f"{stem}_overlay.jpg")
        saved.append(render_segment_overlay(src, segments, out_path=out_path))
    return saved
