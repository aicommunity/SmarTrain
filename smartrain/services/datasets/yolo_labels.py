from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple


@dataclass(frozen=True)
class YoloBBox:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True)
class YoloSegment:
    cls_id: int
    points: Tuple[Tuple[float, float], ...]  # normalized (x,y)


YoloLabel = YoloBBox | YoloSegment


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def xyxy_pixels_to_yolo_bbox(cls_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> YoloBBox | None:
    if img_w <= 0 or img_h <= 0:
        return None
    bw = max(0.0, float(x2) - float(x1))
    bh = max(0.0, float(y2) - float(y1))
    if bw <= 0.0 or bh <= 0.0:
        return None
    cx = float(x1) + bw / 2.0
    cy = float(y1) + bh / 2.0
    return YoloBBox(
        cls_id=int(cls_id),
        cx=_clip01(cx / float(img_w)),
        cy=_clip01(cy / float(img_h)),
        w=_clip01(bw / float(img_w)),
        h=_clip01(bh / float(img_h)),
    )


def polygon_pixels_to_yolo_segment(cls_id: int, polygon_xy: list[Any], img_w: int, img_h: int) -> YoloSegment | None:
    if img_w <= 0 or img_h <= 0:
        return None
    pts: list[tuple[float, float]] = []
    for point in polygon_xy:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except Exception:
            continue
        if max(abs(x), abs(y)) <= 1.0 + 1e-6:
            pts.append((_clip01(x), _clip01(y)))
        else:
            pts.append((_clip01(x / float(img_w)), _clip01(y / float(img_h))))
    if len(pts) < 3:
        return None
    return YoloSegment(cls_id=int(cls_id), points=tuple(pts))


def _det_class_id(det: dict[str, Any]) -> int:
    for key in ("class_index", "class_id"):
        if key in det:
            try:
                return int(det[key])
            except Exception:
                continue
    return 0


def _extract_bbox_original_xyxy(det: dict[str, Any]) -> list[float] | None:
    for key in ("bbox_original_xyxy", "bbox_roi_xyxy", "bbox_xyxy"):
        raw = det.get(key)
        if isinstance(raw, list) and len(raw) >= 4:
            try:
                return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
            except Exception:
                return None
    return None


def _extract_polygon_original_xy(det: dict[str, Any]) -> list[Any] | None:
    for key in ("polygon_original_xy", "polygon_roi_xy", "polygon_xy"):
        raw = det.get(key)
        if isinstance(raw, list) and raw:
            return raw
    return None


from smartrain.core.training.train_profile import task_to_metadata_task_type


def task_output_dict_to_yolo_label(
    det: dict[str, Any],
    img_w: int,
    img_h: int,
    *,
    task_type: str | None = None,
) -> YoloLabel | None:
    cls_id = _det_class_id(det)
    bbox = _extract_bbox_original_xyxy(det)
    poly = _extract_polygon_original_xy(det)
    resolved = task_to_metadata_task_type(task_type)
    if resolved == "segmentation" and poly is not None:
        segment = polygon_pixels_to_yolo_segment(cls_id, poly, img_w, img_h)
        if segment is not None:
            return segment
    if bbox is not None:
        return xyxy_pixels_to_yolo_bbox(cls_id, bbox[0], bbox[1], bbox[2], bbox[3], img_w, img_h)
    if poly is not None:
        return polygon_pixels_to_yolo_segment(cls_id, poly, img_w, img_h)
    return None


def task_outputs_to_yolo_labels(task_type: str, outputs: list[dict[str, Any]], img_w: int, img_h: int) -> list[YoloLabel]:
    labels: list[YoloLabel] = []
    resolved = task_to_metadata_task_type(task_type)
    for item in outputs:
        if not isinstance(item, dict):
            continue
        lb = task_output_dict_to_yolo_label(item, img_w, img_h, task_type=resolved)
        if lb is not None:
            labels.append(lb)
    return labels


def read_yolo_labels(path: str) -> list[YoloLabel]:
    """
    Reads YOLO label file.
    Supports:
    - detection bbox: class_id cx cy w h
    - segment: class_id x1 y1 x2 y2 ...
    """
    p = Path(path)
    if not p.is_file():
        return []
    out: list[YoloLabel] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        if len(parts) == 5:
            try:
                cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                continue
            out.append(YoloBBox(cls_id=cls_id, cx=cx, cy=cy, w=w, h=h))
            continue
        # segment (must be even count >= 6)
        rest = parts[1:]
        if len(rest) < 6 or (len(rest) % 2) != 0:
            continue
        try:
            nums = [float(x) for x in rest]
        except ValueError:
            continue
        pts: list[tuple[float, float]] = []
        for i in range(0, len(nums), 2):
            pts.append((nums[i], nums[i + 1]))
        out.append(YoloSegment(cls_id=cls_id, points=tuple(pts)))
    return out


def write_yolo_labels(path: str, labels: Iterable[YoloLabel]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(serialize_yolo_labels(labels), encoding="utf-8")


def serialize_yolo_labels(labels: Iterable[YoloLabel]) -> str:
    lines: list[str] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            lines.append(
                f"{int(lb.cls_id)} {float(lb.cx):.8f} {float(lb.cy):.8f} {float(lb.w):.8f} {float(lb.h):.8f}"
            )
        else:
            coords: list[str] = []
            for x, y in lb.points:
                coords.append(f"{float(x):.6f}")
                coords.append(f"{float(y):.6f}")
            lines.append(f"{int(lb.cls_id)} " + " ".join(coords))
    return "\n".join(lines) + ("\n" if lines else "")


def rotate_yolo_labels_90cw_k(
    labels: Iterable[YoloLabel],
    *,
    w: int,
    h: int,
    k: int,
) -> tuple[list[YoloLabel], int, int]:
    """
    Rotate labels by k * 90 degrees clockwise.
    Returns (rotated_labels, new_w, new_h).
    """
    kk = int(k) % 4
    if kk == 0:
        return list(labels), int(w), int(h)
    new_w, new_h = (h, w) if (kk % 2 == 1) else (w, h)

    def rot_pt(px: float, py: float) -> tuple[float, float]:
        # coordinates in pixels, origin top-left, x right, y down
        x, y = float(px), float(py)
        if kk == 1:  # 90 cw
            return float(h) - y, x
        if kk == 2:  # 180
            return float(w) - x, float(h) - y
        # kk == 3: 270 cw (90 ccw)
        return y, float(w) - x

    out: list[YoloLabel] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            x1 = (lb.cx - lb.w / 2.0) * w
            y1 = (lb.cy - lb.h / 2.0) * h
            x2 = (lb.cx + lb.w / 2.0) * w
            y2 = (lb.cy + lb.h / 2.0) * h
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            tr = [rot_pt(x, y) for x, y in corners]
            xs = [p[0] for p in tr]
            ys = [p[1] for p in tr]
            nx1 = max(0.0, min(float(new_w), min(xs)))
            nx2 = max(0.0, min(float(new_w), max(xs)))
            ny1 = max(0.0, min(float(new_h), min(ys)))
            ny2 = max(0.0, min(float(new_h), max(ys)))
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            bw = (nx2 - nx1) / float(new_w)
            bh = (ny2 - ny1) / float(new_h)
            cx = (nx1 + nx2) / 2.0 / float(new_w)
            cy = (ny1 + ny2) / 2.0 / float(new_h)
            out.append(YoloBBox(cls_id=int(lb.cls_id), cx=float(cx), cy=float(cy), w=float(bw), h=float(bh)))
            continue
        # segment
        pts_px = [(x * w, y * h) for x, y in lb.points]
        tr_px = [rot_pt(x, y) for x, y in pts_px]
        tr_norm: list[tuple[float, float]] = []
        for x, y in tr_px:
            x = max(0.0, min(float(new_w), float(x)))
            y = max(0.0, min(float(new_h), float(y)))
            tr_norm.append((float(x) / float(new_w), float(y) / float(new_h)))
        out.append(YoloSegment(cls_id=int(lb.cls_id), points=tuple(tr_norm)))
    return out, int(new_w), int(new_h)

