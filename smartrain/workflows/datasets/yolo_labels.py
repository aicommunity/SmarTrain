from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


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

