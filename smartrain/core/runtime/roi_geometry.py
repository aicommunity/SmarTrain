"""Geometry helpers for ROI dataset processing."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np


def clamp_crop(
    x1: float, y1: float, x2: float, y2: float, pad: int, iw: int, ih: int
) -> Tuple[int, int, int, int]:
    x1c = max(0, int(round(x1 - pad)))
    y1c = max(0, int(round(y1 - pad)))
    x2c = min(iw, int(round(x2 + pad)))
    y2c = min(ih, int(round(y2 + pad)))
    if x2c <= x1c:
        x2c = min(iw, x1c + 1)
    if y2c <= y1c:
        y2c = min(ih, y1c + 1)
    return x1c, y1c, x2c, y2c


def select_roi_boxes(
    xyxy: Any,
    cls: Any,
    confs: Any,
    class_ids: Optional[Sequence[int]],
    policy: str,
    iw: int,
    ih: int,
) -> list[Tuple[float, float, float, float]]:
    if xyxy is None or len(xyxy) == 0:
        return []
    xyxy = np.asarray(xyxy, dtype=np.float64)
    cls = np.asarray(cls, dtype=np.int64).reshape(-1)
    confs = np.asarray(confs, dtype=np.float64).reshape(-1)
    boxes: list[Tuple[float, float, float, float, int, float]] = []
    for index in range(xyxy.shape[0]):
        class_id = int(cls[index])
        if class_ids is not None and class_id not in class_ids:
            continue
        x1, y1, x2, y2 = xyxy[index]
        x1 = max(0, min(iw, x1))
        x2 = max(0, min(iw, x2))
        y1 = max(0, min(ih, y1))
        y2 = max(0, min(ih, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((float(x1), float(y1), float(x2), float(y2), class_id, float(confs[index])))
    if not boxes:
        return []
    if policy == "per_box":
        boxes.sort(key=lambda box: -box[5])
        return [(box[0], box[1], box[2], box[3]) for box in boxes]
    if policy == "largest":
        box = max(boxes, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
        return [(box[0], box[1], box[2], box[3])]
    if policy == "best_conf":
        box = max(boxes, key=lambda item: item[5])
        return [(box[0], box[1], box[2], box[3])]
    return [(
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )]


def full_image_crop(iw: int, ih: int) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, float(iw), float(ih)
