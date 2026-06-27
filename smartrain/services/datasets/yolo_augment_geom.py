from __future__ import annotations

from typing import Iterable

import albumentations as A
import numpy as np
from PIL import Image

from smartrain.services.datasets.yolo_labels import (
    YoloBBox,
    YoloLabel,
    YoloSegment,
    read_yolo_labels,
    serialize_yolo_labels,
    write_yolo_labels,
)

LabelKind = str  # "bbox" | "segment" | "mixed" | "empty"


def infer_label_kind(labels: Iterable[YoloLabel]) -> LabelKind:
    has_bbox = False
    has_seg = False
    for lb in labels:
        if isinstance(lb, YoloBBox):
            has_bbox = True
        elif isinstance(lb, YoloSegment):
            has_seg = True
    if has_bbox and has_seg:
        return "mixed"
    if has_seg:
        return "segment"
    if has_bbox:
        return "bbox"
    return "empty"


def resolve_label_kind(label_path: str, *, label_type: str = "auto") -> LabelKind:
    mode = str(label_type or "auto").strip().lower()
    labels = read_yolo_labels(label_path)
    detected = infer_label_kind(labels)
    if mode == "auto":
        return detected
    if mode in {"bbox", "segment"}:
        if detected == "mixed":
            return "mixed"
        if detected == "empty":
            return "empty"
        if mode == "bbox" and detected != "bbox":
            return detected
        if mode == "segment" and detected != "segment":
            return detected
        return mode
    return detected


def count_label_instances(label_path: str) -> int:
    return len(read_yolo_labels(label_path))


def _sanitize_bbox(lb: YoloBBox, *, min_size: float = 1e-6) -> YoloBBox | None:
    x1 = lb.cx - lb.w / 2.0
    y1 = lb.cy - lb.h / 2.0
    x2 = lb.cx + lb.w / 2.0
    y2 = lb.cy + lb.h / 2.0
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return YoloBBox(
        cls_id=int(lb.cls_id),
        cx=float((x1 + x2) / 2.0),
        cy=float((y1 + y2) / 2.0),
        w=float(x2 - x1),
        h=float(y2 - y1),
    )


def _segments_to_keypoints(
    labels: list[YoloLabel], *, w: int, h: int
) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    keypoints: list[tuple[float, float]] = []
    groups: list[tuple[int, int]] = []
    for lb in labels:
        if not isinstance(lb, YoloSegment):
            continue
        groups.append((int(lb.cls_id), len(lb.points)))
        for x, y in lb.points:
            keypoints.append((float(x) * w, float(y) * h))
    return keypoints, groups


def _keypoints_to_segments(
    keypoints: list[tuple[float, float]],
    groups: list[tuple[int, int]],
    *,
    w: int,
    h: int,
) -> list[YoloSegment]:
    out: list[YoloSegment] = []
    i = 0
    for cls_id, n in groups:
        pts: list[tuple[float, float]] = []
        for _ in range(n):
            if i >= len(keypoints):
                break
            x, y = keypoints[i]
            i += 1
            pts.append((max(0.0, min(1.0, float(x) / w)), max(0.0, min(1.0, float(y) / h))))
        if len(pts) >= 3:
            out.append(YoloSegment(cls_id=int(cls_id), points=tuple(pts)))
    return out


def apply_albumentations_to_labels(
    image: np.ndarray,
    labels: list[YoloLabel],
    pipeline: A.Compose,
) -> tuple[np.ndarray, list[YoloLabel]]:
    h, w = image.shape[:2]
    bboxes: list[tuple[float, float, float, float]] = []
    class_labels: list[int] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            sanitized = _sanitize_bbox(lb)
            if sanitized is None:
                continue
            bboxes.append((sanitized.cx, sanitized.cy, sanitized.w, sanitized.h))
            class_labels.append(int(sanitized.cls_id))
    keypoints, groups = _segments_to_keypoints(labels, w=w, h=h)

    kwargs: dict = {"image": image}
    if bboxes:
        kwargs["bboxes"] = bboxes
        kwargs["class_labels"] = class_labels
    if keypoints:
        kwargs["keypoints"] = keypoints

    transformed = pipeline(**kwargs)
    new_img = transformed["image"]
    out: list[YoloLabel] = []
    if bboxes and "bboxes" in transformed:
        for cls, (cx, cy, bw, bh) in zip(transformed["class_labels"], transformed["bboxes"], strict=False):
            sanitized = _sanitize_bbox(YoloBBox(cls_id=int(cls), cx=float(cx), cy=float(cy), w=float(bw), h=float(bh)))
            if sanitized is not None:
                out.append(sanitized)
    if groups and "keypoints" in transformed:
        out.extend(_keypoints_to_segments(transformed["keypoints"], groups, w=new_img.shape[1], h=new_img.shape[0]))
    return new_img, out


def rotate_labels_with_matrix(
    labels: list[YoloLabel],
    matrix: np.ndarray,
    *,
    w: int,
    h: int,
) -> list[YoloLabel]:
    out: list[YoloLabel] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            x1 = (lb.cx - lb.w / 2.0) * w
            y1 = (lb.cy - lb.h / 2.0) * h
            x2 = (lb.cx + lb.w / 2.0) * w
            y2 = (lb.cy + lb.h / 2.0) * h
            corners = np.array(
                [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
                dtype=np.float32,
            )
            tr = (matrix @ corners.T).T
            nx1 = max(0.0, min(float(w), float(np.min(tr[:, 0]))))
            ny1 = max(0.0, min(float(h), float(np.min(tr[:, 1]))))
            nx2 = max(0.0, min(float(w), float(np.max(tr[:, 0]))))
            ny2 = max(0.0, min(float(h), float(np.max(tr[:, 1]))))
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            sanitized = _sanitize_bbox(
                YoloBBox(
                    cls_id=int(lb.cls_id),
                    cx=(nx1 + nx2) / 2.0 / w,
                    cy=(ny1 + ny2) / 2.0 / h,
                    w=(nx2 - nx1) / w,
                    h=(ny2 - ny1) / h,
                )
            )
            if sanitized is not None:
                out.append(sanitized)
            continue
        if isinstance(lb, YoloSegment):
            pts: list[tuple[float, float]] = []
            for x, y in lb.points:
                px, py = float(x) * w, float(y) * h
                tr = matrix @ np.array([px, py, 1.0], dtype=np.float32)
                pts.append((max(0.0, min(1.0, float(tr[0]) / w)), max(0.0, min(1.0, float(tr[1]) / h))))
            if len(pts) >= 3:
                out.append(YoloSegment(cls_id=int(lb.cls_id), points=tuple(pts)))
    return out


def write_augment_label_file(path: str, labels: list[YoloLabel]) -> None:
    write_yolo_labels(path, labels)


def read_augment_label_file(path: str) -> list[YoloLabel]:
    return read_yolo_labels(path)


def labels_to_legacy_tuples(labels: list[YoloLabel]) -> list[tuple[int, float, float, float, float]]:
    """BBox-only legacy tuple form for donor pool / copy-paste paths."""
    out: list[tuple[int, float, float, float, float]] = []
    for lb in labels:
        if isinstance(lb, YoloBBox):
            sanitized = _sanitize_bbox(lb)
            if sanitized is not None:
                out.append((sanitized.cls_id, sanitized.cx, sanitized.cy, sanitized.w, sanitized.h))
        elif isinstance(lb, YoloSegment):
            xs = [p[0] for p in lb.points]
            ys = [p[1] for p in lb.points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            sanitized = _sanitize_bbox(
                YoloBBox(
                    cls_id=int(lb.cls_id),
                    cx=(x1 + x2) / 2.0,
                    cy=(y1 + y2) / 2.0,
                    w=max(x2 - x1, 1e-6),
                    h=max(y2 - y1, 1e-6),
                )
            )
            if sanitized is not None:
                out.append((sanitized.cls_id, sanitized.cx, sanitized.cy, sanitized.w, sanitized.h))
    return out


def legacy_tuples_to_serialized(labels: list[tuple[int, float, float, float, float]]) -> str:
    return serialize_yolo_labels(
        [
            YoloBBox(cls_id=int(c), cx=float(x), cy=float(y), w=float(w), h=float(h))
            for c, x, y, w, h in labels
        ]
    )
