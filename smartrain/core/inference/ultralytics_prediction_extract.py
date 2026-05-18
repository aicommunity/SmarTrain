"""Normalize Ultralytics `predict()` results into SmarTrain task_outputs dicts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smartrain.core.training.train_profile import task_to_metadata_task_type


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    except Exception:
        return None


def _names_map(model: Any | None) -> dict[Any, Any]:
    names = getattr(model, "names", {}) if model is not None else {}
    return names if isinstance(names, dict) else {}


def _resolve_class_names(model: Any | None, class_names: Mapping[Any, str] | None) -> dict[Any, Any]:
    if class_names is not None:
        return dict(class_names)
    return _names_map(model)


def _extract_detections(preds: Any, names: dict[Any, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not preds:
        return rows
    r = preds[0]
    boxes_obj = getattr(r, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return rows
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    for i in range(len(xyxy)):
        cls_idx = int(cls[i])
        class_name = (
            str(names.get(cls_idx, names.get(str(cls_idx), cls_idx))) if names else str(cls_idx)
        )
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        rows.append(
            {
                "bbox_roi_xyxy": [x1, y1, x2, y2],
                "class_index": cls_idx,
                "class_name": class_name,
                "confidence": float(confs[i]),
            }
        )
    return rows


def _extract_classification(preds: Any, names: dict[Any, Any]) -> dict[str, Any] | None:
    if not preds:
        return None
    r = preds[0]
    probs = getattr(r, "probs", None)
    if probs is None:
        return None

    def _class_name(idx: int) -> str:
        return str(names.get(idx, names.get(str(idx), idx))) if names else str(idx)

    top1_idx = getattr(probs, "top1", None)
    top1_conf = getattr(probs, "top1conf", None)
    top5 = getattr(probs, "top5", None)
    top5conf = getattr(probs, "top5conf", None)

    top_k: list[dict[str, Any]] = []
    if isinstance(top5, (list, tuple)):
        for i, cls_idx_raw in enumerate(top5):
            try:
                cls_idx = int(cls_idx_raw)
            except Exception:
                continue
            conf_val = None
            if isinstance(top5conf, (list, tuple)) and i < len(top5conf):
                conf_val = _to_float(top5conf[i])
            top_k.append(
                {
                    "class_index": cls_idx,
                    "class_name": _class_name(cls_idx),
                    "confidence": conf_val,
                }
            )
    if top1_idx is None:
        return {"top_k": top_k}
    try:
        top1_i = int(top1_idx)
    except Exception:
        return {"top_k": top_k}
    return {
        "top1": {
            "class_index": top1_i,
            "class_name": _class_name(top1_i),
            "confidence": _to_float(top1_conf),
        },
        "top_k": top_k,
    }


def _extract_segments(preds: Any, names: dict[Any, Any]) -> list[dict[str, Any]]:
    if not preds:
        return []
    r = preds[0]
    boxes_obj = getattr(r, "boxes", None)
    masks_obj = getattr(r, "masks", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    polygons = getattr(masks_obj, "xy", None) if masks_obj is not None else None
    rows: list[dict[str, Any]] = []
    for i in range(len(xyxy)):
        cls_idx = int(cls[i])
        class_name = (
            str(names.get(cls_idx, names.get(str(cls_idx), cls_idx))) if names else str(cls_idx)
        )
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        polygon_xy: list[list[float]] = []
        if isinstance(polygons, (list, tuple)) and i < len(polygons):
            poly = polygons[i]
            if hasattr(poly, "tolist"):
                poly = poly.tolist()
            if isinstance(poly, list):
                for point in poly:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            polygon_xy.append([float(point[0]), float(point[1])])
                        except Exception:
                            continue
        rows.append(
            {
                "bbox_roi_xyxy": [x1, y1, x2, y2],
                "class_index": cls_idx,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "polygon_roi_xy": polygon_xy,
            }
        )
    return rows


def extract_task_outputs_from_ultralytics_preds(
    model: Any | None,
    preds: Any,
    *,
    task_type: str | None = None,
    class_names: Mapping[Any, str] | None = None,
) -> dict[str, Any]:
    """
    Build task_outputs dict aligned with inference CLI / external launcher JSON.

    ``model`` may be ``None`` (external subprocess path); class names default to index strings unless
    ``class_names`` is provided (overrides ``model.names`` when both are present).
    """
    resolved = task_to_metadata_task_type(task_type)
    names = _resolve_class_names(model, class_names)
    if not preds:
        if resolved == "classification":
            return {"classification": {}}
        if resolved == "segmentation":
            return {"segments": []}
        return {"detections": []}

    if resolved == "classification":
        classification = _extract_classification(preds, names)
        return {"classification": classification or {}}
    if resolved == "segmentation":
        return {"segments": _extract_segments(preds, names)}
    return {"detections": _extract_detections(preds, names)}
