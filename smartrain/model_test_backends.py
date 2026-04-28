from __future__ import annotations

import os
import sys
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from smartrain.confidence_recommendation import (
    compute_confidence_recommendations,
    write_not_available_recommendations,
    write_recommendation_file,
)
from smartrain.model_test_service import (
    format_metrics_path,
    format_metrics_path_for_split,
    format_recommendation_path,
    format_test_dir,
    persist_target_test_artifacts_state,
)


@dataclass
class BackendRunResult:
    format: str
    backend: str
    success: bool
    test_start_time: datetime | None
    test_end_time: datetime | None
    inference: dict[str, Any]
    target_path: str | None
    error: str | None = None


@dataclass
class _Pred:
    image_path: str
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _Gt:
    image_path: str
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class _Box:
    cls_id: int
    conf: float | None
    x1: float
    y1: float
    x2: float
    y2: float


def _iou(a: _Box, b: _Box) -> float:
    xx1 = max(a.x1, b.x1)
    yy1 = max(a.y1, b.y1)
    xx2 = min(a.x2, b.x2)
    yy2 = min(a.y2, b.y2)
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2.0
    y1 = (yc * img_h) - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return x1, y1, x2, y2


def _label_path_for_image(img_path: str) -> str:
    p = Path(img_path)
    parts = list(p.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return str(Path(*parts).with_suffix(".txt"))
    except ValueError:
        return ""


def _read_gt_boxes(image_path: str, img_w: int, img_h: int) -> list[_Gt]:
    label_path = _label_path_for_image(image_path)
    if not label_path or not os.path.isfile(label_path):
        return []
    out: list[_Gt] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            x1, y1, x2, y2 = _xywhn_to_xyxy(xc, yc, w, h, img_w, img_h)
            out.append(_Gt(image_path=image_path, cls_id=cls_id, x1=x1, y1=y1, x2=x2, y2=y2))
    return out


def _split_images_from_yaml(data_yaml_path: str, split_name: str, limit: int) -> list[str]:
    data = _load_data_cfg(data_yaml_path)
    rel = data.get(split_name)
    if not rel or not isinstance(rel, str):
        raise ValueError(f"data.yaml has no split={split_name!r}")
    root = os.path.dirname(os.path.abspath(data_yaml_path))
    img_dir = os.path.abspath(os.path.join(root, rel))
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(img_dir):
        for name in sorted(filenames):
            if name.lower().endswith(exts):
                out.append(os.path.join(dirpath, name))
    return out[:limit] if limit and limit > 0 else out


def _save_metrics_csv_for_format(test_result: Any, root_dir: str, format_name: str) -> str:
    csv_file = format_metrics_path(root_dir, format_name)
    csv_data = test_result.to_csv()
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(csv_data)
    return csv_file


def _load_data_cfg(data_yaml_path: str) -> dict[str, Any]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload if isinstance(payload, dict) else {}


def _load_names(data_yaml_path: str) -> list[str]:
    payload = _load_data_cfg(data_yaml_path)
    names = payload.get("names")
    if isinstance(names, list):
        return [str(x) for x in names]
    if isinstance(names, dict):
        try:
            return [str(v) for _k, v in sorted(names.items())]
        except Exception:
            return [str(v) for v in names.values()]
    return []


def _resolve_imgsz_from_onnx(session: Any, imgsz: int | None) -> tuple[int, int]:
    if imgsz is not None:
        return int(imgsz), int(imgsz)
    try:
        shape = list(session.get_inputs()[0].shape)
        if len(shape) == 4:
            h = int(shape[2]) if isinstance(shape[2], (int, float)) else 640
            w = int(shape[3]) if isinstance(shape[3], (int, float)) else 640
            return max(32, h), max(32, w)
    except Exception:
        pass
    return 640, 640


def _letterbox(im: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    shape = im.shape[:2]
    new_h, new_w = new_shape
    r = min(new_h / shape[0], new_w / shape[1])
    resized_h = max(1, int(round(shape[0] * r)))
    resized_w = max(1, int(round(shape[1] * r)))
    resized = np.array(Image.fromarray(im).resize((resized_w, resized_h)))
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    top = int(round((new_h - resized_h) / 2 - 0.1))
    left = int(round((new_w - resized_w) / 2 - 0.1))
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas, r, (float(left), float(top))


def _preprocess_image(image_path: str, input_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], float, tuple[float, float]]:
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im)
    orig_h, orig_w = arr.shape[:2]
    boxed, gain, pad = _letterbox(arr, input_hw)
    chw = boxed.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(chw, 0), (orig_h, orig_w), gain, pad


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _clip_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)
    return boxes


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(0.0, xx2 - xx1)
    ih = np.maximum(0.0, yy2 - yy1)
    inter = iw * ih
    box_area = np.maximum(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union = box_area + boxes_area - inter
    out = np.zeros_like(union, dtype=np.float32)
    np.divide(inter, union, out=out, where=union > 0.0)
    return out


def _nms_classwise(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, iou_thr: float) -> list[int]:
    keep: list[int] = []
    for cls_id in np.unique(cls_ids):
        inds = np.where(cls_ids == cls_id)[0]
        order = inds[np.argsort(scores[inds])[::-1]]
        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            ious = _box_iou_np(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= float(iou_thr)]
    return keep


def _select_output_tensor(outputs: list[np.ndarray]) -> np.ndarray:
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    arr = outputs[0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] in {6, 7, 84, 85} and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return np.asarray(arr, dtype=np.float32)


def _decode_onnx_predictions(
    raw: np.ndarray,
    *,
    image_path: str,
    names: list[str],
    conf_thr: float,
    iou_thr: float,
    orig_hw: tuple[int, int],
    gain: float,
    pad: tuple[float, float],
) -> list[_Pred]:
    if raw.size == 0:
        return []
    num_classes = len(names)
    if raw.ndim != 2:
        raw = raw.reshape(-1, raw.shape[-1])
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    cls_ids: np.ndarray
    cols = int(raw.shape[1]) if raw.ndim == 2 else 0
    if cols == 6:
        boxes_xyxy = raw[:, :4].copy()
        scores = raw[:, 4].copy()
        cls_ids = raw[:, 5].astype(int)
    else:
        boxes = raw[:, :4].copy()
        if num_classes > 0 and cols == num_classes + 5:
            obj = raw[:, 4:5]
            class_scores = raw[:, 5:]
            fused = obj * class_scores
        else:
            class_scores = raw[:, 4:]
            fused = class_scores
        cls_ids = np.argmax(fused, axis=1).astype(int)
        scores = fused[np.arange(fused.shape[0]), cls_ids]
        boxes_xyxy = _xywh_to_xyxy(boxes)
    valid = scores >= float(conf_thr)
    if not np.any(valid):
        return []
    boxes_xyxy = boxes_xyxy[valid]
    scores = scores[valid]
    cls_ids = cls_ids[valid]
    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy /= max(gain, 1e-9)
    boxes_xyxy = _clip_boxes_xyxy(boxes_xyxy, orig_hw[1], orig_hw[0])
    keep = _nms_classwise(boxes_xyxy, scores, cls_ids, iou_thr)
    preds: list[_Pred] = []
    for idx in keep:
        b = boxes_xyxy[idx]
        preds.append(
            _Pred(
                image_path=image_path,
                cls_id=int(cls_ids[idx]),
                conf=float(scores[idx]),
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
            )
        )
    return preds


def _collect_gt(data_yaml_path: str, split: str) -> tuple[list[_Gt], dict[str, list[_Gt]], list[str]]:
    image_paths = _split_images_from_yaml(data_yaml_path, split, 0)
    gt_rows: list[_Gt] = []
    by_image: dict[str, list[_Gt]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as im:
            img_w, img_h = im.size
        gt_boxes = _read_gt_boxes(image_path, img_w, img_h)
        rows = [
            _Gt(
                image_path=image_path,
                cls_id=int(box.cls_id),
                x1=float(box.x1),
                y1=float(box.y1),
                x2=float(box.x2),
                y2=float(box.y2),
            )
            for box in gt_boxes
        ]
        gt_rows.extend(rows)
        by_image[image_path] = rows
    return gt_rows, by_image, image_paths


def _match_class_predictions(
    preds: list[_Pred],
    gts: list[_Gt],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    gt_by_image: dict[str, list[_Gt]] = {}
    for gt in gts:
        gt_by_image.setdefault(gt.image_path, []).append(gt)
    matched: dict[str, set[int]] = {}
    sorted_preds = sorted(preds, key=lambda x: x.conf, reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)
    for idx, pred in enumerate(sorted_preds):
        candidates = gt_by_image.get(pred.image_path, [])
        if not candidates:
            fp[idx] = 1.0
            continue
        pred_box = _Box(pred.cls_id, pred.conf, pred.x1, pred.y1, pred.x2, pred.y2)
        best_iou = 0.0
        best_gt_idx: int | None = None
        for gt_idx, gt in enumerate(candidates):
            if gt_idx in matched.setdefault(pred.image_path, set()):
                continue
            gt_box = _Box(gt.cls_id, None, gt.x1, gt.y1, gt.x2, gt.y2)
            cur_iou = _iou(pred_box, gt_box)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gt_idx = gt_idx
        if best_gt_idx is not None and best_iou >= float(iou_thr):
            matched[pred.image_path].add(best_gt_idx)
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0
    return tp, fp, len(gts)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _build_pr_payload(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
    iou_thr: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float], dict[int, tuple[np.ndarray, np.ndarray]]]:
    pr_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    ap_by_class: dict[int, float] = {}
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    recall_grid = np.linspace(0.0, 1.0, 101)
    for class_id, class_name in enumerate(names):
        class_preds = [p for p in preds if int(p.cls_id) == class_id]
        class_gts = [g for g in gt_rows if int(g.cls_id) == class_id]
        if not class_gts:
            continue
        tp, fp, total_gt = _match_class_predictions(class_preds, class_gts, iou_thr)
        if tp.size == 0:
            recall = np.asarray([0.0], dtype=np.float32)
            precision = np.asarray([0.0], dtype=np.float32)
        else:
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / max(float(total_gt), 1.0)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = _compute_ap(recall, precision)
        ap_by_class[class_id] = ap
        interp_precision = np.interp(recall_grid, recall, precision, left=1.0 if precision.size else 0.0, right=0.0)
        curves[class_id] = (recall_grid, interp_precision)
        for r, p in zip(recall_grid, interp_precision):
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "recall": float(r),
                    "precision": float(p),
                    "ap": float(ap),
                }
            )
        pr_rows.append({"class_id": class_id, "class_name": class_name, "ap": float(ap)})
    pr_df = pd.DataFrame(pr_rows)
    pr_per_class_df = pd.DataFrame(per_class_rows)
    return pr_df, pr_per_class_df, ap_by_class, curves


def _build_ultralytics_style_stats(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    iouv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_image_gt: dict[str, list[_Gt]] = {}
    by_image_pred: dict[str, list[_Pred]] = {}
    for gt in gt_rows:
        by_image_gt.setdefault(gt.image_path, []).append(gt)
    for pred in preds:
        by_image_pred.setdefault(pred.image_path, []).append(pred)

    all_images = sorted(set(by_image_gt) | set(by_image_pred))
    correct_rows: list[np.ndarray] = []
    conf_rows: list[float] = []
    pred_cls_rows: list[int] = []
    target_cls_rows: list[int] = []

    for image_path in all_images:
        img_gts = by_image_gt.get(image_path, [])
        img_preds = by_image_pred.get(image_path, [])
        if img_gts:
            target_cls_rows.extend([int(g.cls_id) for g in img_gts])
        if not img_preds:
            continue

        pred_boxes = np.asarray([[p.x1, p.y1, p.x2, p.y2] for p in img_preds], dtype=np.float32)
        pred_cls = np.asarray([int(p.cls_id) for p in img_preds], dtype=np.int32)
        pred_conf = np.asarray([float(p.conf) for p in img_preds], dtype=np.float32)
        gt_boxes = np.asarray([[g.x1, g.y1, g.x2, g.y2] for g in img_gts], dtype=np.float32) if img_gts else np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.asarray([int(g.cls_id) for g in img_gts], dtype=np.int32) if img_gts else np.zeros((0,), dtype=np.int32)

        correct = np.zeros((len(img_preds), len(iouv)), dtype=bool)
        if len(img_gts):
            iou_mat = np.zeros((len(img_gts), len(img_preds)), dtype=np.float32)
            for gi in range(len(img_gts)):
                iou_mat[gi, :] = _box_iou_np(gt_boxes[gi], pred_boxes)
            class_mask = gt_cls[:, None] == pred_cls[None, :]
            iou_mat = iou_mat * class_mask
            for ti, thr in enumerate(iouv.tolist()):
                matches = np.argwhere(iou_mat >= float(thr))
                if matches.size == 0:
                    continue
                if len(matches) > 1:
                    vals = iou_mat[matches[:, 0], matches[:, 1]]
                    order = np.argsort(-vals)
                    matches = matches[order]
                    _, keep_pred = np.unique(matches[:, 1], return_index=True)
                    matches = matches[keep_pred]
                    _, keep_gt = np.unique(matches[:, 0], return_index=True)
                    matches = matches[keep_gt]
                correct[matches[:, 1].astype(int), ti] = True

        correct_rows.append(correct)
        conf_rows.extend(pred_conf.tolist())
        pred_cls_rows.extend(pred_cls.tolist())

    tp = np.concatenate(correct_rows, axis=0) if correct_rows else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.asarray(conf_rows, dtype=np.float32) if conf_rows else np.zeros((0,), dtype=np.float32)
    pred_cls = np.asarray(pred_cls_rows, dtype=np.int32) if pred_cls_rows else np.zeros((0,), dtype=np.int32)
    target_cls = np.asarray(target_cls_rows, dtype=np.int32) if target_cls_rows else np.zeros((0,), dtype=np.int32)
    return tp, conf, pred_cls, target_cls


def _compute_ultralytics_style_payload(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
) -> dict[str, Any]:
    iouv = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    tp, conf, pred_cls, target_cls = _build_ultralytics_style_stats(preds, gt_rows, iouv)
    names_map = {idx: name for idx, name in enumerate(names)}
    (
        _tp_cnt,
        _fp_cnt,
        p_cls,
        r_cls,
        f1_cls,
        ap,
        unique_classes,
        p_curve,
        r_curve,
        f1_curve,
        x,
        prec_values,
    ) = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        plot=False,
        names=names_map,
    )

    ap50_by_class: dict[int, float] = {}
    ap5095_by_class: dict[int, float] = {}
    pr_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for idx, cls_id in enumerate(unique_classes.tolist() if isinstance(unique_classes, np.ndarray) else []):
        ap50_val = float(ap[idx, 0]) if ap.ndim == 2 and idx < ap.shape[0] else 0.0
        ap5095_val = float(ap[idx].mean()) if ap.ndim == 2 and idx < ap.shape[0] else 0.0
        ap50_by_class[int(cls_id)] = ap50_val
        ap5095_by_class[int(cls_id)] = ap5095_val
        pr_rows.append({"class_id": int(cls_id), "class_name": names_map.get(int(cls_id), str(cls_id)), "ap": ap50_val})
        for xr, pv in zip(x.tolist(), prec_values[idx].tolist() if idx < len(prec_values) else []):
            per_class_rows.append(
                {
                    "class_id": int(cls_id),
                    "class_name": names_map.get(int(cls_id), str(cls_id)),
                    "recall": float(xr),
                    "precision": float(pv),
                    "ap": ap50_val,
                }
            )

    mp = float(np.mean(p_cls)) if len(p_cls) else 0.0
    mr = float(np.mean(r_cls)) if len(r_cls) else 0.0
    mf1 = float(np.mean(f1_cls)) if len(f1_cls) else 0.0
    map50 = float(ap[:, 0].mean()) if isinstance(ap, np.ndarray) and ap.size else 0.0
    map5095 = float(ap.mean()) if isinstance(ap, np.ndarray) and ap.size else 0.0

    # Curves on confidence grid (1000 points) like Ultralytics.
    p2d = np.asarray(p_curve, dtype=np.float32) if isinstance(p_curve, np.ndarray) else np.zeros((0, 1000), dtype=np.float32)
    r2d = np.asarray(r_curve, dtype=np.float32) if isinstance(r_curve, np.ndarray) else np.zeros((0, 1000), dtype=np.float32)
    f1_2d = np.asarray(f1_curve, dtype=np.float32) if isinstance(f1_curve, np.ndarray) else np.zeros((0, 1000), dtype=np.float32)
    thresholds = np.asarray(x, dtype=np.float32) if isinstance(x, np.ndarray) else np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    return {
        "pr_df": pd.DataFrame(pr_rows),
        "pr_per_class_df": pd.DataFrame(per_class_rows),
        "map50": map50,
        "map5095": map5095,
        "box_p": mp,
        "box_r": mr,
        "box_f1": mf1,
        "thresholds": thresholds,
        "p2d": p2d,
        "r2d": r2d,
        "f1_2d": f1_2d,
        "pr_recall": np.asarray(x, dtype=np.float32) if isinstance(x, np.ndarray) else np.linspace(0.0, 1.0, 1000, dtype=np.float32),
        "pr_precision_mean": np.asarray(prec_values, dtype=np.float32).mean(axis=0)
        if isinstance(prec_values, np.ndarray) and prec_values.size
        else np.zeros((1000,), dtype=np.float32),
    }


def _compute_global_stats(preds: list[_Pred], gt_rows: list[_Gt], names: list[str], iou_thr: float) -> dict[str, float]:
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for class_id, _ in enumerate(names):
        class_preds = [p for p in preds if p.cls_id == class_id]
        class_gts = [g for g in gt_rows if g.cls_id == class_id]
        tp, fp, total_gt = _match_class_predictions(class_preds, class_gts, iou_thr)
        total_tp += float(tp.sum())
        total_fp += float(fp.sum())
        total_fn += float(max(total_gt - tp.sum(), 0.0))
    precision = total_tp / max(total_tp + total_fp, 1e-9)
    recall = total_tp / max(total_tp + total_fn, 1e-9)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {"Box-P": precision, "Box-R": recall, "Box-F1": f1}


def _compute_map5095(preds: list[_Pred], gt_rows: list[_Gt], names: list[str]) -> float:
    thresholds = np.arange(0.5, 1.0, 0.05)
    values: list[float] = []
    for thr in thresholds:
        ap_values: list[float] = []
        for class_id, _ in enumerate(names):
            class_preds = [p for p in preds if p.cls_id == class_id]
            class_gts = [g for g in gt_rows if g.cls_id == class_id]
            if not class_gts:
                continue
            tp, fp, total_gt = _match_class_predictions(class_preds, class_gts, float(thr))
            if tp.size == 0:
                continue
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / max(float(total_gt), 1.0)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
            ap_values.append(_compute_ap(recall, precision))
        if ap_values:
            values.append(float(np.mean(ap_values)))
    return float(np.mean(values)) if values else 0.0


def _compute_threshold_curves(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
    iou_thr: float,
    points: int = 51,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.0, 1.0, points)
    p_rows: list[list[float]] = []
    r_rows: list[list[float]] = []
    for class_id, _ in enumerate(names):
        class_preds_all = [p for p in preds if p.cls_id == class_id]
        class_gts = [g for g in gt_rows if g.cls_id == class_id]
        if not class_gts:
            continue
        p_vals: list[float] = []
        r_vals: list[float] = []
        for thr in thresholds:
            class_preds = [p for p in class_preds_all if p.conf >= float(thr)]
            tp, fp, total_gt = _match_class_predictions(class_preds, class_gts, iou_thr)
            tp_sum = float(tp.sum())
            fp_sum = float(fp.sum())
            fn_sum = float(max(total_gt - tp_sum, 0.0))
            precision = tp_sum / max(tp_sum + fp_sum, 1e-9)
            recall = tp_sum / max(tp_sum + fn_sum, 1e-9)
            p_vals.append(precision)
            r_vals.append(recall)
        p_rows.append(p_vals)
        r_rows.append(r_vals)
    if not p_rows:
        zero = np.zeros((1, points), dtype=np.float32)
        return thresholds, zero, zero
    return thresholds, np.asarray(p_rows, dtype=np.float32), np.asarray(r_rows, dtype=np.float32)


def _save_curve_plot(x: np.ndarray, y: np.ndarray, out_path: str, title: str, x_label: str, y_label: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _save_confusion_matrix(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
    conf_thr: float,
    iou_thr: float,
    out_path: str,
    normalized_out_path: str,
) -> None:
    n = len(names)
    matrix = np.zeros((n + 1, n + 1), dtype=np.float32)
    by_image_gt: dict[str, list[_Gt]] = {}
    by_image_pred: dict[str, list[_Pred]] = {}
    for gt in gt_rows:
        by_image_gt.setdefault(gt.image_path, []).append(gt)
    for pred in preds:
        if pred.conf >= conf_thr:
            by_image_pred.setdefault(pred.image_path, []).append(pred)
    all_images = sorted(set(by_image_gt) | set(by_image_pred))
    for image_path in all_images:
        image_gts = by_image_gt.get(image_path, [])
        image_preds = sorted(by_image_pred.get(image_path, []), key=lambda x: x.conf, reverse=True)
        used_gt: set[int] = set()
        for pred in image_preds:
            pred_box = _Box(pred.cls_id, pred.conf, pred.x1, pred.y1, pred.x2, pred.y2)
            best_iou = 0.0
            best_gt_idx: int | None = None
            for gt_idx, gt in enumerate(image_gts):
                if gt_idx in used_gt:
                    continue
                gt_box = _Box(gt.cls_id, None, gt.x1, gt.y1, gt.x2, gt.y2)
                cur_iou = _iou(pred_box, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_gt_idx = gt_idx
            if best_gt_idx is not None and best_iou >= iou_thr:
                gt = image_gts[best_gt_idx]
                matrix[int(gt.cls_id), int(pred.cls_id)] += 1.0
                used_gt.add(best_gt_idx)
            else:
                matrix[n, int(pred.cls_id)] += 1.0
        for gt_idx, gt in enumerate(image_gts):
            if gt_idx not in used_gt:
                matrix[int(gt.cls_id), n] += 1.0
    labels = names + ["background"]
    for path, current in ((out_path, matrix), (normalized_out_path, matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0))):
        plt.figure(figsize=(8, 6))
        plt.imshow(current, cmap="Blues")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.tight_layout()
        plt.colorbar()
        plt.savefig(path, dpi=220)
        plt.close()


def _write_test_args_yaml(
    test_dir: str,
    *,
    backend: str,
    format_name: str,
    weights_path: str,
    data_yaml_path: str,
    imgsz: int | None,
    conf: float | None,
    iou: float | None,
    batch: int | None,
) -> None:
    payload = {
        "backend": backend,
        "format": format_name,
        "weights": weights_path,
        "data": data_yaml_path,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "batch": batch,
    }
    with open(os.path.join(test_dir, "args.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _build_confidence_metrics_stub(names: list[str], thresholds: np.ndarray, p2d: np.ndarray, r2d: np.ndarray) -> Any:
    class _BoxMetrics:
        def __init__(self, curves_results: list[tuple[Any, Any, Any, Any]]) -> None:
            self.curves_results = curves_results

    class _Metrics:
        def __init__(self) -> None:
            self.names = {idx: name for idx, name in enumerate(names)}
            f1 = np.where((p2d + r2d) > 0.0, 2.0 * p2d * r2d / np.maximum(p2d + r2d, 1e-9), 0.0)
            self.box = _BoxMetrics(
                [
                    (thresholds, f1, "Confidence", "F1"),
                    (thresholds, p2d, "Confidence", "Precision"),
                    (thresholds, r2d, "Confidence", "Recall"),
                ]
            )

    return _Metrics()


def _write_native_eval_artifacts(
    *,
    root_dir: str,
    format_name: str,
    backend_name: str,
    weights_path: str,
    data_yaml_path: str,
    split: str,
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
    conf_thr: float,
    iou_thr: float,
    imgsz: int | None,
    batch: int | None,
) -> dict[str, Any]:
    test_dir = format_test_dir(root_dir, format_name)
    if split == "test":
        os.makedirs(test_dir, exist_ok=True)
    metrics_payload = _compute_ultralytics_style_payload(preds, gt_rows, names)
    pr_df = metrics_payload["pr_df"]
    pr_per_class_df = metrics_payload["pr_per_class_df"]
    map50 = float(metrics_payload["map50"])
    map5095 = float(metrics_payload["map5095"])
    metrics_df = pd.DataFrame(
        [
            {
                "mAP50-95": map5095,
                "mAP50": map50,
                "Box-F1": float(metrics_payload["box_f1"]),
                "Box-P": float(metrics_payload["box_p"]),
                "Box-R": float(metrics_payload["box_r"]),
            }
        ]
    )
    metrics_df.to_csv(format_metrics_path_for_split(root_dir, split, format_name), index=False, encoding="utf-8")
    if split == "test":
        if len(pr_per_class_df) > 0:
            pr_per_class_df.to_csv(os.path.join(test_dir, "pr_per_class.csv"), index=False, encoding="utf-8")
        recall_grid = np.asarray(metrics_payload["pr_recall"], dtype=np.float32)
        mean_precision = np.asarray(metrics_payload["pr_precision_mean"], dtype=np.float32)
        pd.DataFrame({"recall": recall_grid, "precision": mean_precision}).to_csv(
            os.path.join(test_dir, "pr.csv"),
            index=False,
            encoding="utf-8",
        )
        _save_curve_plot(recall_grid, mean_precision, os.path.join(test_dir, "BoxPR_curve.png"), "PR curve", "Recall", "Precision")
    thresholds = np.asarray(metrics_payload["thresholds"], dtype=np.float32)
    p2d = np.asarray(metrics_payload["p2d"], dtype=np.float32)
    r2d = np.asarray(metrics_payload["r2d"], dtype=np.float32)
    f1_2d = np.asarray(metrics_payload["f1_2d"], dtype=np.float32)
    p_mean = np.mean(p2d, axis=0) if p2d.size else np.zeros_like(thresholds)
    r_mean = np.mean(r2d, axis=0) if r2d.size else np.zeros_like(thresholds)
    f1_mean = np.mean(f1_2d, axis=0) if f1_2d.size else np.zeros_like(thresholds)
    if split == "test":
        _save_curve_plot(thresholds, f1_mean, os.path.join(test_dir, "BoxF1_curve.png"), "F1 vs confidence", "Confidence", "F1")
        _save_curve_plot(thresholds, p_mean, os.path.join(test_dir, "BoxP_curve.png"), "Precision vs confidence", "Confidence", "Precision")
        _save_curve_plot(thresholds, r_mean, os.path.join(test_dir, "BoxR_curve.png"), "Recall vs confidence", "Confidence", "Recall")
        _save_confusion_matrix(
            preds,
            gt_rows,
            names,
            conf_thr,
            iou_thr,
            os.path.join(test_dir, "confusion_matrix.png"),
            os.path.join(test_dir, "confusion_matrix_normalized.png"),
        )
        _write_test_args_yaml(
            test_dir,
            backend=backend_name,
            format_name=format_name,
            weights_path=weights_path,
            data_yaml_path=data_yaml_path,
            imgsz=imgsz,
            conf=conf_thr,
            iou=iou_thr,
            batch=batch,
        )
    metrics_stub = _build_confidence_metrics_stub(names, thresholds, p2d, r2d)
    split_payload = compute_confidence_recommendations(metrics_stub, split=split)
    write_recommendation_file(format_recommendation_path(root_dir, split, format_name), split_payload)
    return {"imgsz": imgsz, "conf": conf_thr, "iou": iou_thr, "batch": batch}


def _infer_with_onnx_session(session: Any, image_path: str, input_hw: tuple[int, int], conf_thr: float, iou_thr: float, names: list[str]) -> list[_Pred]:
    input_name = str(session.get_inputs()[0].name)
    tensor, orig_hw, gain, pad = _preprocess_image(image_path, input_hw)
    outputs = session.run(None, {input_name: tensor})
    raw = _select_output_tensor([np.asarray(x) for x in outputs])
    return _decode_onnx_predictions(
        raw,
        image_path=image_path,
        names=names,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        orig_hw=orig_hw,
        gain=gain,
        pad=pad,
    )


def _is_onnx_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("cuda" in msg and "out of memory" in msg) or "cudamalloc" in msg


def _release_cuda_memory_best_effort() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _build_onnx_session_with_retry(ort: Any, weights_path: str, providers: list[str]) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                print(f"[WARN] onnx: retrying session initialization ({attempt}/3).")
            return ort.InferenceSession(weights_path, providers=providers or None)
        except Exception as exc:
            last_error = exc
            if _is_onnx_cuda_oom_error(exc):
                sleep_s = 0.5 * (2 ** (attempt - 1))
                print(f"[WARN] onnx: session init failed due to CUDA OOM (attempt {attempt}/3).")
                _release_cuda_memory_best_effort()
                time.sleep(sleep_s)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("onnxruntime session init failed for unknown reason")


def _run_onnx_split_with_retry(
    *,
    split_name: str,
    image_paths: list[str],
    session: Any,
    input_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    names: list[str],
    format_name: str,
    weights_path: str,
) -> list[_Pred]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        preds: list[_Pred] = []
        try:
            print(f"[INFO] {format_name}: running native {split_name} on {len(image_paths)} images with {weights_path}")
            for image_path in tqdm(image_paths, desc=f"{format_name}:{split_name}", unit="img", file=sys.stdout):
                preds.extend(_infer_with_onnx_session(session, image_path, input_hw, conf_thr, iou_thr, names))
            print(f"[INFO] {format_name}: native {split_name} completed ({len(image_paths)}/{len(image_paths)} images).")
            return preds
        except Exception as exc:
            last_error = exc
            if _is_onnx_cuda_oom_error(exc) and attempt < 3:
                sleep_s = 0.5 * (2 ** (attempt - 1))
                print(
                    f"[WARN] {format_name}: CUDA OOM during {split_name} on attempt {attempt}/3. "
                    "Retrying split from start."
                )
                _release_cuda_memory_best_effort()
                time.sleep(sleep_s)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"onnxruntime {split_name} inference failed for unknown reason")


def _infer_with_pt_model(model: Any, image_path: str, input_hw: tuple[int, int], conf_thr: float, iou_thr: float) -> list[_Pred]:
    out: list[_Pred] = []
    try:
        results = model.predict(
            source=image_path,
            imgsz=int(input_hw[0]),
            conf=float(conf_thr),
            iou=float(iou_thr),
            verbose=False,
        )
        if not results:
            return out
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for b, c, k in zip(xyxy, confs, clss):
            out.append(
                _Pred(
                    image_path=image_path,
                    cls_id=int(k),
                    conf=float(c),
                    x1=float(b[0]),
                    y1=float(b[1]),
                    x2=float(b[2]),
                    y2=float(b[3]),
                )
            )
    except Exception:
        return out
    return out


def _trt_volume(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _cuda_check(status: Any, op: str) -> Any:
    code = status[0] if isinstance(status, tuple) else status
    if int(code) != 0:
        raise RuntimeError(f"{op} failed with CUDA status={code}")
    return status


def _infer_with_trt_engine(
    engine_path: str,
    image_path: str,
    input_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    names: list[str],
) -> list[_Pred]:
    import tensorrt as trt  # type: ignore
    from cuda import cudart  # type: ignore

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    tensor, orig_hw, gain, pad = _preprocess_image(image_path, input_hw)
    input_array = np.ascontiguousarray(tensor.astype(np.float32))
    bindings: list[int] = [0] * int(getattr(engine, "num_bindings"))
    device_allocations: list[int] = []
    host_outputs: list[np.ndarray] = []
    try:
        for binding_idx in range(int(engine.num_bindings)):
            is_input = bool(engine.binding_is_input(binding_idx))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding_idx)))
            if is_input:
                shape = tuple(int(x) for x in input_array.shape)
                if any(int(v) < 0 for v in engine.get_binding_shape(binding_idx)):
                    context.set_binding_shape(binding_idx, shape)
                nbytes = int(input_array.nbytes)
                err, ptr = cudart.cudaMalloc(nbytes)
                _cuda_check((err,), "cudaMalloc(input)")
                device_allocations.append(int(ptr))
                _cuda_check(
                    cudart.cudaMemcpy(
                        int(ptr),
                        input_array.ctypes.data,
                        nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy(H2D)",
                )
                bindings[binding_idx] = int(ptr)
            else:
                shape = tuple(int(x) for x in context.get_binding_shape(binding_idx))
                nbytes = int(_trt_volume(shape) * dtype.itemsize)
                host = np.empty(shape, dtype=dtype)
                err, ptr = cudart.cudaMalloc(nbytes)
                _cuda_check((err,), "cudaMalloc(output)")
                device_allocations.append(int(ptr))
                bindings[binding_idx] = int(ptr)
                host_outputs.append(host)
        if not context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 returned False")
        output_idx = 0
        for binding_idx in range(int(engine.num_bindings)):
            if bool(engine.binding_is_input(binding_idx)):
                continue
            host = host_outputs[output_idx]
            output_idx += 1
            _cuda_check(
                cudart.cudaMemcpy(
                    host.ctypes.data,
                    int(bindings[binding_idx]),
                    int(host.nbytes),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ),
                "cudaMemcpy(D2H)",
            )
        raw = _select_output_tensor([np.asarray(x) for x in host_outputs])
    finally:
        for ptr in device_allocations:
            try:
                cudart.cudaFree(int(ptr))
            except Exception:
                pass
    return _decode_onnx_predictions(
        raw,
        image_path=image_path,
        names=names,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        orig_hw=orig_hw,
        gain=gain,
        pad=pad,
    )


def _ensure_confidence_recommendations_for_explicit_artifact(
    *,
    model: Any,
    primary_test_result: Any,
    root_dir: str,
    format_name: str,
    data_yaml: str,
    imgsz: int | None,
    val_conf: float | None,
    val_iou: float | None,
    val_batch: int | None,
    beta_recall: float,
    beta_precision: float,
    fallback_confidence: float,
) -> None:
    test_path = format_recommendation_path(root_dir, "test", format_name)
    val_path = format_recommendation_path(root_dir, "val", format_name)
    test_payload = compute_confidence_recommendations(
        primary_test_result,
        split="test",
        beta_recall=float(beta_recall),
        beta_precision=float(beta_precision),
        fallback_confidence=float(fallback_confidence),
    )
    write_recommendation_file(test_path, test_payload)

    val_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": "val",
        "plots": False,
        "save": False,
        "verbose": False,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)
    try:
        val_result = model.val(**val_kwargs)
        try:
            val_csv = format_metrics_path_for_split(root_dir, "val", format_name)
            with open(val_csv, "w", encoding="utf-8") as f:
                f.write(val_result.to_csv())
        except Exception:
            pass
        val_payload = compute_confidence_recommendations(
            val_result,
            split="val",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )
        write_recommendation_file(val_path, val_payload)
    except Exception as exc:
        write_not_available_recommendations(
            model_dir=root_dir,
            split=f"val{'' if format_name == 'pt' else '_' + format_name}",
            reason=f"val_split_failed: {exc}",
            beta_recall=float(beta_recall),
            beta_precision=float(beta_precision),
            fallback_confidence=float(fallback_confidence),
        )


def run_ultralytics_backend(
    *,
    root_dir: str,
    weights_path: str,
    dataset_yaml_path: str,
    format_name: str,
    imgsz: int | None = None,
    val_conf: float | None = None,
    val_iou: float | None = None,
    val_batch: int | None = None,
    conf_rec_disable: bool = False,
    conf_rec_beta_recall: float = 2.0,
    conf_rec_beta_precision: float = 0.5,
    conf_rec_fallback: float = 0.25,
) -> BackendRunResult:
    test_start_time = datetime.now()
    model = YOLO(weights_path)
    val_kwargs = {
        "data": dataset_yaml_path,
        "split": "test",
        "project": root_dir,
        "name": os.path.basename(format_test_dir(root_dir, format_name)),
        "exist_ok": True,
    }
    if imgsz is not None:
        val_kwargs["imgsz"] = imgsz
    if val_conf is not None:
        val_kwargs["conf"] = val_conf
    if val_iou is not None:
        val_kwargs["iou"] = val_iou
    if val_batch is not None:
        val_kwargs["batch"] = int(val_batch)
    try:
        result = model.val(**val_kwargs)
        _save_metrics_csv_for_format(result, root_dir, format_name)
        if not conf_rec_disable:
            _ensure_confidence_recommendations_for_explicit_artifact(
                model=model,
                primary_test_result=result,
                root_dir=root_dir,
                format_name=format_name,
                data_yaml=dataset_yaml_path,
                imgsz=imgsz,
                val_conf=val_conf,
                val_iou=val_iou,
                val_batch=val_batch,
                beta_recall=conf_rec_beta_recall,
                beta_precision=conf_rec_beta_precision,
                fallback_confidence=conf_rec_fallback,
            )
        test_end_time = datetime.now()
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            status="ok",
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=True,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
            inference={"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": val_batch},
            target_path=weights_path,
        )
    except Exception as exc:
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            status="failed",
            error=str(exc),
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=False,
            test_start_time=test_start_time,
            test_end_time=datetime.now(),
            inference={"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": val_batch},
            target_path=weights_path,
            error=str(exc),
        )


def run_native_format_backend(
    *,
    root_dir: str,
    weights_path: str,
    dataset_yaml_path: str,
    format_name: str,
    imgsz: int | None = None,
    val_conf: float | None = None,
    val_iou: float | None = None,
    val_batch: int | None = None,
) -> BackendRunResult:
    backend_name = "onnxruntime" if format_name == "onnx" else ("unified_pt" if format_name == "pt_uni" else "tensorrt")
    try:
        if format_name == "onnx":
            import onnxruntime as ort  # type: ignore
            providers = []
            available = list(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            try:
                session = _build_onnx_session_with_retry(ort, weights_path, providers)
            except Exception as primary_exc:
                cpu_only = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else None
                if not cpu_only:
                    raise primary_exc
                print("[WARN] onnx: switching to CPUExecutionProvider after repeated initialization failures.")
                session = ort.InferenceSession(weights_path, providers=cpu_only)
            input_meta = session.get_inputs()[0]
            names = _load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _collect_gt(dataset_yaml_path, "test")
            input_hw = _resolve_imgsz_from_onnx(session, imgsz)
            conf_thr = float(val_conf if val_conf is not None else 0.25)
            iou_thr = float(val_iou if val_iou is not None else 0.45)
            preds = _run_onnx_split_with_retry(
                split_name="test",
                image_paths=image_paths,
                session=session,
                input_hw=input_hw,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                names=names,
                format_name=format_name,
                weights_path=weights_path,
            )
            inference = _write_native_eval_artifacts(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                weights_path=weights_path,
                data_yaml_path=dataset_yaml_path,
                split="test",
                preds=preds,
                gt_rows=gt_rows,
                names=names,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                imgsz=input_hw[0],
                batch=val_batch,
            )
            try:
                gt_rows_val, _bgv, image_paths_val = _collect_gt(dataset_yaml_path, "val")
                preds_val = _run_onnx_split_with_retry(
                    split_name="val",
                    image_paths=image_paths_val,
                    session=session,
                    input_hw=input_hw,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    names=names,
                    format_name=format_name,
                    weights_path=weights_path,
                )
                _write_native_eval_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="val",
                    preds=preds_val,
                    gt_rows=gt_rows_val,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                )
            except Exception:
                pass
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                status="ok",
            )
            return BackendRunResult(
                format=format_name,
                backend=backend_name,
                success=True,
                test_start_time=datetime.now(),
                test_end_time=datetime.now(),
                inference=inference,
                target_path=weights_path,
            )
        elif format_name in {"engine", "trt", "pt_uni"}:
            names = _load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _collect_gt(dataset_yaml_path, "test")
            conf_thr = float(val_conf if val_conf is not None else 0.25)
            iou_thr = float(val_iou if val_iou is not None else 0.45)
            input_hw = (int(imgsz or 640), int(imgsz or 640))
            pt_model = YOLO(weights_path) if format_name == "pt_uni" else None
            preds: list[_Pred] = []
            total_images = len(image_paths)
            print(f"[INFO] {format_name}: running native test on {total_images} images with {weights_path}")
            for image_path in tqdm(image_paths, desc=f"{format_name}:test", unit="img", file=sys.stdout):
                if format_name == "pt_uni" and pt_model is not None:
                    preds.extend(_infer_with_pt_model(pt_model, image_path, input_hw, conf_thr, iou_thr))
                else:
                    preds.extend(_infer_with_trt_engine(weights_path, image_path, input_hw, conf_thr, iou_thr, names))
            print(f"[INFO] {format_name}: native test completed ({total_images}/{total_images} images).")
            inference = _write_native_eval_artifacts(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                weights_path=weights_path,
                data_yaml_path=dataset_yaml_path,
                split="test",
                preds=preds,
                gt_rows=gt_rows,
                names=names,
                conf_thr=conf_thr,
                iou_thr=iou_thr,
                imgsz=input_hw[0],
                batch=val_batch,
            )
            try:
                gt_rows_val, _bgv, image_paths_val = _collect_gt(dataset_yaml_path, "val")
                preds_val: list[_Pred] = []
                print(f"[INFO] {format_name}: running native val on {len(image_paths_val)} images with {weights_path}")
                for image_path in tqdm(image_paths_val, desc=f"{format_name}:val", unit="img", file=sys.stdout):
                    if format_name == "pt_uni" and pt_model is not None:
                        preds_val.extend(_infer_with_pt_model(pt_model, image_path, input_hw, conf_thr, iou_thr))
                    else:
                        preds_val.extend(_infer_with_trt_engine(weights_path, image_path, input_hw, conf_thr, iou_thr, names))
                print(f"[INFO] {format_name}: native val completed ({len(image_paths_val)}/{len(image_paths_val)} images).")
                _write_native_eval_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="val",
                    preds=preds_val,
                    gt_rows=gt_rows_val,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                )
            except Exception:
                pass
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                status="ok",
            )
            return BackendRunResult(
                format=format_name,
                backend=backend_name,
                success=True,
                test_start_time=datetime.now(),
                test_end_time=datetime.now(),
                inference=inference,
                target_path=weights_path,
            )
        raise RuntimeError(f"Unsupported native backend format: {format_name}")
    except Exception as exc:
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend=backend_name,
            status="unavailable",
            error=str(exc),
        )
        return BackendRunResult(
            format=format_name,
            backend=backend_name,
            success=False,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": val_batch},
            target_path=weights_path,
            error=str(exc),
        )
