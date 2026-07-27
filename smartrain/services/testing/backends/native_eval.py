"""Native-format test eval geometry and Ultralytics-style metrics (no workflow imports)."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from datetime import datetime
import yaml
from ultralytics.utils import nms as ultralytics_nms
from ultralytics.utils.metrics import ap_per_class

from smartrain.core.runtime.run_artifacts import ensure_run_layout, read_model_sidecar_metadata
from smartrain.core.testing.artifact_paths import (
    format_metrics_path_for_write,
    format_test_dir_for_write,
)
from smartrain.services.testing.unified_metrics_adapter import collect_ultralytics_style_gt


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
class PerfCollector:
    warmup_images: int = 5

    def __post_init__(self) -> None:
        self._per_image_ns: list[int] = []
        self._stages_ns: dict[str, list[int]] = {}
        self._started_ns = time.perf_counter_ns()
        self._ended_ns: int | None = None

    def record_total_image(self, dt_ns: int) -> None:
        self._per_image_ns.append(int(max(0, dt_ns)))

    def record_stage(self, stage: str, dt_ns: int) -> None:
        key = str(stage).strip()
        if not key:
            return
        self._stages_ns.setdefault(key, []).append(int(max(0, dt_ns)))

    def finish(self) -> None:
        if self._ended_ns is None:
            self._ended_ns = time.perf_counter_ns()

    @staticmethod
    def _stats_ms(values_ns: list[int]) -> dict[str, float | int | None]:
        if not values_ns:
            return {
                "count": 0,
                "mean": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "min": None,
                "max": None,
                "std": None,
            }
        arr = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }

    def to_payload(self) -> dict[str, Any]:
        self.finish()
        ended = self._ended_ns if self._ended_ns is not None else time.perf_counter_ns()
        duration_s = max(0.0, float(ended - self._started_ns) / 1_000_000_000.0)
        all_stats = self._stats_ms(self._per_image_ns)
        steady_values = self._per_image_ns[int(max(0, self.warmup_images)) :]
        steady_stats = self._stats_ms(steady_values)
        throughput = (
            float(len(steady_values) / duration_s) if duration_s > 0.0 and len(steady_values) > 0 else 0.0
        )
        stages: dict[str, dict[str, float | int | None]] = {}
        for stage, vals in self._stages_ns.items():
            stages[stage] = self._stats_ms(vals)
        return {
            "images_total": int(len(self._per_image_ns)),
            "warmup_images": int(max(0, self.warmup_images)),
            "duration_s": duration_s,
            "throughput_img_s": throughput,
            "latency_ms": {"all": all_stats, "steady": steady_stats},
            "breakdown_ms": stages,
        }


def _write_perf_artifact(root_dir: str, format_name: str, target_path: str, performance: dict[str, Any]) -> str:
    ensure_run_layout(root_dir)
    test_dir = format_test_dir_for_write(root_dir, format_name)
    os.makedirs(test_dir, exist_ok=True)
    stem = Path(target_path).stem if target_path else format_name
    out_path = os.path.join(test_dir, f"perf_{stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(performance, f, ensure_ascii=False, indent=2)
    return out_path


def _collect_test_system_profile(
    *,
    root_dir: str,
    format_name: str,
    backend_name: str,
    runtime_provider: str | None = None,
    runtime_provider_actual: str | None = None,
    runtime_device: str | None = None,
) -> dict[str, Any]:
    try:
        from smartrain.core.runtime.system_profile import collect_system_profile

        payload = collect_system_profile(root_dir)
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["runtime"] = {
        "stage": "test",
        "format": str(format_name),
        "backend": str(backend_name),
        "provider": str(runtime_provider) if runtime_provider else None,
        "provider_actual": str(runtime_provider_actual) if runtime_provider_actual else None,
        "device": str(runtime_device) if runtime_device else None,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    return payload


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
    ensure_run_layout(root_dir)
    csv_file = format_metrics_path_for_write(root_dir, format_name)
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
    requested = int(imgsz) if isinstance(imgsz, (int, float)) and int(imgsz) > 0 else None
    try:
        shape = list(session.get_inputs()[0].shape)
        if len(shape) == 4:
            h = int(shape[2]) if isinstance(shape[2], (int, float)) else 640
            w = int(shape[3]) if isinstance(shape[3], (int, float)) else 640
            model_h, model_w = max(32, h), max(32, w)
            if requested is not None and requested != model_h and model_h == model_w:
                print(
                    f"[WARN] onnx imgsz mismatch: requested={requested}, model={model_h}. "
                    "Auto-aligning to model input size."
                )
            if requested is not None and requested == model_h == model_w:
                return requested, requested
            return model_h, model_w
    except Exception:
        pass
    if requested is not None:
        return requested, requested
    return 640, 640


def _resolve_input_hw_from_native_artifact(weights_path: str, requested_imgsz: int | None) -> tuple[int, int]:
    req = int(requested_imgsz) if isinstance(requested_imgsz, (int, float)) and int(requested_imgsz) > 0 else None
    artifact_imgsz: int | None = None
    meta = read_model_sidecar_metadata(weights_path) or {}
    if isinstance(meta, dict):
        params = meta.get("params")
        if isinstance(params, dict):
            for key in ("imgsz", "image_size", "img_size"):
                value = params.get(key)
                if isinstance(value, (int, float)) and int(value) > 0:
                    artifact_imgsz = int(value)
                    break
    if artifact_imgsz is None:
        m = re.search(r"_imgsz(\d+)x(\d+)_", Path(weights_path).name)
        if m and m.group(1) == m.group(2):
            artifact_imgsz = int(m.group(1))
    if artifact_imgsz is not None:
        if req is not None and req != artifact_imgsz:
            print(
                "[WARN] native eval imgsz mismatch: "
                f"requested={req}, artifact={artifact_imgsz}. Using artifact size."
            )
        return artifact_imgsz, artifact_imgsz
    if req is not None:
        return req, req
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


def _load_image_rgb(image_path: str) -> np.ndarray:
    im = Image.open(image_path).convert("RGB")
    return np.asarray(im)


def _preprocess_array(arr: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], float, tuple[float, float]]:
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
    cols = int(raw.shape[1]) if raw.ndim == 2 else 0
    if cols == 6:
        boxes_xyxy = raw[:, :4].copy()
        scores = raw[:, 4].copy()
        cls_ids = raw[:, 5].astype(int)
        valid = scores >= float(conf_thr)
        if not np.any(valid):
            return []
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]
        merged = np.concatenate([boxes_xyxy, scores[:, None], cls_ids[:, None]], axis=1).astype(np.float32)
        nms_out = torch.tensor(merged, dtype=torch.float32).unsqueeze(0)
    else:
        boxes = raw[:, :4].copy().astype(np.float32)
        class_logits = raw[:, 4:].copy().astype(np.float32)
        if num_classes > 0 and cols == num_classes + 5:
            obj = class_logits[:, :1]
            cls = class_logits[:, 1:]
            class_logits = np.concatenate([obj, cls], axis=1)
        # Build tensor in BCN format expected by Ultralytics NMS.
        pred = np.concatenate([boxes, class_logits], axis=1).astype(np.float32)  # N x (4 + C or 5 + C)
        nms_out = torch.tensor(pred, dtype=torch.float32).T.unsqueeze(0)
    nms_result = ultralytics_nms.non_max_suppression(
        nms_out,
        conf_thres=float(conf_thr),
        iou_thres=float(iou_thr),
        agnostic=False,
        multi_label=True,
        max_det=300,
        nc=max(num_classes, 0),
    )[0]
    if nms_result is None or nms_result.numel() == 0:
        return []
    kept = nms_result.detach().cpu().numpy()
    boxes_xyxy = kept[:, :4].copy()
    scores = kept[:, 4].copy()
    cls_ids = kept[:, 5].astype(int)
    boxes_xyxy[:, [0, 2]] -= pad[0]
    boxes_xyxy[:, [1, 3]] -= pad[1]
    boxes_xyxy /= max(gain, 1e-9)
    boxes_xyxy = _clip_boxes_xyxy(boxes_xyxy, orig_hw[1], orig_hw[0])
    preds: list[_Pred] = []
    for idx in range(len(boxes_xyxy)):
        b = boxes_xyxy[idx]
        preds.append(
            _Pred(
                image_path=image_path,
                cls_id=int(max(cls_ids[idx], 0)),
                conf=float(max(scores[idx], 0.0)),
                x1=float(b[0]),
                y1=float(b[1]),
                x2=float(b[2]),
                y2=float(b[3]),
            )
        )
    return preds


def _collect_gt(data_yaml_path: str, split: str) -> tuple[list[_Gt], dict[str, list[_Gt]], list[str]]:
    names = _load_names(data_yaml_path)
    gt_payload, image_paths, _issues = collect_ultralytics_style_gt(data_yaml_path, split, names)
    gt_rows: list[_Gt] = [
        _Gt(
            image_path=row.image_path,
            cls_id=int(row.cls_id),
            x1=float(row.x1),
            y1=float(row.y1),
            x2=float(row.x2),
            y2=float(row.y2),
        )
        for row in gt_payload
    ]
    by_image: dict[str, list[_Gt]] = {}
    for gt in gt_rows:
        by_image.setdefault(gt.image_path, []).append(gt)
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
    deep_diagnostics: bool = False,
    image_paths: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    by_image_gt: dict[str, list[_Gt]] = {}
    by_image_pred: dict[str, list[_Pred]] = {}
    for gt in gt_rows:
        by_image_gt.setdefault(gt.image_path, []).append(gt)
    for pred in preds:
        by_image_pred.setdefault(pred.image_path, []).append(pred)

    all_images = sorted((set(image_paths) if image_paths else set()) | set(by_image_gt) | set(by_image_pred))
    correct_rows: list[np.ndarray] = []
    conf_rows: list[float] = []
    pred_cls_rows: list[int] = []
    target_cls_rows: list[int] = []
    deep_records: list[dict[str, Any]] = []
    best_iou_bins = np.linspace(0.0, 1.0, 11, dtype=np.float32)  # 10 bins

    for image_path in all_images:
        img_gts = by_image_gt.get(image_path, [])
        img_preds = by_image_pred.get(image_path, [])
        if img_gts:
            target_cls_rows.extend([int(g.cls_id) for g in img_gts])
        if not img_preds:
            if deep_diagnostics:
                deep_records.append(
                    {
                        "image_path": image_path,
                        "n_gts": int(len(img_gts)),
                        "n_preds": 0,
                        "tp_counts_by_iou": [0 for _ in iouv.tolist()],
                        "fp_counts_by_iou": [0 for _ in iouv.tolist()],
                        "best_iou_bins": best_iou_bins.tolist(),
                        "best_iou_tp_hist_by_iou": [[0 for _ in range(len(best_iou_bins) - 1)] for _ in iouv.tolist()],
                        "best_iou_fp_hist_by_iou": [[0 for _ in range(len(best_iou_bins) - 1)] for _ in iouv.tolist()],
                    }
                )
            continue

        pred_boxes = np.asarray([[p.x1, p.y1, p.x2, p.y2] for p in img_preds], dtype=np.float32)
        pred_cls = np.asarray([int(p.cls_id) for p in img_preds], dtype=np.int32)
        pred_conf = np.asarray([float(p.conf) for p in img_preds], dtype=np.float32)
        gt_boxes = np.asarray([[g.x1, g.y1, g.x2, g.y2] for g in img_gts], dtype=np.float32) if img_gts else np.zeros((0, 4), dtype=np.float32)
        gt_cls = np.asarray([int(g.cls_id) for g in img_gts], dtype=np.int32) if img_gts else np.zeros((0,), dtype=np.int32)

        correct = np.zeros((len(img_preds), len(iouv)), dtype=bool)
        best_iou_per_pred = np.zeros((len(img_preds),), dtype=np.float32)
        if len(img_gts):
            iou_mat = np.zeros((len(img_gts), len(img_preds)), dtype=np.float32)
            for gi in range(len(img_gts)):
                iou_mat[gi, :] = _box_iou_np(gt_boxes[gi], pred_boxes)
            class_mask = gt_cls[:, None] == pred_cls[None, :]
            iou_mat = iou_mat * class_mask
            if deep_diagnostics and iou_mat.size:
                # “Best IoU per pred” is used for localization diagnostics.
                best_iou_per_pred = iou_mat.max(axis=0) if iou_mat.shape[1] else np.zeros((0,), dtype=np.float32)
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
        if deep_diagnostics:
            tp_counts_by_iou = [int(correct[:, ti].sum()) for ti in range(correct.shape[1])]
            fp_counts_by_iou = [int(len(img_preds) - tp_counts_by_iou[ti]) for ti in range(len(tp_counts_by_iou))]
            hist_tp_by_iou: list[list[int]] = []
            hist_fp_by_iou: list[list[int]] = []
            for ti in range(correct.shape[1]):
                tp_mask = correct[:, ti]
                fp_mask = ~tp_mask
                tp_vals = best_iou_per_pred[tp_mask]
                fp_vals = best_iou_per_pred[fp_mask]
                tp_hist = np.histogram(tp_vals, bins=best_iou_bins, range=(0.0, 1.0))[0].astype(int).tolist()
                fp_hist = np.histogram(fp_vals, bins=best_iou_bins, range=(0.0, 1.0))[0].astype(int).tolist()
                hist_tp_by_iou.append(tp_hist)
                hist_fp_by_iou.append(fp_hist)
            deep_records.append(
                {
                    "image_path": image_path,
                    "n_gts": int(len(img_gts)),
                    "n_preds": int(len(img_preds)),
                    "tp_counts_by_iou": tp_counts_by_iou,
                    "fp_counts_by_iou": fp_counts_by_iou,
                    "best_iou_bins": best_iou_bins.tolist(),
                    "best_iou_tp_hist_by_iou": hist_tp_by_iou,
                    "best_iou_fp_hist_by_iou": hist_fp_by_iou,
                }
            )

    tp = np.concatenate(correct_rows, axis=0) if correct_rows else np.zeros((0, len(iouv)), dtype=bool)
    conf = np.asarray(conf_rows, dtype=np.float32) if conf_rows else np.zeros((0,), dtype=np.float32)
    pred_cls = np.asarray(pred_cls_rows, dtype=np.int32) if pred_cls_rows else np.zeros((0,), dtype=np.int32)
    target_cls = np.asarray(target_cls_rows, dtype=np.int32) if target_cls_rows else np.zeros((0,), dtype=np.int32)
    if deep_diagnostics:
        return tp, conf, pred_cls, target_cls, deep_records
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


def _compute_ultralytics_style_deep_payload(
    preds: list[_Pred],
    gt_rows: list[_Gt],
    names: list[str],
    image_paths: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    iouv = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    tp, conf, pred_cls, target_cls, deep_records = _build_ultralytics_style_stats(
        preds, gt_rows, iouv, deep_diagnostics=True, image_paths=image_paths
    )

    names_map = {idx: name for idx, name in enumerate(names)}
    (
        _tp_cnt,
        _fp_cnt,
        _p_cls,
        _r_cls,
        _f1_cls,
        ap,
        unique_classes,
        _p_curve,
        _r_curve,
        _f1_curve,
        x,
        _prec_values,
    ) = ap_per_class(
        tp=tp,
        conf=conf,
        pred_cls=pred_cls,
        target_cls=target_cls,
        plot=False,
        names=names_map,
    )

    if isinstance(ap, np.ndarray) and ap.ndim == 2 and ap.size:
        ap_mean_by_iou = ap.mean(axis=0).astype(np.float32).tolist()
        map50 = float(ap[:, 0].mean()) if ap.shape[0] else 0.0
        map5095 = float(ap.mean()) if ap.size else 0.0
    else:
        ap_mean_by_iou = [0.0 for _ in iouv.tolist()]
        map50 = 0.0
        map5095 = 0.0

    agg_tp = np.zeros((len(iouv),), dtype=np.int64)
    agg_fp = np.zeros((len(iouv),), dtype=np.int64)
    total_gts = 0
    total_preds = 0
    for rec in deep_records:
        total_gts += int(rec.get("n_gts", 0))
        total_preds += int(rec.get("n_preds", 0))
        tp_counts_by_iou = rec.get("tp_counts_by_iou") or []
        fp_counts_by_iou = rec.get("fp_counts_by_iou") or []
        if isinstance(tp_counts_by_iou, list) and len(tp_counts_by_iou) == len(iouv):
            agg_tp += np.asarray(tp_counts_by_iou, dtype=np.int64)
        if isinstance(fp_counts_by_iou, list) and len(fp_counts_by_iou) == len(iouv):
            agg_fp += np.asarray(fp_counts_by_iou, dtype=np.int64)

    summary: dict[str, Any] = {
        "iou_thresholds": iouv.astype(np.float32).tolist(),
        "total_gts": int(total_gts),
        "total_preds": int(total_preds),
        "tp_counts_by_iou": agg_tp.astype(np.int64).tolist(),
        "fp_counts_by_iou": agg_fp.astype(np.int64).tolist(),
        "map50": map50,
        "map5095": map5095,
        "ap_mean_by_iou": ap_mean_by_iou,
    }
    return summary, deep_records


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

