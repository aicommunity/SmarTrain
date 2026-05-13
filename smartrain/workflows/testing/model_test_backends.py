from __future__ import annotations

import os
import sys
import gc
import time
import tempfile
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import nms as ultralytics_nms
from ultralytics.utils.metrics import ap_per_class

from smartrain.core.training.confidence_recommendation import (
    compute_confidence_recommendations,
    write_not_available_recommendations,
    write_recommendation_file,
)
from smartrain.workflows.testing.model_test_service import (
    format_metrics_path,
    format_metrics_path_for_split,
    format_metrics_path_for_split_write,
    format_metrics_path_for_write,
    format_recommendation_path,
    format_recommendation_path_for_write,
    format_test_dir,
    format_test_dir_for_write,
    persist_target_test_artifacts_state,
)
from smartrain.core.runtime.run_artifacts import ensure_run_layout, read_model_sidecar_metadata
from smartrain.workflows.models import tensorrt_checks as trt_checks
from smartrain.core.runtime.ultralytics_ephemeral import best_effort_prune_runs_detect_near_run, ultralytics_sidecar_dir
from smartrain.workflows.testing.unified_metrics_adapter import collect_ultralytics_style_gt
from smartrain.workflows.testing.unified_validator_core import EvalProvenance, normalize_eval_params


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
        from smartrain.workflows.training.model_training_module import collect_system_profile

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
    inference_source: str,
    gt_source: str,
    nms_profile: str,
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
        "inference_source": inference_source,
        "gt_source": gt_source,
        "nms_profile": nms_profile,
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
    inference_source: str,
    gt_source: str,
    nms_profile: str,
) -> dict[str, Any]:
    ensure_run_layout(root_dir)
    test_dir = format_test_dir_for_write(root_dir, format_name)
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
    metrics_df.to_csv(format_metrics_path_for_split_write(root_dir, split, format_name), index=False, encoding="utf-8")
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
            inference_source=inference_source,
            gt_source=gt_source,
            nms_profile=nms_profile,
        )
    metrics_stub = _build_confidence_metrics_stub(names, thresholds, p2d, r2d)
    split_payload = compute_confidence_recommendations(metrics_stub, split=split)
    write_recommendation_file(format_recommendation_path_for_write(root_dir, split, format_name), split_payload)
    return {
        "imgsz": imgsz,
        "conf": conf_thr,
        "iou": iou_thr,
        "batch": batch,
        "inference_source": inference_source,
        "gt_source": gt_source,
        "nms_profile": nms_profile,
        "mAP50-95": map5095,
        "mAP50": map50,
        "Box-F1": float(metrics_payload["box_f1"]),
        "Box-P": float(metrics_payload["box_p"]),
        "Box-R": float(metrics_payload["box_r"]),
    }


def _write_deep_diagnostics_artifacts(
    *,
    root_dir: str,
    format_name: str,
    backend_name: str,
    weights_path: str,
    data_yaml_path: str,
    split: str,
    preds: list[_Pred],
    gt_rows: list[_Gt],
    image_paths: list[str],
    names: list[str],
    conf_thr: float,
    iou_thr: float,
    imgsz: int | None,
    batch: int | None,
    inference_source: str,
    gt_source: str,
    nms_profile: str,
) -> dict[str, Any]:
    ensure_run_layout(root_dir)
    test_dir = format_test_dir_for_write(root_dir, format_name)
    deep_dir = os.path.join(test_dir, "deep_diagnostics")
    os.makedirs(deep_dir, exist_ok=True)

    deep_summary, deep_records = _compute_ultralytics_style_deep_payload(
        preds=preds,
        gt_rows=gt_rows,
        names=names,
        image_paths=image_paths,
    )
    deep_by_image = {rec.get("image_path"): rec for rec in deep_records if isinstance(rec, dict)}

    by_image_gt: dict[str, list[_Gt]] = {}
    for gt in gt_rows:
        by_image_gt.setdefault(gt.image_path, []).append(gt)

    by_image_pred: dict[str, list[_Pred]] = {}
    for pred in preds:
        by_image_pred.setdefault(pred.image_path, []).append(pred)

    jsonl_path = os.path.join(deep_dir, f"debug_{split}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for image_path in sorted(image_paths):
            rec = deep_by_image.get(image_path)
            tp_counts_by_iou = rec.get("tp_counts_by_iou", [0 for _ in deep_summary.get("iou_thresholds", [])]) if isinstance(rec, dict) else []
            fp_counts_by_iou = rec.get("fp_counts_by_iou", [0 for _ in deep_summary.get("iou_thresholds", [])]) if isinstance(rec, dict) else []
            best_iou_bins = rec.get("best_iou_bins", []) if isinstance(rec, dict) else []
            best_iou_tp_hist_by_iou = rec.get("best_iou_tp_hist_by_iou", []) if isinstance(rec, dict) else []
            best_iou_fp_hist_by_iou = rec.get("best_iou_fp_hist_by_iou", []) if isinstance(rec, dict) else []

            gts = by_image_gt.get(image_path, [])
            image_preds = by_image_pred.get(image_path, [])
            payload = {
                "image_path": image_path,
                "gts": [
                    {"cls_id": int(g.cls_id), "x1": float(g.x1), "y1": float(g.y1), "x2": float(g.x2), "y2": float(g.y2)}
                    for g in gts
                ],
                "preds": [
                    {
                        "cls_id": int(p.cls_id),
                        "conf": float(p.conf),
                        "x1": float(p.x1),
                        "y1": float(p.y1),
                        "x2": float(p.x2),
                        "y2": float(p.y2),
                    }
                    for p in image_preds
                ],
                "matching": {
                    "tp_counts_by_iou": tp_counts_by_iou,
                    "fp_counts_by_iou": fp_counts_by_iou,
                    "best_iou_bins": best_iou_bins,
                    "best_iou_tp_hist_by_iou": best_iou_tp_hist_by_iou,
                    "best_iou_fp_hist_by_iou": best_iou_fp_hist_by_iou,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary_path = os.path.join(deep_dir, f"debug_{split}_summary.json")
    summary_payload: dict[str, Any] = {
        "split": split,
        "format": format_name,
        "backend": backend_name,
        "weights_path": weights_path,
        "data_yaml_path": data_yaml_path,
        "imgsz": imgsz,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "batch": batch,
        "inference_source": inference_source,
        "gt_source": gt_source,
        "nms_profile": nms_profile,
        **deep_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    params_path = os.path.join(deep_dir, "debug_params.json")
    params_payload = {
        "format": format_name,
        "backend": backend_name,
        "weights_path": weights_path,
        "data_yaml_path": data_yaml_path,
        "imgsz": imgsz,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "batch": batch,
        "inference_source": inference_source,
        "gt_source": gt_source,
        "nms_profile": nms_profile,
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_payload, f, ensure_ascii=False, indent=2)

    return params_payload


def _infer_with_onnx_session(
    session: Any,
    image_path: str,
    input_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    names: list[str],
) -> tuple[list[_Pred], dict[str, int]]:
    input_name = str(session.get_inputs()[0].name)
    t_io0 = time.perf_counter_ns()
    arr = _load_image_rgb(image_path)
    t_io1 = time.perf_counter_ns()
    t_pre0 = time.perf_counter_ns()
    tensor, orig_hw, gain, pad = _preprocess_array(arr, input_hw)
    t_pre1 = time.perf_counter_ns()
    t_inf0 = time.perf_counter_ns()
    outputs = session.run(None, {input_name: tensor})
    t_inf1 = time.perf_counter_ns()
    t_dec0 = time.perf_counter_ns()
    raw = _select_output_tensor([np.asarray(x) for x in outputs])
    preds = _decode_onnx_predictions(
        raw,
        image_path=image_path,
        names=names,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        orig_hw=orig_hw,
        gain=gain,
        pad=pad,
    )
    t1 = time.perf_counter_ns()
    runtime_total = int((t_pre1 - t_pre0) + (t_inf1 - t_inf0) + (t1 - t_dec0))
    return preds, {
        "total": runtime_total,
        "io_load": int(t_io1 - t_io0),
        "preprocess": int(t_pre1 - t_pre0),
        "infer": int(t_inf1 - t_inf0),
        "decode_nms": int(t1 - t_dec0),
    }


def _is_onnx_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    # ONNXRuntime CUDA failures are not always "out of memory" text-only;
    # sometimes you get CUBLAS_STATUS_ALLOC_FAILED / cudaMalloc failure.
    if ("cuda" in msg and "out of memory" in msg) or "cudamalloc" in msg or "bfc_arena" in msg:
        return True
    if "cublas" in msg and ("alloc_failed" in msg or "status_alloc_failed" in msg):
        return True
    if "cublas_status_alloc_failed" in msg:
        return True
    return False


def _classify_onnx_error_text(message: str) -> str:
    msg = str(message or "").lower()
    if "timeout" in msg:
        return "timeout"
    if "terminated by signal" in msg:
        return "signal_terminated"
    if "out of memory" in msg or "cudamalloc" in msg or "bfc_arena" in msg:
        return "oom_gpu"
    if "cudaexecutionprovider" in msg and ("unavailable" in msg or "not found" in msg):
        return "provider_unavailable"
    if "cuda" in msg and ("init" in msg or "failed" in msg):
        return "cuda_init_failed"
    if "invalid_argument" in msg and "expected:" in msg and "got:" in msg:
        return "shape_mismatch"
    if "inferencesession" in msg or "session init" in msg:
        return "init_session_failed"
    if "onnxruntimeerror" in msg or "runtime_exception" in msg:
        return "runtime_exception"
    return "unknown"


def _format_onnx_error(code: str, detail: str) -> str:
    return f"[{str(code or 'unknown').strip()}] {str(detail or '').strip()}"


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
    perf_collector: PerfCollector | None = None,
) -> list[_Pred]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        preds: list[_Pred] = []
        try:
            print(f"[INFO] {format_name}: running native {split_name} on {len(image_paths)} images with {weights_path}")
            for image_path in tqdm(image_paths, desc=f"{format_name}:{split_name}", unit="img", file=sys.stdout):
                infer_out = _infer_with_onnx_session(session, image_path, input_hw, conf_thr, iou_thr, names)
                if isinstance(infer_out, tuple) and len(infer_out) == 2:
                    image_preds, perf_ns = infer_out
                else:
                    image_preds, perf_ns = infer_out, {}
                preds.extend(image_preds)
                if perf_collector is not None:
                    perf_collector.record_total_image(int(perf_ns.get("total", 0)))
                    perf_collector.record_stage("preprocess_ms", int(perf_ns.get("preprocess", 0)))
                    perf_collector.record_stage("infer_ms", int(perf_ns.get("infer", 0)))
                    perf_collector.record_stage("decode_nms_ms", int(perf_ns.get("decode_nms", 0)))
                    perf_collector.record_stage("io_load_ms", int(perf_ns.get("io_load", 0)))
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


def _run_onnx_split_in_subprocess(
    *,
    split_name: str,
    image_paths: list[str],
    weights_path: str,
    dataset_yaml_path: str,
    imgsz: int | None,
    conf_thr: float,
    iou_thr: float,
    providers: list[str],
    provider_policy: str = "gpu_preferred",
    timeout_s: int = 1800,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
) -> tuple[list[_Pred], tuple[int, int], dict[str, Any] | None, str | None]:
    request = {
        "weights_path": weights_path,
        "dataset_yaml_path": dataset_yaml_path,
        "split_name": split_name,
        "image_paths": image_paths,
        "imgsz": imgsz,
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "providers": providers,
        "provider_policy": str(provider_policy),
        "max_retries": 3,
        "collect_performance": bool(collect_performance),
        "perf_warmup_images": int(max(0, perf_warmup_images)),
    }
    cmd = [sys.executable, "-m", "smartrain.workflows.testing.model_test_onnx_worker"]
    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            timeout=max(30, int(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(_format_onnx_error("timeout", f"onnx worker timeout during {split_name}: {exc}")) from exc
    stdout_text = (completed.stdout or "").strip()
    payload: dict[str, Any] = {}
    if stdout_text:
        try:
            payload = json.loads(stdout_text)
        except Exception as exc:
            raise RuntimeError(
                _format_onnx_error(
                    "runtime_exception",
                    f"onnx worker returned non-json output during {split_name}. "
                    f"exit={completed.returncode}, stdout={stdout_text[:400]}",
                )
            ) from exc
    if completed.returncode != 0 or not bool(payload.get("ok")):
        worker_error = payload.get("error") if isinstance(payload, dict) else None
        msg_raw = str(worker_error or f"onnx worker failed for {split_name} (exit={completed.returncode})")
        msg = _format_onnx_error(_classify_onnx_error_text(msg_raw), msg_raw)
        raise RuntimeError(msg)
    preds_payload = payload.get("preds") if isinstance(payload, dict) else []
    preds: list[_Pred] = []
    if isinstance(preds_payload, list):
        for row in preds_payload:
            if not isinstance(row, dict):
                continue
            preds.append(
                _Pred(
                    image_path=str(row.get("image_path", "")),
                    cls_id=int(row.get("cls_id", 0)),
                    conf=float(row.get("conf", 0.0)),
                    x1=float(row.get("x1", 0.0)),
                    y1=float(row.get("y1", 0.0)),
                    x2=float(row.get("x2", 0.0)),
                    y2=float(row.get("y2", 0.0)),
                )
            )
    hw = payload.get("input_hw") if isinstance(payload, dict) else None
    perf_payload = payload.get("performance") if isinstance(payload.get("performance"), dict) else None
    provider_used = str(payload.get("provider") or "").strip() if isinstance(payload, dict) else ""
    provider_used = provider_used or None
    if isinstance(hw, list) and len(hw) == 2:
        return preds, (int(hw[0]), int(hw[1])), perf_payload, provider_used
    if isinstance(imgsz, int):
        return preds, (int(imgsz), int(imgsz)), perf_payload, provider_used
    return preds, (640, 640), perf_payload, provider_used


def _infer_with_pt_model(
    model: Any,
    image_path: str,
    input_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    *,
    ultra_predict_project: str | None = None,
) -> tuple[list[_Pred], dict[str, int]]:
    t0 = time.perf_counter_ns()
    out: list[_Pred] = []
    proj = ultra_predict_project or ultralytics_sidecar_dir(
        tempfile.gettempdir(), "smartrain_ultralytics_pt_predict"
    )
    try:
        results = model.predict(
            source=image_path,
            imgsz=int(input_hw[0]),
            conf=float(conf_thr),
            iou=float(iou_thr),
            verbose=False,
            save=False,
            project=proj,
            name="model-test-pt",
            exist_ok=True,
        )
        if not results:
            return out, {"total": int(time.perf_counter_ns() - t0), "infer": int(time.perf_counter_ns() - t0)}
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out, {"total": int(time.perf_counter_ns() - t0), "infer": int(time.perf_counter_ns() - t0)}
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
        return out, {"total": int(time.perf_counter_ns() - t0), "infer": int(time.perf_counter_ns() - t0)}
    t1 = time.perf_counter_ns()
    dt = int(t1 - t0)
    return out, {"total": dt, "infer": dt}


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


def _prepare_trt_runtime(engine_path: str) -> dict[str, Any]:
    import tensorrt as trt  # type: ignore

    try:
        cudart, _import_path = trt_checks.resolve_cuda_runtime_module()
    except Exception as e:
        raise RuntimeError(
            "python CUDA runtime is unavailable. "
            f"Install CUDA Python bindings and verify runtime libraries: {e}"
        ) from e

    t0 = time.perf_counter_ns()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        blob = f.read()

    # Ultralytics .engine can contain a JSON metadata prefix:
    # [4-byte little-endian metadata_len][metadata JSON][raw TRT plan].
    # Native .trt plans start directly with "ftrt".
    payload = blob
    if len(blob) >= 8 and not blob.startswith(b"ftrt"):
        try:
            meta_len = int.from_bytes(blob[:4], "little")
            start = 4 + meta_len
            if 0 < meta_len < len(blob) and start < len(blob) and blob[start : start + 4] == b"ftrt":
                payload = blob[start:]
        except Exception:
            payload = blob

    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError(
            f"Failed to deserialize TensorRT engine: {engine_path}. "
            "This usually means engine/runtime incompatibility (TensorRT/CUDA/GPU mismatch). "
            "Rebuild engine/trt artifacts on this machine with current runtime."
        )
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")
    t1 = time.perf_counter_ns()
    return {
        "trt": trt,
        "cudart": cudart,
        "engine": engine,
        "context": context,
        "init_ns": int(t1 - t0),
    }


def _infer_with_trt_engine(
    runtime_state: dict[str, Any],
    image_path: str,
    input_hw: tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    names: list[str],
) -> tuple[list[_Pred], dict[str, int]]:
    trt = runtime_state.get("trt")
    cudart = runtime_state.get("cudart")
    engine = runtime_state.get("engine")
    context = runtime_state.get("context")
    if trt is None or cudart is None or engine is None or context is None:
        raise RuntimeError("invalid TensorRT runtime state")
    t_io0 = time.perf_counter_ns()
    arr = _load_image_rgb(image_path)
    t_io1 = time.perf_counter_ns()
    t_pre0 = time.perf_counter_ns()
    tensor, orig_hw, gain, pad = _preprocess_array(arr, input_hw)
    t_pre1 = time.perf_counter_ns()
    input_array = np.ascontiguousarray(tensor.astype(np.float32))
    device_allocations: list[int] = []
    host_outputs: list[np.ndarray] = []
    alloc_ns = 0
    h2d_ns = 0
    d2h_ns = 0
    exec_ns = 0
    try:
        output_ptrs: list[int] = []
        if hasattr(engine, "num_io_tensors"):
            # TensorRT 10+ tensor-based API.
            tensor_names = [str(engine.get_tensor_name(i)) for i in range(int(engine.num_io_tensors))]
            for tensor_name in tensor_names:
                mode = engine.get_tensor_mode(tensor_name)
                is_input = bool(mode == trt.TensorIOMode.INPUT)
                dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(tensor_name)))
                if is_input:
                    shape = tuple(int(x) for x in input_array.shape)
                    declared = tuple(int(v) for v in engine.get_tensor_shape(tensor_name))
                    if any(v < 0 for v in declared):
                        context.set_input_shape(tensor_name, shape)
                    nbytes = int(input_array.nbytes)
                    t_alloc0 = time.perf_counter_ns()
                    err, ptr = cudart.cudaMalloc(nbytes)
                    _cuda_check((err,), "cudaMalloc(input)")
                    alloc_ns += int(time.perf_counter_ns() - t_alloc0)
                    device_allocations.append(int(ptr))
                    t_h2d0 = time.perf_counter_ns()
                    _cuda_check(
                        cudart.cudaMemcpy(
                            int(ptr),
                            input_array.ctypes.data,
                            nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        ),
                        "cudaMemcpy(H2D)",
                    )
                    h2d_ns += int(time.perf_counter_ns() - t_h2d0)
                    context.set_tensor_address(tensor_name, int(ptr))
                else:
                    shape = tuple(int(x) for x in context.get_tensor_shape(tensor_name))
                    nbytes = int(_trt_volume(shape) * dtype.itemsize)
                    host = np.empty(shape, dtype=dtype)
                    t_alloc0 = time.perf_counter_ns()
                    err, ptr = cudart.cudaMalloc(nbytes)
                    _cuda_check((err,), "cudaMalloc(output)")
                    alloc_ns += int(time.perf_counter_ns() - t_alloc0)
                    device_allocations.append(int(ptr))
                    context.set_tensor_address(tensor_name, int(ptr))
                    host_outputs.append(host)
                    output_ptrs.append(int(ptr))
            t_exec0 = time.perf_counter_ns()
            if not context.execute_async_v3(0):
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            exec_ns += int(time.perf_counter_ns() - t_exec0)
            for host, ptr in zip(host_outputs, output_ptrs):
                t_d2h0 = time.perf_counter_ns()
                _cuda_check(
                    cudart.cudaMemcpy(
                        host.ctypes.data,
                        int(ptr),
                        int(host.nbytes),
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    ),
                    "cudaMemcpy(D2H)",
                )
                d2h_ns += int(time.perf_counter_ns() - t_d2h0)
        else:
            # TensorRT 8/9 legacy bindings API.
            bindings: list[int] = [0] * int(getattr(engine, "num_bindings"))
            for binding_idx in range(int(engine.num_bindings)):
                is_input = bool(engine.binding_is_input(binding_idx))
                dtype = np.dtype(trt.nptype(engine.get_binding_dtype(binding_idx)))
                if is_input:
                    shape = tuple(int(x) for x in input_array.shape)
                    if any(int(v) < 0 for v in engine.get_binding_shape(binding_idx)):
                        context.set_binding_shape(binding_idx, shape)
                    nbytes = int(input_array.nbytes)
                    t_alloc0 = time.perf_counter_ns()
                    err, ptr = cudart.cudaMalloc(nbytes)
                    _cuda_check((err,), "cudaMalloc(input)")
                    alloc_ns += int(time.perf_counter_ns() - t_alloc0)
                    device_allocations.append(int(ptr))
                    t_h2d0 = time.perf_counter_ns()
                    _cuda_check(
                        cudart.cudaMemcpy(
                            int(ptr),
                            input_array.ctypes.data,
                            nbytes,
                            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        ),
                        "cudaMemcpy(H2D)",
                    )
                    h2d_ns += int(time.perf_counter_ns() - t_h2d0)
                    bindings[binding_idx] = int(ptr)
                else:
                    shape = tuple(int(x) for x in context.get_binding_shape(binding_idx))
                    nbytes = int(_trt_volume(shape) * dtype.itemsize)
                    host = np.empty(shape, dtype=dtype)
                    t_alloc0 = time.perf_counter_ns()
                    err, ptr = cudart.cudaMalloc(nbytes)
                    _cuda_check((err,), "cudaMalloc(output)")
                    alloc_ns += int(time.perf_counter_ns() - t_alloc0)
                    device_allocations.append(int(ptr))
                    bindings[binding_idx] = int(ptr)
                    host_outputs.append(host)
                    output_ptrs.append(int(ptr))
            t_exec0 = time.perf_counter_ns()
            if not context.execute_v2(bindings):
                raise RuntimeError("TensorRT execute_v2 returned False")
            exec_ns += int(time.perf_counter_ns() - t_exec0)
            for host, ptr in zip(host_outputs, output_ptrs):
                t_d2h0 = time.perf_counter_ns()
                _cuda_check(
                    cudart.cudaMemcpy(
                        host.ctypes.data,
                        int(ptr),
                        int(host.nbytes),
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    ),
                    "cudaMemcpy(D2H)",
                )
                d2h_ns += int(time.perf_counter_ns() - t_d2h0)
    finally:
        for ptr in device_allocations:
            try:
                cudart.cudaFree(int(ptr))
            except Exception:
                pass
    t_dec0 = time.perf_counter_ns()
    raw = _select_output_tensor([np.asarray(x) for x in host_outputs])
    preds = _decode_onnx_predictions(
        raw,
        image_path=image_path,
        names=names,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        orig_hw=orig_hw,
        gain=gain,
        pad=pad,
    )
    t1 = time.perf_counter_ns()
    infer_runtime_ns = int(h2d_ns + exec_ns + d2h_ns)
    runtime_total = int((t_pre1 - t_pre0) + infer_runtime_ns + (t1 - t_dec0))
    return preds, {
        "total": runtime_total,
        "io_load": int(t_io1 - t_io0),
        "preprocess": int(t_pre1 - t_pre0),
        "infer": infer_runtime_ns,
        "decode_nms": int(t1 - t_dec0),
        "diagnostics_alloc": int(alloc_ns),
        "diagnostics_h2d": int(h2d_ns),
        "diagnostics_execute": int(exec_ns),
        "diagnostics_d2h": int(d2h_ns),
    }


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
    ensure_run_layout(root_dir)
    test_path = format_recommendation_path_for_write(root_dir, "test", format_name)
    val_path = format_recommendation_path_for_write(root_dir, "val", format_name)
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
        "project": root_dir,
        "name": f"val-recs-{format_name}",
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
        val_result = model.val(**val_kwargs)
        try:
            val_csv = format_metrics_path_for_split(root_dir, "val", format_name)
            with open(val_csv, "w", encoding="utf-8") as f:
                f.write(val_result.to_csv())
        except Exception as csv_exc:
            print(f"[WARN] {format_name}: failed to persist val metrics csv: {csv_exc}")
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


def _ultralytics_val_task_kw(task_type: str | None) -> dict[str, str]:
    t = str(task_type or "").strip().lower()
    if t in {"classification", "classify", "cls"}:
        return {"task": "classify"}
    if t in {"segmentation", "segment", "seg"}:
        return {"task": "segment"}
    if t in {"detection", "detect", ""}:
        return {}
    return {}


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
    deep_diagnostics: bool = False,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
    runtime_device: str | None = None,
    task_type: str | None = None,
) -> BackendRunResult:
    def _ultralytics_perf_payload_from_result(
        val_result: Any, *, duration_s: float, warmup_images: int, images_count: int | None = None
    ) -> dict[str, Any]:
        speed = getattr(val_result, "speed", None)
        speed_map = speed if isinstance(speed, dict) else {}

        def _as_float(v: Any) -> float | None:
            try:
                if v is None:
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

        preprocess_ms = _as_float(speed_map.get("preprocess"))
        infer_ms = _as_float(speed_map.get("inference"))
        postprocess_ms = _as_float(speed_map.get("postprocess"))
        total_ms = _as_float(speed_map.get("total"))
        if total_ms is None:
            parts = [x for x in (preprocess_ms, infer_ms, postprocess_ms) if x is not None]
            total_ms = float(sum(parts)) if parts else None

        throughput = None
        if infer_ms is not None and infer_ms > 0:
            throughput = 1000.0 / infer_ms
        elif total_ms is not None and total_ms > 0:
            throughput = 1000.0 / total_ms

        count_guess = int(images_count) if isinstance(images_count, int) and images_count > 0 else None
        for attr in ("seen", "nt_per_image", "nt_per_class"):
            if isinstance(images_count, int) and images_count > 0:
                break
            raw = getattr(val_result, attr, None)
            try:
                if raw is None:
                    continue
                if hasattr(raw, "sum"):
                    count_guess = int(raw.sum())
                elif isinstance(raw, (list, tuple)):
                    count_guess = int(sum(int(x) for x in raw))
                else:
                    count_guess = int(raw)
            except Exception:
                count_guess = None
            if count_guess and count_guess > 0:
                break

        def _stats(val_ms: float | None) -> dict[str, Any]:
            if val_ms is None:
                return {}
            return {
                "count": int(count_guess or 0),
                "mean": float(val_ms),
                "p50": float(val_ms),
                "p90": float(val_ms),
                "p95": float(val_ms),
                "min": float(val_ms),
                "max": float(val_ms),
                "std": 0.0,
            }

        breakdown: dict[str, Any] = {}
        if preprocess_ms is not None:
            breakdown["preprocess_ms"] = _stats(preprocess_ms)
        if infer_ms is not None:
            breakdown["infer_ms"] = _stats(infer_ms)
        if postprocess_ms is not None:
            breakdown["decode_nms_ms"] = _stats(postprocess_ms)
        if total_ms is not None:
            breakdown["infer_total_only_ms"] = _stats(total_ms)

        latency_stats = _stats(infer_ms if infer_ms is not None else total_ms)
        return {
            "images_total": int(count_guess) if isinstance(count_guess, int) and count_guess > 0 else None,
            "warmup_images": int(max(0, warmup_images)),
            "duration_s": float(max(0.0, duration_s)),
            "throughput_img_s": float(throughput) if throughput is not None else 0.0,
            "latency_ms": {"all": latency_stats, "steady": latency_stats},
            "breakdown_ms": breakdown,
            "infer_total_only": True,
            "source": "ultralytics_speed_dict",
            "eval_batch": int(val_batch) if val_batch is not None else None,
            "eval_device": str(runtime_device) if runtime_device else None,
        }

    test_start_time = datetime.now()
    model = YOLO(weights_path)
    val_kwargs = {
        "data": dataset_yaml_path,
        "split": "test",
        "project": root_dir,
        "name": os.path.basename(format_test_dir_for_write(root_dir, format_name)),
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
    if runtime_device is not None and str(runtime_device).strip():
        val_kwargs["device"] = str(runtime_device).strip()
    val_kwargs.update(_ultralytics_val_task_kw(task_type))
    try:
        test_image_count = len(_split_images_from_yaml(dataset_yaml_path, "test", 0))
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
        if deep_diagnostics:
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            input_hw = (int(eval_params["imgsz"]), int(eval_params["imgsz"]))
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            names = _load_names(dataset_yaml_path)
            pred_proj = ultralytics_sidecar_dir(root_dir, ".ultralytics_predict_scratch")
            pred_common = {
                "save": False,
                "project": pred_proj,
                "name": "deep-diagnostics",
                "exist_ok": True,
            }

            # Deep diagnostics are optional, but when enabled they must be produced for test and val.
            for split in ("test", "val"):
                try:
                    gt_rows_split, _bgv_split, image_paths_split = _collect_gt(dataset_yaml_path, split)
                except Exception as exc:
                    if split == "test":
                        raise
                    print(f"[WARN] {format_name}: deep-diagnostics could not collect GT for split={split}: {exc}")
                    continue

                # Batched prediction for deep diagnostics can spike memory.
                # We chunk the input to keep peak RSS low and avoid OOM-killer.
                preds_split: list[_Pred] = []
                predict_chunk_size = 10

                def _append_preds_from_results(results_iter: Any, *, chunk_paths: list[str], chunk_start_idx: int) -> None:
                    # idx->image_path mapping relies on Ultralytics preserving input order.
                    for idx, r0 in tqdm(
                        enumerate(results_iter),
                        desc=f"{format_name}:deep_{split}",
                        unit="img",
                        file=sys.stdout,
                        total=len(chunk_paths),
                    ):
                        image_path = (
                            chunk_paths[idx]
                            if idx < len(chunk_paths)
                            else str(getattr(r0, "path", ""))
                        )
                        boxes = getattr(r0, "boxes", None)
                        if boxes is None:
                            del r0
                            continue
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                        for b, c, k in zip(xyxy, confs, clss):
                            preds_split.append(
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
                        # Best-effort cleanup: Results objects keep references to large arrays.
                        del r0, boxes, xyxy, confs, clss
                        # Periodic GC to control peak RSS without destroying performance.
                        if (chunk_start_idx + idx) % 10 == 0:
                            _release_cuda_memory_best_effort()

                def _is_cuda_oom_text(exc: Exception) -> bool:
                    msg = str(exc).lower()
                    return ("out of memory" in msg and "cuda" in msg) or "cudamemoryerror" in msg

                for chunk_start in range(0, len(image_paths_split), predict_chunk_size):
                    chunk_paths = image_paths_split[chunk_start : chunk_start + predict_chunk_size]
                    if not chunk_paths:
                        continue
                    try:
                        # stream=True prevents Ultralytics from buffering all Results objects in RAM.
                        results_iter = model.predict(
                            source=chunk_paths,
                            imgsz=int(input_hw[0]),
                            conf=float(conf_thr),
                            iou=float(iou_thr),
                            verbose=False,
                            batch=int(val_batch) if val_batch is not None else 1,
                            stream=True,
                            **pred_common,
                        )
                        _append_preds_from_results(results_iter, chunk_paths=chunk_paths, chunk_start_idx=chunk_start)
                    except Exception as exc:
                        if _is_cuda_oom_text(exc):
                            print(
                                f"[WARN] {format_name}: deep-diagnostics predict OOM on GPU for split={split}, chunk={chunk_start}. "
                                "Retrying on CPU.",
                                file=sys.stderr,
                            )
                            _release_cuda_memory_best_effort()
                            results_iter = model.predict(
                                source=chunk_paths,
                                imgsz=int(input_hw[0]),
                                conf=float(conf_thr),
                                iou=float(iou_thr),
                                verbose=False,
                                device="cpu",
                                batch=1,
                                stream=True,
                                **pred_common,
                            )
                            _append_preds_from_results(results_iter, chunk_paths=chunk_paths, chunk_start_idx=chunk_start)
                        else:
                            raise
                    _release_cuda_memory_best_effort()

                _write_deep_diagnostics_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name="ultralytics_predict",
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split=split,
                    preds=preds_split,
                    gt_rows=gt_rows_split,
                    image_paths=image_paths_split,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="ultralytics_model_predict",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_validator_multilabel",
                )
        test_end_time = datetime.now()
        test_system_profile = _collect_test_system_profile(
            root_dir=root_dir,
            format_name=format_name,
            backend_name="ultralytics",
            runtime_provider="ultralytics",
            runtime_device=str(val_kwargs.get("device", "")) or (str(runtime_device) if runtime_device else None),
        )
        perf_payload: dict[str, Any] | None = None
        if collect_performance:
            duration_s = max(0.0, (test_end_time - test_start_time).total_seconds())
            perf_payload = _ultralytics_perf_payload_from_result(
                result,
                duration_s=duration_s,
                warmup_images=perf_warmup_images,
                images_count=test_image_count,
            )
            _write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            performance=perf_payload,
            test_system_profile=test_system_profile,
            status="ok",
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=True,
            test_start_time=test_start_time,
            test_end_time=test_end_time,
            inference={
                "imgsz": imgsz,
                "conf": val_conf,
                "iou": val_iou,
                "batch": val_batch,
                "inference_source": "ultralytics_model_val",
                "gt_source": "ultralytics_validator",
                "nms_profile": "ultralytics_validator_multilabel",
                "performance": perf_payload,
                "test_system_profile": test_system_profile,
            },
            target_path=weights_path,
        )
    except Exception as exc:
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend="ultralytics",
            test_system_profile=_collect_test_system_profile(
                root_dir=root_dir,
                format_name=format_name,
                backend_name="ultralytics",
                runtime_provider="ultralytics",
            ),
            status="failed",
            error=str(exc),
        )
        return BackendRunResult(
            format=format_name,
            backend="ultralytics",
            success=False,
            test_start_time=test_start_time,
            test_end_time=datetime.now(),
            inference={
                "imgsz": imgsz,
                "conf": val_conf,
                "iou": val_iou,
                "batch": val_batch,
                "inference_source": "ultralytics_model_val",
                "gt_source": "ultralytics_validator",
                "nms_profile": "ultralytics_validator_multilabel",
            },
            target_path=weights_path,
            error=str(exc),
        )
    finally:
        best_effort_prune_runs_detect_near_run(root_dir)


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
    deep_diagnostics: bool = False,
    collect_performance: bool = False,
    perf_warmup_images: int = 5,
    onnx_provider_policy: str | None = None,
    runtime_device: str | None = None,
    task_type: str | None = None,
) -> BackendRunResult:
    backend_name = "onnxruntime" if format_name == "onnx" else ("unified_pt" if format_name == "pt_uni" else "tensorrt")
    provider_by_format = {
        "onnx": "onnxruntime",
        "engine": "tensorrt",
        "trt": "tensorrt",
        "pt_uni": "ultralytics",
    }
    try:
        if format_name == "pt_uni":
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            result = run_ultralytics_backend(
                root_dir=root_dir,
                weights_path=weights_path,
                dataset_yaml_path=dataset_yaml_path,
                format_name="pt_uni",
                imgsz=int(eval_params["imgsz"]),
                val_conf=float(eval_params["conf"]),
                val_iou=float(eval_params["iou"]),
                val_batch=val_batch,
                deep_diagnostics=deep_diagnostics,
                collect_performance=collect_performance,
                perf_warmup_images=perf_warmup_images,
                task_type=task_type,
            )
            if result.success:
                test_dir = format_test_dir_for_write(root_dir, "pt_uni")
                os.makedirs(test_dir, exist_ok=True)
                _write_test_args_yaml(
                    test_dir,
                    backend=backend_name,
                    format_name="pt_uni",
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    imgsz=int(eval_params["imgsz"]),
                    conf=float(eval_params["conf"]),
                    iou=float(eval_params["iou"]),
                    batch=val_batch,
                    inference_source="ultralytics_model_val",
                    gt_source="ultralytics_validator",
                    nms_profile="ultralytics_validator_multilabel",
                )
                persist_target_test_artifacts_state(
                    root_dir,
                    format_name="pt_uni",
                    target_path=weights_path,
                    dataset_yaml=dataset_yaml_path,
                    backend=backend_name,
                    performance=result.inference.get("performance") if isinstance(result.inference, dict) else None,
                    test_system_profile=(
                        result.inference.get("test_system_profile")
                        if isinstance(result.inference, dict)
                        else None
                    ),
                    status="ok",
                )
            else:
                persist_target_test_artifacts_state(
                    root_dir,
                    format_name="pt_uni",
                    target_path=weights_path,
                    dataset_yaml=dataset_yaml_path,
                    backend=backend_name,
                    test_system_profile=_collect_test_system_profile(
                        root_dir=root_dir,
                        format_name="pt_uni",
                        backend_name=backend_name,
                        runtime_provider="ultralytics",
                        runtime_device=runtime_device,
                    ),
                    status="failed",
                    error=result.error,
                )
            result.backend = backend_name
            if isinstance(result.inference, dict):
                result.inference.update(
                    EvalProvenance(
                        inference_source="ultralytics_model_val",
                        gt_source="ultralytics_validator",
                        nms_profile="ultralytics_validator_multilabel",
                    ).as_dict()
                )
            return result
        if format_name == "onnx":
            inferred_policy = "cpu_only" if str(runtime_device or "").strip().lower() == "cpu" else None
            policy = str(onnx_provider_policy or inferred_policy or os.getenv("SMARTTRAIN_ONNX_PROVIDER_POLICY", "gpu_preferred")).strip().lower()
            if policy not in {"gpu_strict", "gpu_preferred", "cpu_only"}:
                policy = "gpu_preferred"
            providers = (
                ["CPUExecutionProvider"]
                if policy == "cpu_only"
                else ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            names = _load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _collect_gt(dataset_yaml_path, "test")
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            requested_imgsz = int(eval_params["imgsz"])
            strict_imgsz = str(os.getenv("SMARTTRAIN_ONNX_IMGSZ_STRICT", "0")).strip().lower() in {"1", "true", "yes"}
            use_worker = str(os.getenv("SMARTTRAIN_ONNX_USE_SUBPROCESS", "1")).strip().lower() not in {"0", "false", "no"}
            perf_collector = PerfCollector(warmup_images=perf_warmup_images) if collect_performance else None
            perf_payload: dict[str, Any] | None = None
            provider_actual: str | None = None
            session_init_ns = 0
            provider_switched_to_cpu = False
            if use_worker:
                preds, input_hw, perf_payload, provider_actual = _run_onnx_split_in_subprocess(
                    split_name="test",
                    image_paths=image_paths,
                    weights_path=weights_path,
                    dataset_yaml_path=dataset_yaml_path,
                    imgsz=requested_imgsz,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    providers=providers,
                    provider_policy=policy,
                    collect_performance=collect_performance,
                    perf_warmup_images=perf_warmup_images,
                )
            else:
                import onnxruntime as ort  # type: ignore

                available = list(ort.get_available_providers())
                providers_local = [p for p in providers if p in available]
                if policy == "gpu_strict" and "CUDAExecutionProvider" not in providers_local:
                    raise RuntimeError(_format_onnx_error("provider_unavailable", f"CUDAExecutionProvider is unavailable. available={available}"))
                try:
                    t_sess0 = time.perf_counter_ns()
                    session = _build_onnx_session_with_retry(ort, weights_path, providers_local)
                    session_init_ns = int(time.perf_counter_ns() - t_sess0)
                except Exception as primary_exc:
                    if policy == "gpu_strict":
                        raise RuntimeError(_format_onnx_error(_classify_onnx_error_text(str(primary_exc)), str(primary_exc))) from primary_exc
                    cpu_only = ["CPUExecutionProvider"] if "CPUExecutionProvider" in available else None
                    if not cpu_only:
                        raise primary_exc
                    print("[WARN] onnx: switching to CPUExecutionProvider after repeated initialization failures.")
                    provider_switched_to_cpu = True
                    t_sess0 = time.perf_counter_ns()
                    session = ort.InferenceSession(weights_path, providers=cpu_only)
                    session_init_ns = int(time.perf_counter_ns() - t_sess0)
                try:
                    provider_actual = str((session.get_providers() or [None])[0] or "")
                except Exception:
                    provider_actual = None
                input_hw = _resolve_imgsz_from_onnx(session, requested_imgsz)
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
                    perf_collector=perf_collector,
                )
                if perf_collector is not None:
                    perf_payload = perf_collector.to_payload()
            if isinstance(perf_payload, dict):
                perf_payload.setdefault("diagnostics_overhead", {})
                perf_payload["diagnostics_overhead"].update(
                    {
                        "session_init_ms": float(session_init_ns / 1_000_000.0),
                        "provider_switched_to_cpu": bool(provider_switched_to_cpu),
                    }
                )
            if not isinstance(input_hw, tuple):
                input_hw = (requested_imgsz, requested_imgsz)
            if strict_imgsz and requested_imgsz != int(input_hw[0]):
                raise RuntimeError(
                    _format_onnx_error(
                        "shape_mismatch",
                        f"requested imgsz={requested_imgsz} does not match model input={int(input_hw[0])}",
                    )
                )
            if perf_payload:
                _write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
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
                inference_source="onnxruntime_session",
                gt_source="ultralytics_verify_image_label",
                nms_profile="ultralytics_nms_multilabel",
            )
            if deep_diagnostics:
                _write_deep_diagnostics_artifacts(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    weights_path=weights_path,
                    data_yaml_path=dataset_yaml_path,
                    split="test",
                    preds=preds,
                    gt_rows=gt_rows,
                    image_paths=image_paths,
                    names=names,
                    conf_thr=conf_thr,
                    iou_thr=iou_thr,
                    imgsz=input_hw[0],
                    batch=val_batch,
                    inference_source="onnxruntime_session",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
            split_status: dict[str, Any] = {"test": {"status": "ok", "error": None}, "val": {"status": "ok", "error": None}}
            try:
                gt_rows_val, _bgv, image_paths_val = _collect_gt(dataset_yaml_path, "val")
                if use_worker:
                    preds_val, _input_hw_val, _perf_val, _provider_val = _run_onnx_split_in_subprocess(
                        split_name="val",
                        image_paths=image_paths_val,
                        weights_path=weights_path,
                        dataset_yaml_path=dataset_yaml_path,
                        imgsz=input_hw[0],
                        conf_thr=conf_thr,
                        iou_thr=iou_thr,
                        providers=providers,
                        provider_policy=policy,
                        collect_performance=False,
                    )
                else:
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
                    inference_source="onnxruntime_session",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
                if deep_diagnostics:
                    _write_deep_diagnostics_artifacts(
                        root_dir=root_dir,
                        format_name=format_name,
                        backend_name=backend_name,
                        weights_path=weights_path,
                        data_yaml_path=dataset_yaml_path,
                        split="val",
                        preds=preds_val,
                        gt_rows=gt_rows_val,
                        image_paths=image_paths_val,
                        names=names,
                        conf_thr=conf_thr,
                        iou_thr=iou_thr,
                        imgsz=input_hw[0],
                        batch=val_batch,
                        inference_source="onnxruntime_session",
                        gt_source="ultralytics_verify_image_label",
                        nms_profile="ultralytics_nms_multilabel",
                    )
            except Exception as val_exc:
                split_status["val"] = {"status": "failed", "error": str(val_exc)}
            overall_status = "ok" if split_status["val"]["status"] == "ok" else "partial_ok"
            overall_error = None if overall_status == "ok" else f"val split failed: {split_status['val']['error']}"
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                performance=perf_payload,
                test_system_profile=_collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider=("onnxruntime-worker" if use_worker else "onnxruntime-session"),
                    runtime_provider_actual=provider_actual,
                    runtime_device=runtime_device,
                ),
                status=overall_status,
                error=overall_error,
                split_status=split_status,
            )
            if isinstance(inference, dict):
                inference["onnx_provider_policy"] = policy
                inference["onnx_provider_actual"] = provider_actual
                inference["performance"] = perf_payload
                inference["test_system_profile"] = _collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider=("onnxruntime-worker" if use_worker else "onnxruntime-session"),
                    runtime_provider_actual=provider_actual,
                    runtime_device=runtime_device,
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
        elif format_name in {"engine", "trt"}:
            names = _load_names(dataset_yaml_path)
            gt_rows, _by_image_gt, image_paths = _collect_gt(dataset_yaml_path, "test")
            eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
            conf_thr = float(eval_params["conf"])
            iou_thr = float(eval_params["iou"])
            input_hw = _resolve_input_hw_from_native_artifact(weights_path, int(eval_params["imgsz"]))
            perf_collector = PerfCollector(warmup_images=perf_warmup_images) if collect_performance else None
            trt_runtime = _prepare_trt_runtime(weights_path)
            preds: list[_Pred] = []
            total_images = len(image_paths)
            print(f"[INFO] {format_name}: running native test on {total_images} images with {weights_path}")
            for image_path in tqdm(image_paths, desc=f"{format_name}:test", unit="img", file=sys.stdout):
                infer_out = _infer_with_trt_engine(trt_runtime, image_path, input_hw, conf_thr, iou_thr, names)
                if isinstance(infer_out, tuple) and len(infer_out) == 2:
                    image_preds, perf_ns = infer_out
                else:
                    image_preds, perf_ns = infer_out, {}
                preds.extend(image_preds)
                if perf_collector is not None:
                    perf_collector.record_total_image(int(perf_ns.get("total", 0)))
                    perf_collector.record_stage("preprocess_ms", int(perf_ns.get("preprocess", 0)))
                    perf_collector.record_stage("infer_ms", int(perf_ns.get("infer", 0)))
                    perf_collector.record_stage("decode_nms_ms", int(perf_ns.get("decode_nms", 0)))
                    perf_collector.record_stage("io_load_ms", int(perf_ns.get("io_load", 0)))
                    perf_collector.record_stage("diagnostics_alloc_ms", int(perf_ns.get("diagnostics_alloc", 0)))
                    perf_collector.record_stage("diagnostics_h2d_ms", int(perf_ns.get("diagnostics_h2d", 0)))
                    perf_collector.record_stage("diagnostics_execute_ms", int(perf_ns.get("diagnostics_execute", 0)))
                    perf_collector.record_stage("diagnostics_d2h_ms", int(perf_ns.get("diagnostics_d2h", 0)))
            print(f"[INFO] {format_name}: native test completed ({total_images}/{total_images} images).")
            perf_payload = perf_collector.to_payload() if perf_collector is not None else None
            if isinstance(perf_payload, dict):
                perf_payload.setdefault("diagnostics_overhead", {})
                perf_payload["diagnostics_overhead"]["engine_init_ms"] = float(
                    int(trt_runtime.get("init_ns", 0)) / 1_000_000.0
                )
            if perf_payload:
                _write_perf_artifact(root_dir, format_name, weights_path, perf_payload)
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
                inference_source="tensorrt_engine",
                gt_source="ultralytics_verify_image_label",
                nms_profile="ultralytics_nms_multilabel",
            )
            split_status: dict[str, Any] = {"test": {"status": "ok", "error": None}, "val": {"status": "ok", "error": None}}
            native_debug: dict[str, Any] = {
                "imgsz": input_hw[0],
                "test_gt_count": len(gt_rows),
                "test_pred_count": len(preds),
            }
            try:
                native_debug["invalid_metrics_candidate"] = bool(
                    all(abs(float(inference.get(k) or 0.0)) <= 1e-12 for k in ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R"))
                )
            except Exception:
                native_debug["invalid_metrics_candidate"] = False
            try:
                gt_rows_val, _bgv, image_paths_val = _collect_gt(dataset_yaml_path, "val")
                preds_val: list[_Pred] = []
                print(f"[INFO] {format_name}: running native val on {len(image_paths_val)} images with {weights_path}")
                for image_path in tqdm(image_paths_val, desc=f"{format_name}:val", unit="img", file=sys.stdout):
                    infer_out = _infer_with_trt_engine(trt_runtime, image_path, input_hw, conf_thr, iou_thr, names)
                    if isinstance(infer_out, tuple) and len(infer_out) == 2:
                        preds_val.extend(infer_out[0])
                    else:
                        preds_val.extend(infer_out)
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
                    inference_source="tensorrt_engine",
                    gt_source="ultralytics_verify_image_label",
                    nms_profile="ultralytics_nms_multilabel",
                )
                native_debug["val_gt_count"] = len(gt_rows_val)
                native_debug["val_pred_count"] = len(preds_val)
            except Exception as val_exc:
                split_status["val"] = {"status": "failed", "error": str(val_exc)}
            overall_status = "ok" if split_status["val"]["status"] == "ok" else "partial_ok"
            overall_error = None if overall_status == "ok" else f"val split failed: {split_status['val']['error']}"
            persist_target_test_artifacts_state(
                root_dir,
                format_name=format_name,
                target_path=weights_path,
                dataset_yaml=dataset_yaml_path,
                backend=backend_name,
                performance=perf_payload,
                test_system_profile=_collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider="tensorrt",
                    runtime_device=runtime_device,
                ),
                status=overall_status,
                error=overall_error,
                split_status=split_status,
                native_debug=native_debug,
            )
            if isinstance(inference, dict):
                inference["performance"] = perf_payload
                inference["test_system_profile"] = _collect_test_system_profile(
                    root_dir=root_dir,
                    format_name=format_name,
                    backend_name=backend_name,
                    runtime_provider="tensorrt",
                    runtime_device=runtime_device,
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
        err_text = str(exc)
        if format_name == "onnx" and not err_text.strip().startswith("["):
            err_text = _format_onnx_error(_classify_onnx_error_text(err_text), err_text)
        persist_target_test_artifacts_state(
            root_dir,
            format_name=format_name,
            target_path=weights_path,
            dataset_yaml=dataset_yaml_path,
            backend=backend_name,
            test_system_profile=_collect_test_system_profile(
                root_dir=root_dir,
                format_name=format_name,
                backend_name=backend_name,
                runtime_provider=provider_by_format.get(format_name),
                runtime_device=runtime_device,
            ),
            status="unavailable",
            error=err_text,
        )
        return BackendRunResult(
            format=format_name,
            backend=backend_name,
            success=False,
            test_start_time=datetime.now(),
            test_end_time=datetime.now(),
            inference={"imgsz": imgsz, "conf": val_conf, "iou": val_iou, "batch": val_batch},
            target_path=weights_path,
            error=err_text,
        )
