from __future__ import annotations

import os
import sys
import gc
import time
import tempfile
import json
import re
import shutil
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

from smartrain.core.inference.ultralytics_metrics_pr import (
    extract_pr_curve_from_ultralytics_metrics,
    extract_pr_curve_per_class_from_ultralytics_metrics,
)
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
from smartrain.core.runtime.run_artifacts import (
    canonicalize_run_ultralytics_layout,
    ensure_run_layout,
    read_model_sidecar_metadata,
    run_tests_dir,
)
from smartrain.workflows.models import tensorrt_checks as trt_checks
from smartrain.core.runtime.ultralytics_ephemeral import (
    best_effort_prune_runs_detect_near_run,
    prune_empty_sidecar_dirs,
    ultralytics_sidecar_dir,
)
from smartrain.services.testing.unified_metrics_adapter import collect_ultralytics_style_gt
from smartrain.workflows.testing.unified_validator_core import EvalProvenance, normalize_eval_params


from smartrain.services.testing.backends import native_eval as _native_eval
from smartrain.services.testing.backends.native_eval import (
    BackendRunResult,
    PerfCollector,
    _Pred,
    _Gt,
    _Box,
    _box_iou_np,
    _build_pr_payload,
    _build_ultralytics_style_stats,
    _collect_gt,
    _collect_test_system_profile,
    _compute_ap,
    _compute_global_stats,
    _compute_map5095,
    _compute_threshold_curves,
    _compute_ultralytics_style_deep_payload,
    _compute_ultralytics_style_payload,
    _decode_onnx_predictions,
    _label_path_for_image,
    _letterbox,
    _load_data_cfg,
    _load_image_rgb,
    _load_names,
    _match_class_predictions,
    _preprocess_array,
    _read_gt_boxes,
    _resolve_imgsz_from_onnx,
    _resolve_input_hw_from_native_artifact,
    _save_metrics_csv_for_format,
    _select_output_tensor,
    _split_images_from_yaml,
    _write_perf_artifact,
    _xywhn_to_xyxy,
    _iou,
)

# Re-export for tests/scripts that import private helpers from workflows.
for _name in (
    "_Pred",
    "_Gt",
    "_Box",
    "_build_ultralytics_style_stats",
    "_compute_ultralytics_style_payload",
):
    globals()[_name] = getattr(_native_eval, _name)



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
        "project": str(run_tests_dir(root_dir)),
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
    try:
        canonicalize_run_ultralytics_layout(root_dir)
    except Exception:
        pass


def _ultralytics_val_task_kw(task_type: str | None) -> dict[str, str]:
    t = str(task_type or "").strip().lower()
    if t in {"classification", "classify", "cls"}:
        return {"task": "classify"}
    if t in {"segmentation", "segment", "seg"}:
        return {"task": "segment"}
    if t in {"detection", "detect", ""}:
        return {}
    return {}


def _finalize_ultralytics_pt_test_dir(
    *,
    root_dir: str,
    format_name: str,
    result: Any,
    weights_path: str,
    dataset_yaml_path: str,
    imgsz: int | None,
    val_conf: float | None,
    val_iou: float | None,
    val_batch: int | None,
) -> None:
    """Normalize Ultralytics val() output into the canonical tests/test-ultralytics layout."""
    if str(format_name or "pt").strip().lower() != "pt":
        return
    test_dir = format_test_dir_for_write(root_dir, format_name)
    os.makedirs(test_dir, exist_ok=True)
    for src_name, dst_name in (
        ("PR_curve.png", "BoxPR_curve.png"),
        ("F1_curve.png", "BoxF1_curve.png"),
        ("P_curve.png", "BoxP_curve.png"),
        ("R_curve.png", "BoxR_curve.png"),
    ):
        src_p = os.path.join(test_dir, src_name)
        dst_p = os.path.join(test_dir, dst_name)
        if os.path.isfile(src_p) and not os.path.isfile(dst_p):
            shutil.copy2(src_p, dst_p)

    pr_pair = extract_pr_curve_from_ultralytics_metrics(result)
    if pr_pair is not None:
        recall, precision = pr_pair
        pd.DataFrame({"recall": recall.astype(float), "precision": precision.astype(float)}).to_csv(
            os.path.join(test_dir, "pr.csv"), index=False, encoding="utf-8"
        )
    elif not os.path.isfile(os.path.join(test_dir, "pr.csv")):
        pd.DataFrame({"recall": [0.0], "precision": [0.0]}).to_csv(
            os.path.join(test_dir, "pr.csv"), index=False, encoding="utf-8"
        )

    per_class = extract_pr_curve_per_class_from_ultralytics_metrics(result)
    names_list = _load_names(dataset_yaml_path)
    pc_path = os.path.join(test_dir, "pr_per_class.csv")
    if per_class is not None:
        rx, y2d = per_class
        rows: list[dict[str, Any]] = []
        for class_id in range(y2d.shape[0]):
            class_name = names_list[class_id] if class_id < len(names_list) else f"class_{class_id}"
            ap = float(np.trapz(y2d[class_id], rx))
            for idx in range(len(rx)):
                rows.append(
                    {
                        "run_dir": root_dir,
                        "model": "pt",
                        "class_id": class_id,
                        "class_name": class_name,
                        "recall": float(rx[idx]),
                        "precision": float(y2d[class_id, idx]),
                        "ap": ap,
                    }
                )
        pd.DataFrame(rows).to_csv(pc_path, index=False, encoding="utf-8")
    elif not os.path.isfile(pc_path):
        pd.DataFrame(
            columns=["run_dir", "model", "class_id", "class_name", "recall", "precision", "ap"],
        ).to_csv(pc_path, index=False, encoding="utf-8")

    if not os.path.isfile(os.path.join(test_dir, "args.yaml")):
        eval_params = normalize_eval_params(imgsz=imgsz, conf=val_conf, iou=val_iou)
        _write_test_args_yaml(
            test_dir,
            backend="ultralytics",
            format_name=format_name,
            weights_path=weights_path,
            data_yaml_path=dataset_yaml_path,
            imgsz=int(eval_params["imgsz"]),
            conf=float(eval_params["conf"]),
            iou=float(eval_params["iou"]),
            batch=int(val_batch) if val_batch is not None else None,
            inference_source="ultralytics_model_val",
            gt_source="ultralytics_validator",
            nms_profile="ultralytics_validator_multilabel",
        )

    if not any(
        os.path.isfile(os.path.join(test_dir, n))
        for n in ("val_batch0_pred.jpg", "val_batch0_labels.jpg", "val_batch1_pred.jpg")
    ):
        print(
            f"[WARN] {format_name}: expected val_batch preview images not found under {test_dir}; "
            "Ultralytics version or dataset may omit them.",
            file=sys.stderr,
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
        "project": str(run_tests_dir(root_dir)),
        "name": "test-ultralytics",
        "exist_ok": True,
        "plots": True,
        "save": True,
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
        ensure_run_layout(root_dir)
        test_image_count = len(_split_images_from_yaml(dataset_yaml_path, "test", 0))
        result = model.val(**val_kwargs)
        _save_metrics_csv_for_format(result, root_dir, format_name)
        _finalize_ultralytics_pt_test_dir(
            root_dir=root_dir,
            format_name=format_name,
            result=result,
            weights_path=weights_path,
            dataset_yaml_path=dataset_yaml_path,
            imgsz=imgsz,
            val_conf=val_conf,
            val_iou=val_iou,
            val_batch=val_batch,
        )
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
        try:
            canonicalize_run_ultralytics_layout(root_dir)
        except Exception:
            pass
        best_effort_prune_runs_detect_near_run(root_dir)
        try:
            prune_empty_sidecar_dirs(root_dir)
        except Exception:
            pass


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
