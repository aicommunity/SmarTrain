"""Test backend runners facade (monkeypatch surface for tests)."""

from __future__ import annotations

from smartrain.services.testing.backends.format_runners_native import run_native_format_backend
from smartrain.services.testing.backends.format_runners_support import (
    YOLO,
    subprocess,
    persist_target_test_artifacts_state,
    _build_onnx_session_with_retry,
    _classify_onnx_error_text,
    _collect_gt,
    _collect_test_system_profile,
    _ensure_confidence_recommendations_for_explicit_artifact,
    _finalize_ultralytics_pt_test_dir,
    _format_onnx_error,
    _infer_with_onnx_session,
    _infer_with_pt_model,
    _infer_with_trt_engine,
    _is_onnx_cuda_oom_error,
    _prepare_trt_runtime,
    _release_cuda_memory_best_effort,
    _resolve_imgsz_from_onnx,
    _run_onnx_split_in_subprocess,
    _run_onnx_split_with_retry,
    _save_metrics_csv_for_format,
)
from smartrain.services.testing.backends.format_runners_ultralytics import run_ultralytics_backend

__all__ = [
    "run_ultralytics_backend",
    "run_native_format_backend",
    "YOLO",
    "subprocess",
    "persist_target_test_artifacts_state",
    "_save_metrics_csv_for_format",
    "_finalize_ultralytics_pt_test_dir",
    "_ensure_confidence_recommendations_for_explicit_artifact",
    "_collect_test_system_profile",
    "_collect_gt",
    "_infer_with_onnx_session",
    "_is_onnx_cuda_oom_error",
    "_release_cuda_memory_best_effort",
    "_resolve_imgsz_from_onnx",
    "_prepare_trt_runtime",
    "_infer_with_trt_engine",
    "_build_onnx_session_with_retry",
    "_run_onnx_split_in_subprocess",
    "_run_onnx_split_with_retry",
    "_format_onnx_error",
    "_classify_onnx_error_text",
    "_infer_with_pt_model",
]
