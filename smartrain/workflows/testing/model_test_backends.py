"""CLI facade: test backend runners in smartrain.services.testing.backends."""

from __future__ import annotations

from smartrain.services.testing.backends.format_runners import (
    run_native_format_backend,
    run_ultralytics_backend,
    _infer_with_onnx_session,
    _is_onnx_cuda_oom_error,
    _release_cuda_memory_best_effort,
    _resolve_imgsz_from_onnx,
)
from smartrain.services.testing.backends.native_eval import (
    BackendRunResult,
    PerfCollector,
    _Pred,
    _Gt,
    _Box,
    _build_ultralytics_style_stats,
    _collect_gt,
    _compute_ultralytics_style_payload,
    _load_names,
)

__all__ = [
    "BackendRunResult",
    "PerfCollector",
    "run_native_format_backend",
    "run_ultralytics_backend",
    "_Pred",
    "_Gt",
    "_Box",
    "_build_ultralytics_style_stats",
    "_collect_gt",
    "_compute_ultralytics_style_payload",
    "_infer_with_onnx_session",
    "_is_onnx_cuda_oom_error",
    "_load_names",
    "_release_cuda_memory_best_effort",
    "_resolve_imgsz_from_onnx",
]
