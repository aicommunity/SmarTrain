from __future__ import annotations

from typing import Any

from smartrain.workflows.inference.inference_perf import DualPerfProfiler
from smartrain.core.runtime.dataset_resolution import find_yaml_file, resolve_dataset_root_for_entry
from smartrain.core.runtime.roi_geometry import clamp_crop, full_image_crop, select_roi_boxes
from smartrain.core.models.model_artifact_imgsz import (
    extract_batch_from_sidecar_payload,
    extract_onnx_input_batch,
)
from smartrain.core.models.model_context import (
    FALLBACK_IMGSZ_SOURCE,
    DEFAULT_INFERENCE_IMGSZ,
    infer_img_size_from_model_context,
    infer_img_size_with_source,
    resolve_inference_imgsz,
)

__all__ = [
    "DualPerfProfiler",
    "resolve_dataset_root_for_entry",
    "clamp_crop",
    "full_image_crop",
    "select_roi_boxes",
    "find_yaml_file",
    "DEFAULT_INFERENCE_IMGSZ",
    "FALLBACK_IMGSZ_SOURCE",
    "infer_img_size_from_model_context",
    "infer_img_size_with_source",
    "resolve_inference_imgsz",
    "extract_batch_from_sidecar_payload",
    "extract_onnx_input_batch",
]
