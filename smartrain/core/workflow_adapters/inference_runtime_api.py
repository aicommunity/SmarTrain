from __future__ import annotations

from typing import Any

from smartrain.workflows.inference.inference_perf import DualPerfProfiler
from smartrain.services.datasets.dataset_access import resolve_dataset_root_for_entry
from smartrain.services.datasets.dataset_roi_yolo import _clamp_crop, _full_image_crop, _select_roi_boxes
from smartrain.services.datasets.dataset_scan import find_yaml_file
from smartrain.workflows.models.model_context import (
    FALLBACK_IMGSZ_SOURCE,
    DEFAULT_INFERENCE_IMGSZ,
    infer_img_size_from_model_context,
    infer_img_size_with_source,
    resolve_inference_imgsz,
)

__all__ = [
    "DualPerfProfiler",
    "resolve_dataset_root_for_entry",
    "_clamp_crop",
    "_full_image_crop",
    "_select_roi_boxes",
    "find_yaml_file",
    "DEFAULT_INFERENCE_IMGSZ",
    "FALLBACK_IMGSZ_SOURCE",
    "infer_img_size_from_model_context",
    "infer_img_size_with_source",
    "resolve_inference_imgsz",
]
