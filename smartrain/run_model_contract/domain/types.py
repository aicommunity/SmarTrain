from __future__ import annotations

from typing import Literal

SourceKind = Literal["run", "model"]
TaskType = Literal["detection", "classification", "segmentation"]
BackendType = Literal["ultralytics", "onnxruntime", "tensorrt", "external"]
ModelFormat = Literal["pt", "onnx", "engine", "trt"]

