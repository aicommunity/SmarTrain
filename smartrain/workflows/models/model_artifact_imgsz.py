from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_INFERENCE_IMGSZ = 640

_IMGSZ_FILENAME_RE = re.compile(r"_imgsz(\d+)x(\d+)_", re.IGNORECASE)


def parse_imgsz_from_artifact_filename(path: str | Path) -> int | None:
    """Parse square imgsz from model convert variant names like ``*_imgsz1280x1280_*``."""
    match = _IMGSZ_FILENAME_RE.search(Path(path).name)
    if not match:
        return None
    h = int(match.group(1))
    w = int(match.group(2))
    if h <= 0 or w <= 0:
        return None
    if h != w:
        return max(h, w)
    return h


def extract_imgsz_from_sidecar_payload(payload: dict[str, Any]) -> int | None:
    params = payload.get("params")
    if isinstance(params, dict):
        for key in ("imgsz", "image_size", "img_size"):
            value = params.get(key)
            if isinstance(value, (int, float)) and int(value) > 0:
                return int(value)
    for key in ("imgsz", "image_size", "img_size"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return None


def extract_onnx_input_imgsz(onnx_path: Path) -> int | None:
    """Read static H/W from ONNX graph input when available."""
    try:
        import onnx  # type: ignore
    except Exception:
        return None
    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None
    try:
        if not model.graph.input:
            return None
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) < 4:
            return None
        h = int(getattr(dims[2], "dim_value", 0) or 0)
        w = int(getattr(dims[3], "dim_value", 0) or 0)
        if h <= 0 or w <= 0:
            return None
        if h != w:
            return max(h, w)
        return h
    except Exception:
        return None
