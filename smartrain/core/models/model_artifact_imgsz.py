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


def extract_batch_from_sidecar_payload(payload: dict[str, Any]) -> tuple[int | None, bool | None]:
    """Read ``(batch, dynamic)`` from convert sidecar payload when present."""
    params = payload.get("params")
    source: dict[str, Any] = params if isinstance(params, dict) else payload
    batch_raw = source.get("batch")
    dynamic_raw = source.get("dynamic")
    batch: int | None = None
    if isinstance(batch_raw, (int, float)) and int(batch_raw) > 0:
        batch = int(batch_raw)
    dynamic: bool | None = None
    if isinstance(dynamic_raw, bool):
        dynamic = dynamic_raw
    elif isinstance(dynamic_raw, (int, float)):
        dynamic = bool(dynamic_raw)
    elif isinstance(dynamic_raw, str):
        token = dynamic_raw.strip().lower()
        if token in {"1", "true", "yes", "on", "dynamic"}:
            dynamic = True
        elif token in {"0", "false", "no", "off", "static"}:
            dynamic = False
    if batch is None and dynamic is None:
        return None, None
    return batch, dynamic


def extract_onnx_input_batch(onnx_path: Path) -> tuple[int | None, bool | None]:
    """Read ``(fixed_batch, is_dynamic)`` from ONNX graph input dim0 when available.

    Returns ``(None, None)`` when the graph cannot be read. When dynamic, ``fixed_batch``
    may still be set if a static dim_value is present; callers should treat ``dynamic=True``
    as unconstrained for inference batch clamping.
    """
    try:
        import onnx  # type: ignore
    except Exception:
        return None, None
    try:
        model = onnx.load(str(onnx_path))
    except Exception:
        return None, None
    try:
        if not model.graph.input:
            return None, None
        dims = model.graph.input[0].type.tensor_type.shape.dim
        if len(dims) < 1:
            return None, None
        d0 = dims[0]
        d2 = dims[2] if len(dims) > 2 else None
        d3 = dims[3] if len(dims) > 3 else None
        dyn = bool(getattr(d0, "dim_param", "") or "")
        if d2 is not None:
            dyn = dyn or bool(getattr(d2, "dim_param", "") or "")
        if d3 is not None:
            dyn = dyn or bool(getattr(d3, "dim_param", "") or "")
        batch: int | None = None
        dim_value = int(getattr(d0, "dim_value", 0) or 0)
        if dim_value > 0:
            batch = dim_value
        if batch is None and not dyn:
            return None, None
        return batch, dyn
    except Exception:
        return None, None
