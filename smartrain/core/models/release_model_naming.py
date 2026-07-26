"""Canonical model-weight naming derived from training metadata."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _normalize_task(task_type: str | None) -> str:
    return {
        "detection": "detect",
        "det": "detect",
        "detect": "detect",
        "classification": "classify",
        "classify": "classify",
        "cls": "classify",
        "segmentation": "segment",
        "segment": "segment",
        "seg": "segment",
    }.get((task_type or "").strip().lower(), (task_type or "").strip().lower() or "detect")


def _model_token(model_version: str) -> str:
    model = Path(model_version).name.removesuffix(".pt").removesuffix(".yaml")
    return re.sub(r"[^a-zA-Z0-9._+-]+", "-", model).strip("-") or "model"


def _parse_start_timestamp(payload: dict[str, Any]) -> datetime | None:
    timestamps = payload.get("timestamps")
    training = timestamps.get("training") if isinstance(timestamps, dict) else None
    raw = training.get("start") if isinstance(training, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def build_model_weights_stem_from_metadata(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    training_info = payload.get("training_info")
    if not isinstance(training_info, dict):
        return None
    hyperparameters = training_info.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or not training_info.get("model"):
        return None
    try:
        epochs = int(hyperparameters.get("epochs"))
        image_size = int(hyperparameters.get("image_size"))
        batch_value = hyperparameters.get("batch_size")
        batch = f"b{str(batch_value).replace('.', 'p')}" if isinstance(batch_value, float) and not batch_value.is_integer() else f"b{int(batch_value)}"
    except (TypeError, ValueError):
        return None
    timestamp = _parse_start_timestamp(payload)
    timestamp_value = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_normalize_task(training_info.get('task_type'))}_{_model_token(str(training_info['model']))}_{timestamp_value}_{image_size}px_{epochs}epochs_{batch}"
