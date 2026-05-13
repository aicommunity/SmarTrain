from __future__ import annotations

from typing import Any


def normalize_detection_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    aliases = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")
    for key in aliases:
        value = payload.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out

