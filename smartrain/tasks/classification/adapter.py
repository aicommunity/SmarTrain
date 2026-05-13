from __future__ import annotations

from typing import Any


def normalize_classification_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("top1", "top5", "f1_macro", "precision_macro", "recall_macro"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out

