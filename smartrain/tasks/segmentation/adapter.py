from __future__ import annotations

from typing import Any


def normalize_segmentation_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("mIoU", "Dice", "mask_AP50", "mask_AP50-95"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out

