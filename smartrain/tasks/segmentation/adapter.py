from __future__ import annotations

from typing import Any

# Ultralytics SegmentMetrics / test CSV column names -> canonical SmarTrain keys.
SEGMENTATION_METRIC_ALIASES: dict[str, str] = {
    "mask_mAP50-95": "mask_mAP50-95",
    "mask_mAP50": "mask_mAP50",
    "Mask-F1": "Mask-F1",
    "Mask-P": "Mask-P",
    "Mask-R": "Mask-R",
    "mask_AP50-95": "mask_mAP50-95",
    "mask_AP50": "mask_mAP50",
    "mIoU": "mIoU",
    "Dice": "Dice",
    "metrics/mAP50-95(M)": "mask_mAP50-95",
    "metrics/mAP50(M)": "mask_mAP50",
    "metrics/precision(M)": "Mask-P",
    "metrics/recall(M)": "Mask-R",
    "metrics/mAP50-95(B)": "box_mAP50-95",
    "metrics/mAP50(B)": "box_mAP50",
    "metrics/precision(B)": "Box-P",
    "metrics/recall(B)": "Box-R",
    "mAP50-95": "box_mAP50-95",
    "mAP50": "box_mAP50",
    "Box-F1": "Box-F1",
    "Box-P": "Box-P",
    "Box-R": "Box-R",
}

_CANONICAL_ORDER = (
    "mask_mAP50-95",
    "mask_mAP50",
    "Mask-F1",
    "Mask-P",
    "Mask-R",
    "box_mAP50-95",
    "box_mAP50",
    "mIoU",
    "Dice",
)


def normalize_segmentation_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Normalize segmentation metrics from CSV rows or flat dicts."""
    mapped: dict[str, float] = {}
    for raw_key, value in payload.items():
        if value is None:
            continue
        key = str(raw_key).strip()
        canonical = SEGMENTATION_METRIC_ALIASES.get(key, key)
        try:
            mapped[canonical] = float(value)
        except (TypeError, ValueError):
            continue
    out: dict[str, float] = {}
    for key in _CANONICAL_ORDER:
        if key in mapped:
            out[key] = mapped[key]
    for key, val in mapped.items():
        if key not in out:
            out[key] = val
    return out
