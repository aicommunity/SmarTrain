from __future__ import annotations

from smartrain.tasks.segmentation.adapter import normalize_segmentation_metrics


def test_normalize_segmentation_metrics_ultralytics_mask_columns() -> None:
    payload = {
        "metrics/mAP50-95(M)": 0.42,
        "metrics/mAP50(M)": 0.55,
        "metrics/precision(M)": 0.61,
        "metrics/recall(M)": 0.48,
    }
    out = normalize_segmentation_metrics(payload)
    assert out["mask_mAP50-95"] == 0.42
    assert out["mask_mAP50"] == 0.55
    assert out["Mask-P"] == 0.61
    assert out["Mask-R"] == 0.48


def test_normalize_segmentation_metrics_box_fallback() -> None:
    payload = {
        "metrics/mAP50-95(B)": 0.33,
        "metrics/mAP50(B)": 0.44,
        "Box-F1": 0.5,
    }
    out = normalize_segmentation_metrics(payload)
    assert out["box_mAP50-95"] == 0.33
    assert out["box_mAP50"] == 0.44
    assert out["Box-F1"] == 0.5


def test_normalize_segmentation_metrics_legacy_keys() -> None:
    payload = {"mIoU": 0.7, "Dice": 0.8, "mask_AP50": 0.6}
    out = normalize_segmentation_metrics(payload)
    assert out["mIoU"] == 0.7
    assert out["Dice"] == 0.8
    assert out["mask_mAP50"] == 0.6


def test_normalize_segmentation_metrics_skips_invalid() -> None:
    out = normalize_segmentation_metrics({"mask_mAP50": "n/a", "metrics/mAP50(M)": 0.1})
    assert out == {"mask_mAP50": 0.1}
