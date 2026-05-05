from __future__ import annotations

import pytest

from smartrain.tasks.contracts import TaskContext
from smartrain.tasks.context import TaskExecutionContext
from smartrain.tasks.metrics import resolve_task_metrics_adapter
from smartrain.tasks.classification.adapter import normalize_classification_metrics
from smartrain.tasks.detection.adapter import normalize_detection_metrics
from smartrain.tasks.segmentation.adapter import normalize_segmentation_metrics


def test_task_context_normalization() -> None:
    ctx = TaskContext(task_type="Detection")
    assert ctx.normalized() == "detection"


def test_task_context_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        TaskContext(task_type="pose").normalized()


def test_detection_metrics_adapter_filters_numeric_payload() -> None:
    payload = {"mAP50-95": "0.5", "Box-F1": 0.7, "invalid": "x"}
    out = normalize_detection_metrics(payload)
    assert out["mAP50-95"] == 0.5
    assert out["Box-F1"] == 0.7
    assert "invalid" not in out


def test_classification_and_segmentation_metrics_adapters_filter_numeric_payload() -> None:
    cls_payload = {"top1": "0.7", "top5": 0.9, "bad": "x"}
    seg_payload = {"mIoU": "0.5", "Dice": 0.8, "bad": "x"}
    cls_out = normalize_classification_metrics(cls_payload)
    seg_out = normalize_segmentation_metrics(seg_payload)
    assert cls_out["top1"] == 0.7
    assert cls_out["top5"] == 0.9
    assert "bad" not in cls_out
    assert seg_out["mIoU"] == 0.5
    assert seg_out["Dice"] == 0.8
    assert "bad" not in seg_out


def test_task_execution_context_builds_metrics_namespace() -> None:
    ctx = TaskExecutionContext(task_type="Segmentation")
    assert ctx.metrics_namespace(format_name="onnx") == "segmentation/test_onnx"


def test_resolve_task_metrics_adapter_returns_task_specific_adapter() -> None:
    cls_adapter = resolve_task_metrics_adapter("classification")
    seg_adapter = resolve_task_metrics_adapter("segmentation")
    assert cls_adapter.normalize({"top1": "0.88"}).get("top1") == 0.88
    assert seg_adapter.normalize({"mIoU": "0.44"}).get("mIoU") == 0.44

