from __future__ import annotations

import pytest

from smartrain.tasks.contracts import TaskContext
from smartrain.tasks.detection.adapter import normalize_detection_metrics


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

