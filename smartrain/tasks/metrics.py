from __future__ import annotations

from typing import Any, Protocol

from smartrain.tasks.classification.adapter import normalize_classification_metrics
from smartrain.tasks.contracts import TASK_CLASSIFICATION, TASK_DETECTION, TASK_SEGMENTATION, TaskContext
from smartrain.tasks.detection.adapter import normalize_detection_metrics
from smartrain.tasks.segmentation.adapter import normalize_segmentation_metrics


class TaskMetricsAdapter(Protocol):
    def normalize(self, payload: dict[str, Any]) -> dict[str, float]:
        ...


class DetectionMetricsAdapter:
    def normalize(self, payload: dict[str, Any]) -> dict[str, float]:
        return normalize_detection_metrics(payload)


class ClassificationMetricsAdapter:
    def normalize(self, payload: dict[str, Any]) -> dict[str, float]:
        return normalize_classification_metrics(payload)


class SegmentationMetricsAdapter:
    def normalize(self, payload: dict[str, Any]) -> dict[str, float]:
        return normalize_segmentation_metrics(payload)


_ADAPTERS: dict[str, TaskMetricsAdapter] = {
    TASK_DETECTION: DetectionMetricsAdapter(),
    TASK_CLASSIFICATION: ClassificationMetricsAdapter(),
    TASK_SEGMENTATION: SegmentationMetricsAdapter(),
}


def resolve_task_metrics_adapter(task_type: str) -> TaskMetricsAdapter:
    normalized = TaskContext(task_type=task_type).normalized()
    return _ADAPTERS.get(normalized, _ADAPTERS[TASK_DETECTION])
