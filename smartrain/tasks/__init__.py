"""Task adapters and contracts."""

from smartrain.tasks.context import TaskExecutionContext
from smartrain.tasks.contracts import (
    KNOWN_TASKS,
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_SEGMENTATION,
    TaskTypeLabel,
    normalize_task_type,
)
from smartrain.tasks.metrics import resolve_task_metrics_adapter

__all__ = [
    "KNOWN_TASKS",
    "TASK_DETECTION",
    "TASK_CLASSIFICATION",
    "TASK_SEGMENTATION",
    "TaskTypeLabel",
    "normalize_task_type",
    "TaskExecutionContext",
    "resolve_task_metrics_adapter",
]

