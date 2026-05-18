from __future__ import annotations

from dataclasses import dataclass

TASK_DETECTION = "detection"
TASK_CLASSIFICATION = "classification"
TASK_SEGMENTATION = "segmentation"
KNOWN_TASKS = (TASK_DETECTION, TASK_CLASSIFICATION, TASK_SEGMENTATION)


def normalize_task_type(task_type: str) -> str:
    value = str(task_type or "").strip().lower()
    if value not in KNOWN_TASKS:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    return value


@dataclass(frozen=True)
class TaskTypeLabel:
    task_type: str

    def normalized(self) -> str:
        return normalize_task_type(self.task_type)
