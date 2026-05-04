from __future__ import annotations

from dataclasses import dataclass

TASK_DETECTION = "detection"
TASK_CLASSIFICATION = "classification"
TASK_SEGMENTATION = "segmentation"
KNOWN_TASKS = (TASK_DETECTION, TASK_CLASSIFICATION, TASK_SEGMENTATION)


@dataclass(frozen=True)
class TaskContext:
    task_type: str

    def normalized(self) -> str:
        value = str(self.task_type or "").strip().lower()
        if value not in KNOWN_TASKS:
            raise ValueError(f"Unsupported task_type: {self.task_type!r}")
        return value

