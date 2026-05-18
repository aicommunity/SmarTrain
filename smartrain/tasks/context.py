from __future__ import annotations

from dataclasses import dataclass

from smartrain.tasks.contracts import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_SEGMENTATION,
    normalize_task_type,
)


@dataclass(frozen=True)
class TaskExecutionContext:
    task_type: str
    stage: str = "test"
    split: str = "test"

    def normalized_task_type(self) -> str:
        return normalize_task_type(self.task_type)

    def metrics_namespace(self, *, format_name: str) -> str:
        task = self.normalized_task_type()
        fmt = str(format_name or "").strip().lower()
        if task == TASK_DETECTION:
            return f"{TASK_DETECTION}/{self.stage}_{fmt}"
        if task == TASK_CLASSIFICATION:
            return f"{TASK_CLASSIFICATION}/{self.stage}_{fmt}"
        if task == TASK_SEGMENTATION:
            return f"{TASK_SEGMENTATION}/{self.stage}_{fmt}"
        return f"{task}/{self.stage}_{fmt}"
