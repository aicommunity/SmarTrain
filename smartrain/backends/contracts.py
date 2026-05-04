from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    task_types: tuple[str, ...]
    model_formats: tuple[str, ...]
    can_train: bool = False
    can_test: bool = False
    can_infer: bool = False

    def supports(self, *, task_type: str, model_format: str) -> bool:
        task = str(task_type or "").strip().lower()
        fmt = str(model_format or "").strip().lower()
        return task in self.task_types and fmt in self.model_formats

