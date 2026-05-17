from __future__ import annotations

from dataclasses import dataclass

from smartrain.unified.domain.types import BackendType, ModelFormat, TaskType


@dataclass(frozen=True)
class UnifiedIdentity:
    """Resolved task/backend identity for a canonical target (PR 6.5 gateway surface)."""

    source_kind: str
    source_ref: str
    task_type: TaskType
    backend_type: BackendType
    model_format: ModelFormat
    model_id: str
    run_id: str | None
    dataset_ref: str | None
