from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smartrain.unified.domain.types import BackendType, ModelFormat, SourceKind, TaskType


@dataclass(frozen=True)
class UnifiedArtifactRef:
    id: str
    source_kind: SourceKind
    created_at: str
    task_type: TaskType
    backend_type: BackendType


@dataclass(frozen=True)
class UnifiedModelRef:
    model_id: str
    format: ModelFormat
    weights_path: str
    config_path: str | None
    labels_path: str | None
    provenance: dict[str, Any]
    task_type: TaskType
    backend_type: BackendType


@dataclass(frozen=True)
class UnifiedRunRef:
    run_id: str
    workspace: str
    dataset_ref: str | None
    training_ref: str | None
    test_refs: list[str] = field(default_factory=list)
    inference_refs: list[str] = field(default_factory=list)
    task_type: TaskType = "detection"
    backend_type: BackendType = "ultralytics"


@dataclass(frozen=True)
class UnifiedMetricsRef:
    namespace: str
    primary_metrics: dict[str, Any]
    secondary_metrics: dict[str, Any]
    raw_path: str
    producer: str
    task_type: TaskType


@dataclass(frozen=True)
class UnifiedPredictionRef:
    task_type: TaskType
    items_path: str
    schema_version: str
    producer: str
    count: int


@dataclass(frozen=True)
class UnifiedPayload:
    schema_version: str
    generated_at: str
    producer: str
    artifacts: list[UnifiedArtifactRef] = field(default_factory=list)
    models: list[UnifiedModelRef] = field(default_factory=list)
    runs: list[UnifiedRunRef] = field(default_factory=list)
    metrics: list[UnifiedMetricsRef] = field(default_factory=list)
    predictions: list[UnifiedPredictionRef] = field(default_factory=list)

