from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smartrain.domain.canonical.types import BackendType, ModelFormat, SourceKind, TaskType


@dataclass(frozen=True)
class CanonicalArtifactRef:
    id: str
    source_kind: SourceKind
    created_at: str
    task_type: TaskType
    backend_type: BackendType


@dataclass(frozen=True)
class CanonicalModelRef:
    model_id: str
    format: ModelFormat
    weights_path: str
    config_path: str | None
    labels_path: str | None
    provenance: dict[str, Any]
    task_type: TaskType
    backend_type: BackendType


@dataclass(frozen=True)
class CanonicalRunRef:
    run_id: str
    workspace: str
    dataset_ref: str | None
    training_ref: str | None
    test_refs: list[str] = field(default_factory=list)
    inference_refs: list[str] = field(default_factory=list)
    task_type: TaskType = "detection"
    backend_type: BackendType = "ultralytics"


@dataclass(frozen=True)
class CanonicalMetricsRef:
    namespace: str
    primary_metrics: dict[str, Any]
    secondary_metrics: dict[str, Any]
    raw_path: str
    producer: str
    task_type: TaskType


@dataclass(frozen=True)
class CanonicalPredictionRef:
    task_type: TaskType
    items_path: str
    schema_version: str
    producer: str
    count: int


@dataclass(frozen=True)
class CanonicalPayload:
    schema_version: str
    generated_at: str
    producer: str
    artifacts: list[CanonicalArtifactRef] = field(default_factory=list)
    models: list[CanonicalModelRef] = field(default_factory=list)
    runs: list[CanonicalRunRef] = field(default_factory=list)
    metrics: list[CanonicalMetricsRef] = field(default_factory=list)
    predictions: list[CanonicalPredictionRef] = field(default_factory=list)

