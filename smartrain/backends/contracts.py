from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


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


@dataclass(frozen=True)
class BackendExecutionResult:
    """
    Common result envelope for backend contract methods.

    This keeps train/test/inference adapters on the same success/error/report
    shape regardless of concrete backend implementation.
    """

    success: bool
    backend: str
    task_type: str
    model_format: str
    error: str | None = None
    artifacts: Mapping[str, str] | None = None
    metrics: Mapping[str, float] | None = None
    metadata: Mapping[str, Any] | None = None


@runtime_checkable
class TrainBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities

    def train(self, *, request: Any) -> BackendExecutionResult:
        ...


@runtime_checkable
class TestBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities

    def test(self, *, request: Any) -> BackendExecutionResult:
        ...


@runtime_checkable
class InferenceBackend(Protocol):
    backend_id: str
    capabilities: BackendCapabilities

    def infer(self, *, request: Any) -> BackendExecutionResult:
        ...

