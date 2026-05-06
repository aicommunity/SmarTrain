from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smartrain.backends.contracts import BackendCapabilities, BackendExecutionResult, InferenceBackend
from smartrain.workflows.inference.inference_backends import InferenceBackendRegistry
from smartrain.tasks.contracts import KNOWN_TASKS
from smartrain.core.training.train_profile import task_to_metadata_task_type


@dataclass(frozen=True)
class UltralyticsAdapter:
    """
    Reference backend adapter for Phase D.

    This adapter is intentionally thin: runtime train/test paths remain in
    existing service modules, while inference path is wired through this class.
    """

    backend_id: str = "ultralytics"
    capabilities: BackendCapabilities = BackendCapabilities(
        backend="ultralytics",
        task_types=KNOWN_TASKS,
        model_formats=("pt", "onnx", "engine", "trt"),
        can_train=True,
        can_test=True,
        can_infer=True,
    )

    def create_inference_backend(self, *, model_format: str, model_path: str, task_type: str | None = None) -> InferenceBackend:
        registry = InferenceBackendRegistry()
        return registry.create_local_backend(model_format=model_format, model_path=model_path, task_type=task_type)

    def infer(self, *, request: Any) -> BackendExecutionResult:
        model_format = str(getattr(request, "model_format", "")).strip().lower()
        model_path = str(getattr(request, "model_path", ""))
        task_type = task_to_metadata_task_type(getattr(request, "task_type", None))
        if not model_format or not model_path:
            return BackendExecutionResult(
                success=False,
                backend=self.backend_id,
                task_type=task_type,
                model_format=model_format or "unknown",
                error="model_format/model_path are required for UltralyticsAdapter.infer",
            )
        self.create_inference_backend(model_format=model_format, model_path=model_path, task_type=task_type)
        return BackendExecutionResult(
            success=True,
            backend=self.backend_id,
            task_type=task_type,
            model_format=model_format,
            metadata={"model_path": model_path},
        )
