from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smartrain.backends.contracts import BackendExecutionResult, InferenceBackend
from smartrain.inference_backends import ExternalProviderBackend


@dataclass(frozen=True)
class ExternalProviderAdapter:
    """
    Adapter wrapper for external-provider inference execution.

    Normalizes external provider execution entrypoint to backend contract layer.
    """

    provider_id: str
    repo_path: str
    venv_path: str

    @property
    def backend_id(self) -> str:
        return f"external:{self.provider_id}"

    def create_runtime_backend(self) -> ExternalProviderBackend:
        return ExternalProviderBackend(self.provider_id, self.repo_path, self.venv_path)

    def run_batch(
        self,
        *,
        model_path: str,
        source_path: str,
        conf: float,
        imgsz: int,
        device: str | None,
    ) -> int:
        backend = self.create_runtime_backend()
        return backend.run_batch(
            model_path=model_path,
            source_path=source_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )

    def infer(self, *, request: Any) -> BackendExecutionResult:
        model_path = str(getattr(request, "model_path", "") or "")
        source_path = str(getattr(request, "source_path", "") or "")
        if not model_path or not source_path:
            return BackendExecutionResult(
                success=False,
                backend=self.backend_id,
                task_type="detection",
                model_format=str(getattr(request, "model_format", "") or "external"),
                error="model_path/source_path are required for ExternalProviderAdapter.infer",
            )
        rc = self.run_batch(
            model_path=model_path,
            source_path=source_path,
            conf=float(getattr(request, "conf", 0.25)),
            imgsz=int(getattr(request, "imgsz", 640)),
            device=str(getattr(request, "device", "") or "") or None,
        )
        return BackendExecutionResult(
            success=(int(rc) == 0),
            backend=self.backend_id,
            task_type="detection",
            model_format=str(getattr(request, "model_format", "") or "external"),
            error=None if int(rc) == 0 else f"external inference returned code {int(rc)}",
            metadata={"return_code": int(rc)},
        )
