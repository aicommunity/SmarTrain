from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smartrain.backends.contracts import BackendExecutionResult, InferenceBackend
from smartrain.inference_backends import ExternalProviderBackend
from smartrain.external_providers.runner import run_external_train


@dataclass(frozen=True)
class ExternalProviderAdapter:
    """
    Adapter wrapper for external-provider inference execution.

    Normalizes external provider execution entrypoint to backend contract layer.
    """

    provider_id: str
    repo_path: str
    venv_path: str
    train_runner: Callable[..., int] | None = None
    infer_runner: Callable[..., int] | None = None

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
        target_dir: str | None = None,
        run_name: str | None = None,
    ) -> int:
        if self.infer_runner is not None:
            return int(
                self.infer_runner(
                    self.provider_id,
                    self.repo_path,
                    self.venv_path,
                    model_path=model_path,
                    source_path=source_path,
                    conf=conf,
                    imgsz=imgsz,
                    device=device,
                    target_dir=target_dir,
                    run_name=run_name,
                )
            )
        backend = self.create_runtime_backend()
        return backend.run_batch(
            model_path=model_path,
            source_path=source_path,
            conf=conf,
            imgsz=imgsz,
            device=device,
        )

    def run_train(
        self,
        *,
        dataset_path: str,
        model: str,
        epochs: int,
        batch: int,
        imgsz: int,
        device: str | None = None,
        target_dir: str | None = None,
        run_name: str | None = None,
    ) -> int:
        if self.train_runner is not None:
            return int(
                self.train_runner(
                    self.provider_id,
                    self.repo_path,
                    self.venv_path,
                    dataset_path=dataset_path,
                    model=model,
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    device=device,
                    target_dir=target_dir,
                    run_name=run_name,
                )
            )
        return run_external_train(
            self.provider_id,
            self.repo_path,
            self.venv_path,
            dataset_path=dataset_path,
            model=model,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            target_dir=target_dir,
            run_name=run_name,
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
