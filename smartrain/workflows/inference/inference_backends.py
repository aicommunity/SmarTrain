from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from smartrain.backends.contracts import BackendCapabilities
from smartrain.backends.registry import CapabilityRegistry
from smartrain.tasks.contracts import KNOWN_TASKS
from smartrain.core.inference.ultralytics_prediction_extract import extract_task_outputs_from_ultralytics_preds
from smartrain.core.training.train_profile import task_to_metadata_task_type
from smartrain.core.runtime.ultralytics_ephemeral import ultralytics_sidecar_dir

from smartrain.external_providers.runner import run_external_infer


@dataclass
class BackendPrediction:
    task_type: str
    infer_only_ns: int
    stage_ns: dict[str, int]
    outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def detections(self) -> list[dict[str, Any]]:
        raw = self.outputs.get("detections")
        return raw if isinstance(raw, list) else []


class InferenceBackend:
    name: str

    def predict(
        self,
        image_source: Any,
        *,
        conf: float,
        imgsz: int,
        device: str | None,
        half: bool,
        task_type: str | None = None,
    ) -> BackendPrediction:
        raise NotImplementedError


class UltralyticsBackend(InferenceBackend):
    def __init__(self, weights_path: str, *, backend_name: str = "ultralytics") -> None:
        from ultralytics import YOLO

        self.name = backend_name
        self._model = YOLO(str(weights_path))
        self._predict_project = ultralytics_sidecar_dir(
            tempfile.gettempdir(), "smartrain_ultralytics_inference"
        )

    def predict(
        self,
        image_source: Any,
        *,
        conf: float,
        imgsz: int,
        device: str | None,
        half: bool,
        task_type: str | None = None,
    ) -> BackendPrediction:
        t0 = time.perf_counter_ns()
        preds = self._model.predict(
            source=image_source,
            conf=float(conf),
            imgsz=int(imgsz),
            verbose=False,
            device=str(device) if device is not None else None,
            half=bool(half),
            save=False,
            project=self._predict_project,
            name="inference-cli",
            exist_ok=True,
        )
        t1 = time.perf_counter_ns()
        resolved_task = task_to_metadata_task_type(task_type)
        return BackendPrediction(
            task_type=resolved_task,
            outputs=extract_task_outputs_from_ultralytics_preds(self._model, preds, task_type=resolved_task),
            infer_only_ns=int(t1 - t0),
            stage_ns={},
        )


class ExternalProviderBackend:
    def __init__(self, provider_id: str, repo_path: str, venv_path: str) -> None:
        self.name = "external"
        self.provider_id = provider_id
        self.repo_path = repo_path
        self.venv_path = venv_path

    def run_batch(
        self,
        *,
        model_path: str,
        source_path: str,
        conf: float,
        imgsz: int,
        device: str | None,
        task_type: str | None = None,
    ) -> int | dict[str, object]:
        return run_external_infer(
            self.provider_id,
            self.repo_path,
            self.venv_path,
            model_path=model_path,
            source_path=source_path,
            conf=float(conf),
            imgsz=int(imgsz),
            device=str(device) if device else None,
            task_type=str(task_type) if task_type else None,
        )


class InferenceBackendRegistry:
    def __init__(self) -> None:
        self._capabilities = CapabilityRegistry()
        self._capabilities.register(
            BackendCapabilities(
                backend="ultralytics",
                task_types=KNOWN_TASKS,
                model_formats=("pt", "onnx", "engine", "trt"),
                can_infer=True,
            )
        )

    def create_local_backend(self, *, model_format: str, model_path: str, task_type: str | None = None) -> InferenceBackend:
        fmt = str(model_format or "").strip().lower()
        resolved_task_type = task_to_metadata_task_type(task_type)
        caps = self._capabilities.resolve(task_type=resolved_task_type, model_format=fmt, require="infer")
        backend_id = str(caps.backend or "").strip().lower()
        if backend_id == "ultralytics":
            return UltralyticsBackend(model_path, backend_name=f"ultralytics:{fmt}")
        raise ValueError(
            "Unsupported local inference backend capability "
            f"{caps.backend!r} for model format {model_format!r}"
        )


