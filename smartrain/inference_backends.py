from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Any

from smartrain.backends.contracts import BackendCapabilities
from smartrain.backends.registry import CapabilityRegistry
from smartrain.tasks.contracts import TASK_DETECTION
from smartrain.ultralytics_ephemeral import ultralytics_sidecar_dir

from smartrain.external_providers.runner import run_external_infer


@dataclass
class BackendPrediction:
    detections: list[dict[str, Any]]
    infer_only_ns: int
    stage_ns: dict[str, int]


class InferenceBackend:
    name: str

    def predict(self, image_source: Any, *, conf: float, imgsz: int, device: str | None, half: bool) -> BackendPrediction:
        raise NotImplementedError


class UltralyticsBackend(InferenceBackend):
    def __init__(self, weights_path: str, *, backend_name: str = "ultralytics") -> None:
        from ultralytics import YOLO

        self.name = backend_name
        self._model = YOLO(str(weights_path))
        self._predict_project = ultralytics_sidecar_dir(
            tempfile.gettempdir(), "smartrain_ultralytics_inference"
        )

    def predict(self, image_source: Any, *, conf: float, imgsz: int, device: str | None, half: bool) -> BackendPrediction:
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
        return BackendPrediction(detections=_extract_detections(self._model, preds), infer_only_ns=int(t1 - t0), stage_ns={})


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
    ) -> int:
        return run_external_infer(
            self.provider_id,
            self.repo_path,
            self.venv_path,
            model_path=model_path,
            source_path=source_path,
            conf=float(conf),
            imgsz=int(imgsz),
            device=str(device) if device else None,
        )


class InferenceBackendRegistry:
    def __init__(self) -> None:
        self._capabilities = CapabilityRegistry()
        self._capabilities.register(
            BackendCapabilities(
                backend="ultralytics",
                task_types=(TASK_DETECTION,),
                model_formats=("pt", "onnx", "engine", "trt"),
                can_infer=True,
            )
        )

    def create_local_backend(self, *, model_format: str, model_path: str) -> InferenceBackend:
        fmt = str(model_format or "").strip().lower()
        self._capabilities.resolve(task_type=TASK_DETECTION, model_format=fmt, require="infer")
        if fmt in {"pt", "onnx", "engine", "trt"}:
            return UltralyticsBackend(model_path, backend_name=f"ultralytics:{fmt}")
        raise ValueError(f"Unsupported model format for local inference backend: {model_format}")


def _extract_detections(model: Any, preds: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not preds:
        return rows
    r = preds[0]
    boxes_obj = getattr(r, "boxes", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return rows
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    names = getattr(model, "names", {})
    for i in range(len(xyxy)):
        cls_idx = int(cls[i])
        class_name = str(names.get(cls_idx, names.get(str(cls_idx), cls_idx))) if isinstance(names, dict) else str(cls_idx)
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        rows.append(
            {
                "bbox_roi_xyxy": [x1, y1, x2, y2],
                "class_index": cls_idx,
                "class_name": class_name,
                "confidence": float(confs[i]),
            }
        )
    return rows
