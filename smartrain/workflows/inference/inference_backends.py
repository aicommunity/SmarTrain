from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from smartrain.backends.contracts import BackendCapabilities
from smartrain.backends.registry import CapabilityRegistry
from smartrain.tasks.contracts import KNOWN_TASKS
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
            outputs=_extract_task_outputs(self._model, preds, task_type=resolved_task),
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


def _extract_classification(model: Any, preds: Any) -> dict[str, Any] | None:
    if not preds:
        return None
    r = preds[0]
    probs = getattr(r, "probs", None)
    if probs is None:
        return None
    names = getattr(model, "names", {})
    top1_idx = getattr(probs, "top1", None)
    top1_conf = getattr(probs, "top1conf", None)
    top5 = getattr(probs, "top5", None)
    top5conf = getattr(probs, "top5conf", None)

    def _to_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            if hasattr(v, "item"):
                return float(v.item())
            return float(v)
        except Exception:
            return None

    def _class_name(idx: int) -> str:
        if isinstance(names, dict):
            return str(names.get(idx, names.get(str(idx), idx)))
        return str(idx)

    top_k: list[dict[str, Any]] = []
    if isinstance(top5, (list, tuple)):
        for i, cls_idx_raw in enumerate(top5):
            try:
                cls_idx = int(cls_idx_raw)
            except Exception:
                continue
            conf_val = None
            if isinstance(top5conf, (list, tuple)) and i < len(top5conf):
                conf_val = _to_float(top5conf[i])
            top_k.append(
                {
                    "class_index": cls_idx,
                    "class_name": _class_name(cls_idx),
                    "confidence": conf_val,
                }
            )
    if top1_idx is None:
        return {"top_k": top_k}
    try:
        top1_i = int(top1_idx)
    except Exception:
        return {"top_k": top_k}
    return {
        "top1": {
            "class_index": top1_i,
            "class_name": _class_name(top1_i),
            "confidence": _to_float(top1_conf),
        },
        "top_k": top_k,
    }


def _extract_segments(model: Any, preds: Any) -> list[dict[str, Any]]:
    if not preds:
        return []
    r = preds[0]
    boxes_obj = getattr(r, "boxes", None)
    masks_obj = getattr(r, "masks", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy()
    confs = boxes_obj.conf.cpu().numpy()
    polygons = getattr(masks_obj, "xy", None) if masks_obj is not None else None
    names = getattr(model, "names", {})
    rows: list[dict[str, Any]] = []
    for i in range(len(xyxy)):
        cls_idx = int(cls[i])
        class_name = str(names.get(cls_idx, names.get(str(cls_idx), cls_idx))) if isinstance(names, dict) else str(cls_idx)
        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
        polygon_xy: list[list[float]] = []
        if isinstance(polygons, (list, tuple)) and i < len(polygons):
            poly = polygons[i]
            if hasattr(poly, "tolist"):
                poly = poly.tolist()
            if isinstance(poly, list):
                for point in poly:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        try:
                            polygon_xy.append([float(point[0]), float(point[1])])
                        except Exception:
                            continue
        rows.append(
            {
                "bbox_roi_xyxy": [x1, y1, x2, y2],
                "class_index": cls_idx,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "polygon_roi_xy": polygon_xy,
            }
        )
    return rows


def _extract_task_outputs(model: Any, preds: Any, *, task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        classification = _extract_classification(model, preds)
        return {"classification": classification or {}}
    if task_type == "segmentation":
        segments = _extract_segments(model, preds)
        return {"segments": segments}
    return {"detections": _extract_detections(model, preds)}
