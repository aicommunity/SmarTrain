from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartrain.run_model_contract.refs import unified_target_from_model_dir
from smartrain.run_model_contract.domain.models import UnifiedModelRef, UnifiedPayload

from .normalizers import normalize_backend, normalize_path, normalize_task


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _infer_task_from_model_name(model_name: str) -> str | None:
    name = str(model_name or "").strip().lower()
    if "-cls" in name or "class" in name:
        return "classification"
    if "-seg" in name or "segment" in name:
        return "segmentation"
    return None


def _infer_backend_from_format(model_format: str) -> str:
    fmt = str(model_format or "").strip().lower()
    if fmt == "onnx":
        return "onnxruntime"
    if fmt in {"engine", "trt"}:
        return "tensorrt"
    return "ultralytics"


class ModelAdapter:
    def read(self, source_ref: str, options: dict[str, Any] | None = None) -> UnifiedPayload:
        model_dir = Path(source_ref).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        target = unified_target_from_model_dir(model_dir)
        model_format = target.model_path.suffix.lower().lstrip(".") or "pt"
        manifest = _load_json(model_dir / "model_manifest.json")
        training_metadata = _load_json(model_dir / "training_metadata.json")
        training_info = (
            training_metadata.get("training_info", {})
            if isinstance(training_metadata.get("training_info"), dict)
            else {}
        )
        provider = training_info.get("provider", {}) if isinstance(training_info, dict) else {}
        raw_task = manifest.get("task_type") or training_info.get("task_type")
        utrain = training_info.get("ultralytics_train") if isinstance(training_info, dict) else None
        if not str(raw_task or "").strip() and isinstance(utrain, dict) and utrain.get("task"):
            raw_task = utrain.get("task")
        raw_backend = manifest.get("backend_type") or (
            provider.get("id") if isinstance(provider, dict) else None
        )

        inferred_task = _infer_task_from_model_name(target.model_path.name)
        if str(raw_task or "").strip():
            task_type = normalize_task(str(raw_task))
            task_resolution = "metadata"
        elif inferred_task:
            task_type = normalize_task(inferred_task)
            task_resolution = "name_hint"
        else:
            raise ValueError(
                f"Cannot resolve task_type for model directory {model_dir}: "
                "set task_type in model_manifest.json or training_metadata.json, "
                "or use a weights filename with -cls/-seg hint."
            )

        if str(raw_backend or "").strip():
            backend_type = normalize_backend(str(raw_backend))
            backend_resolution = "metadata"
        else:
            backend_type = _infer_backend_from_format(model_format)
            backend_resolution = "format_hint"

        model = UnifiedModelRef(
            model_id=target.source_id,
            format=model_format,
            weights_path=normalize_path(str(target.model_path)),
            config_path=None,
            labels_path=None,
            provenance={
                "source_kind": "model",
                "source_ref": str(model_dir),
                "task_resolution": task_resolution,
                "backend_resolution": backend_resolution,
            },
            task_type=task_type,
            backend_type=backend_type,
        )
        return UnifiedPayload(
            schema_version="2.0.0",
            generated_at="1970-01-01T00:00:00Z",
            producer="canonical.model_adapter",
            models=[model],
        )

