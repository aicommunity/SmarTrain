from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartrain.core.runtime.path_portable import posix_relpath, store_path_under_workspace
from smartrain.run_model_contract.refs import unified_target_from_run
from smartrain.run_model_contract.domain.models import UnifiedModelRef, UnifiedPayload, UnifiedRunRef

from .normalizers import normalize_backend, normalize_task


def _dataset_ref_from_run_dir(run_dir: Path, ti: dict[str, Any]) -> str | None:
    dataset = ti.get("dataset") if isinstance(ti, dict) else None
    if isinstance(dataset, dict):
        name = dataset.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    parts = [str(p).strip() for p in run_dir.parts]
    lower_parts = [p.lower() for p in parts]
    if "runs" in lower_parts:
        idx = lower_parts.index("runs")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate:
                return candidate
    parent_name = run_dir.parent.name.strip()
    return parent_name or None


def _infer_task_from_target_name(target_name: str) -> str:
    name = str(target_name or "").strip().lower()
    if "-cls" in name or "class" in name:
        return "classification"
    if "-seg" in name or "segment" in name:
        return "segmentation"
    return "detection"


def _infer_backend_from_format(model_format: str) -> str:
    fmt = str(model_format or "").strip().lower()
    if fmt == "onnx":
        return "onnxruntime"
    if fmt in {"engine", "trt"}:
        return "tensorrt"
    return "ultralytics"


class RunAdapter:
    def read(self, source_ref: str, options: dict[str, Any] | None = None) -> UnifiedPayload:
        run_dir = Path(source_ref).expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        md = {}
        md_path = run_dir / "training_metadata.json"
        if md_path.is_file():
            try:
                md = json.loads(md_path.read_text(encoding="utf-8"))
            except Exception:
                md = {}
        ti = md.get("training_info", {}) if isinstance(md, dict) else {}

        target = unified_target_from_run(run_dir)
        model_format = target.model_path.suffix.lower().lstrip(".") or "pt"
        raw_task = ti.get("task_type") if isinstance(ti, dict) else None
        task_type = normalize_task(raw_task)
        if not str(raw_task or "").strip():
            task_type = _infer_task_from_target_name(target.model_path.name)
        provider = ti.get("provider", {}) if isinstance(ti, dict) else {}
        raw_backend = provider.get("id") if isinstance(provider, dict) else None
        backend = normalize_backend(raw_backend)
        if not str(raw_backend or "").strip():
            backend = _infer_backend_from_format(model_format)

        model = UnifiedModelRef(
            model_id=target.source_id,
            format=model_format,
            weights_path=posix_relpath(str(target.model_path), str(run_dir)),
            config_path=None,
            labels_path=None,
            provenance={"source_kind": "run", "source_ref": posix_relpath(str(run_dir), str(run_dir.parent.parent.parent)) if len(run_dir.parts) >= 3 else run_dir.name},
            task_type=task_type,
            backend_type=backend,
        )
        ws_guess = str(run_dir.parent.parent.parent) if len(run_dir.parts) >= 3 else str(run_dir.parent)
        run = UnifiedRunRef(
            run_id=run_dir.name,
            workspace=".",
            dataset_ref=_dataset_ref_from_run_dir(run_dir, ti),
            training_ref=store_path_under_workspace(ws_guess, str(run_dir)),
            task_type=task_type,
            backend_type=backend,
        )
        return UnifiedPayload(
            schema_version="2.0.0",
            generated_at="1970-01-01T00:00:00Z",
            producer="canonical.run_adapter",
            models=[model],
            runs=[run],
        )

