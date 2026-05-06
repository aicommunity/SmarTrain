from __future__ import annotations

from pathlib import Path
from typing import Any

from smartrain.canonical.refs import canonical_target_from_model_dir
from smartrain.domain.canonical.models import CanonicalModelRef, CanonicalPayload

from .normalizers import normalize_path


class ModelAdapter:
    def read(self, source_ref: str, options: dict[str, Any] | None = None) -> CanonicalPayload:
        model_dir = Path(source_ref).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        target = canonical_target_from_model_dir(model_dir)
        model = CanonicalModelRef(
            model_id=target.source_id,
            format=target.model_path.suffix.lower().lstrip(".") or "pt",
            weights_path=normalize_path(str(target.model_path)),
            config_path=None,
            labels_path=None,
            provenance={"source_kind": "model", "source_ref": str(model_dir)},
            task_type="detection",
            backend_type="ultralytics",
        )
        return CanonicalPayload(
            schema_version="2.0.0",
            generated_at="1970-01-01T00:00:00Z",
            producer="canonical.model_adapter",
            models=[model],
        )

