from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartrain.canonical_refs import canonical_target_from_run
from smartrain.domain.canonical.models import CanonicalModelRef, CanonicalPayload, CanonicalRunRef

from .normalizers import normalize_backend, normalize_path, normalize_task


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


class RunAdapter:
    def read(self, source_ref: str, options: dict[str, Any] | None = None) -> CanonicalPayload:
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
        task_type = normalize_task(ti.get("task_type") if isinstance(ti, dict) else None)
        provider = ti.get("provider", {}) if isinstance(ti, dict) else {}
        backend = normalize_backend(provider.get("id") if isinstance(provider, dict) else None)

        target = canonical_target_from_run(run_dir)
        model = CanonicalModelRef(
            model_id=target.source_id,
            format=target.model_path.suffix.lower().lstrip(".") or "pt",
            weights_path=normalize_path(str(target.model_path)),
            config_path=None,
            labels_path=None,
            provenance={"source_kind": "run", "source_ref": str(run_dir)},
            task_type=task_type,
            backend_type=backend,
        )
        run = CanonicalRunRef(
            run_id=run_dir.name,
            workspace=str(run_dir.parent.parent.parent) if len(run_dir.parts) >= 3 else str(run_dir.parent),
            dataset_ref=_dataset_ref_from_run_dir(run_dir, ti),
            training_ref=str(run_dir),
            task_type=task_type,
            backend_type=backend,
        )
        return CanonicalPayload(
            schema_version="2.0.0",
            generated_at="1970-01-01T00:00:00Z",
            producer="canonical.run_adapter",
            models=[model],
            runs=[run],
        )

