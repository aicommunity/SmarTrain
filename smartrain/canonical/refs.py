from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from smartrain.core.runtime.run_artifacts import canonical_run_model_path, materialize_canonical_run_model, scan_run_models

SourceKind = Literal["runs", "models", "weights"]
SUPPORTED_INFERENCE_EXTS = {".pt", ".onnx", ".engine", ".trt"}


@dataclass(frozen=True)
class CanonicalModelTarget:
    source_kind: SourceKind
    source_id: str
    model_path: Path


def _pick_preferred(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    ext_rank = {".pt": 4, ".onnx": 3, ".engine": 2, ".trt": 1}
    return max(paths, key=lambda p: (ext_rank.get(p.suffix.lower(), 0), p.stat().st_mtime))


def canonical_target_from_run(run_dir: Path) -> CanonicalModelTarget:
    candidates: list[Path] = []
    for rec in scan_run_models(str(run_dir)):
        path = Path(str(rec.get("path") or "")).expanduser()
        if path.is_file() and path.suffix.lower() in SUPPORTED_INFERENCE_EXTS:
            candidates.append(path.resolve())
    if not candidates:
        best = Path(canonical_run_model_path(str(run_dir), ".pt"))
        if not best.is_file():
            materialized = materialize_canonical_run_model(str(run_dir), ext=".pt", move=True, normalize_metadata=True)
            if materialized is not None:
                best = Path(materialized)
        if best.is_file():
            candidates.append(best.resolve())
    picked = _pick_preferred(candidates)
    if picked is None:
        raise FileNotFoundError(f"run model not found in run: {run_dir}")
    return CanonicalModelTarget(source_kind="runs", source_id=run_dir.name, model_path=picked)


def canonical_target_from_model_dir(model_dir: Path) -> CanonicalModelTarget:
    files = sorted(p for p in model_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_INFERENCE_EXTS)
    picked = _pick_preferred(files)
    if picked is None:
        raise FileNotFoundError(f"No supported model files found in: {model_dir}")
    return CanonicalModelTarget(source_kind="models", source_id=model_dir.name, model_path=picked.resolve())

