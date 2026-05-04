from __future__ import annotations

from pathlib import Path


def infer_source_kind(source_ref: str) -> str:
    p = Path(source_ref).expanduser()
    parts_lower = [x.lower() for x in p.parts]
    name = p.name.lower()
    if "run" in parts_lower or "runs" in parts_lower or (p / "training_metadata.json").is_file():
        return "run"
    if "model" in parts_lower or "models" in parts_lower:
        return "model"
    if name in {"runs", "models"}:
        return "run" if name == "runs" else "model"
    return "model"

