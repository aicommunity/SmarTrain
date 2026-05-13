from __future__ import annotations

from pathlib import Path


def normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def normalize_task(task: str | None) -> str:
    value = str(task or "").strip().lower()
    if value in {"detect", "detection", ""}:
        return "detection"
    if value in {"classify", "classification"}:
        return "classification"
    if value in {"segment", "segmentation"}:
        return "segmentation"
    return "detection"


def normalize_backend(backend: str | None) -> str:
    value = str(backend or "").strip().lower()
    if value:
        return value
    return "ultralytics"

