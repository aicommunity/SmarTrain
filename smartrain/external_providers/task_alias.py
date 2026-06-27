from __future__ import annotations


def ultralytics_task_alias(task_type: str | None) -> str:
    t = str(task_type or "detection").strip().lower()
    if t in {"segmentation", "segment", "seg"}:
        return "segment"
    if t in {"classification", "classify", "cls"}:
        return "classify"
    return "detect"
