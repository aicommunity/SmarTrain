from __future__ import annotations

import json
import os
from typing import Any, Iterable

from smartrain.tasks.contracts import (
    TASK_CLASSIFICATION,
    TASK_DETECTION,
    TASK_SEGMENTATION,
    normalize_task_type,
)

METRIC_AGG_COLUMNS_BY_TASK: dict[str, tuple[str, ...]] = {
    TASK_DETECTION: ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R"),
    TASK_SEGMENTATION: ("mask_mAP50-95", "mask_mAP50", "Mask-F1", "Mask-P", "Mask-R"),
    TASK_CLASSIFICATION: ("top1_acc", "top5_acc"),
}

# Fallback columns when task-specific mask metrics are absent in CSV.
SEGMENTATION_BOX_FALLBACK_COLUMNS: tuple[str, ...] = ("mAP50-95", "mAP50", "Box-F1", "Box-P", "Box-R")


def metric_agg_columns(task_type: str | None) -> tuple[str, ...]:
    raw = str(task_type or TASK_DETECTION).strip().lower()
    try:
        normalized = normalize_task_type(raw)
    except ValueError:
        if raw in {"segment", "seg"}:
            normalized = TASK_SEGMENTATION
        elif raw in {"detect", "det"}:
            normalized = TASK_DETECTION
        elif raw in {"classify", "cls"}:
            normalized = TASK_CLASSIFICATION
        else:
            normalized = TASK_DETECTION
    return METRIC_AGG_COLUMNS_BY_TASK.get(normalized, METRIC_AGG_COLUMNS_BY_TASK[TASK_DETECTION])


def metric_agg_columns_with_fallback(task_type: str | None, available: set[str]) -> tuple[str, ...]:
    primary = metric_agg_columns(task_type)
    picked = tuple(c for c in primary if c in available)
    if picked:
        return picked
    try:
        normalized = normalize_task_type(str(task_type or TASK_DETECTION))
    except ValueError:
        normalized = TASK_DETECTION
    if normalized == TASK_SEGMENTATION:
        return tuple(c for c in SEGMENTATION_BOX_FALLBACK_COLUMNS if c in available) or primary
    return primary


def read_run_task_type(run_dir: str) -> str:
    metadata_path = os.path.join(run_dir, "training_metadata.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return TASK_DETECTION
    if not isinstance(payload, dict):
        return TASK_DETECTION
    ti = payload.get("training_info")
    if isinstance(ti, dict):
        raw = ti.get("task_type") or ti.get("task")
        if raw:
            return str(raw).strip().lower()
    return TASK_DETECTION


def metric_fields_from_row(
    metric_row: dict[str, Any],
    task_type: str | None,
    *,
    nullify: bool = False,
) -> dict[str, Any]:
    available = {str(k) for k in metric_row.keys()}
    cols = metric_agg_columns_with_fallback(task_type, available)
    if not cols:
        cols = metric_agg_columns(task_type)
    return {col: (None if nullify else metric_row.get(col)) for col in cols}


def detect_task_type_from_columns(columns: Iterable[str]) -> str | None:
    normalized = {str(c).strip() for c in columns}
    stripped = {c[5:] if c.startswith("test_") else c for c in normalized}
    if any(c.startswith("mask_") or c.startswith("Mask-") for c in stripped):
        return TASK_SEGMENTATION
    if any(c in {"top1_acc", "top5_acc", "top1", "top5"} for c in stripped):
        return TASK_CLASSIFICATION
    if any(c.startswith("delta_mask_") or c.startswith("delta_Mask-") for c in normalized):
        return TASK_SEGMENTATION
    return None


def primary_quality_columns(task_type: str | None = None) -> tuple[str, ...]:
    return metric_agg_columns(task_type)[:3]


def compare_delta_column_names() -> tuple[str, ...]:
    out: list[str] = []
    for task in (TASK_DETECTION, TASK_SEGMENTATION):
        for col in metric_agg_columns(task):
            name = f"delta_{col}"
            if name not in out:
                out.append(name)
    return tuple(out)


def runs_summary_test_column_names() -> tuple[str, ...]:
    out: list[str] = []
    for task in (TASK_DETECTION, TASK_SEGMENTATION, TASK_CLASSIFICATION):
        for col in metric_agg_columns(task):
            name = f"test_{col}"
            if name not in out:
                out.append(name)
    return tuple(out)


def format_metrics_table_column_names() -> tuple[str, ...]:
    out: list[str] = []
    for task in (TASK_DETECTION, TASK_SEGMENTATION):
        for col in metric_agg_columns(task):
            if col not in out:
                out.append(col)
    return tuple(out)
