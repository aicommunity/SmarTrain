from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from smartrain.adapters.canonical.read.factory import ReadAdapterFactory
from smartrain.adapters.canonical.read.resolvers import infer_source_kind
from smartrain.adapters.canonical.write.writer import WriteReport, write_canonical_snapshot
from smartrain.domain.canonical.context import TaskContext
from smartrain.domain.canonical.models import CanonicalMetricsRef, CanonicalPayload, CanonicalPredictionRef
from smartrain.domain.canonical.types import TaskType
from smartrain.domain.canonical.validators import validate_payload
from smartrain.metrics_reader import (
    METRIC_AGG_COLUMNS,
    read_test_metrics_by_format,
    read_test_metrics_row,
)


@dataclass(frozen=True)
class CanonicalGatewayOptions:
    validate: bool = True


def load_target(ref: str, *, source_kind: str | None = None, options: CanonicalGatewayOptions | None = None) -> CanonicalPayload:
    opts = options or CanonicalGatewayOptions()
    adapter = ReadAdapterFactory().resolve(source_kind, ref)
    payload = adapter.read(ref, options={})
    if opts.validate:
        validate_payload(payload)
    return payload


def persist_canonical_snapshot(ref: str, *, source_kind: str | None = None) -> WriteReport:
    """Load canonical payload for ref and write snapshot + manifest under target root."""
    payload = load_target(ref, source_kind=source_kind, options=CanonicalGatewayOptions(validate=True))
    return write_canonical_snapshot(payload, ref)


def resolve_task_context(
    ref: str,
    *,
    source_kind: str | None = None,
    options: CanonicalGatewayOptions | None = None,
) -> TaskContext:
    """Derive task/backend/model identity for a run or model directory (PR 6.5)."""
    payload = load_target(ref, source_kind=source_kind, options=options)
    if not payload.models:
        raise ValueError(f"No canonical model for ref={ref!r}")
    m = payload.models[0]
    r = payload.runs[0] if payload.runs else None
    sk = (source_kind or "").strip().lower() or infer_source_kind(ref)
    ap = os.path.abspath(os.path.expanduser(str(ref).strip()))
    return TaskContext(
        source_kind=sk,
        source_ref=ap,
        task_type=m.task_type,
        backend_type=m.backend_type,
        model_format=m.format,
        model_id=m.model_id,
        run_id=r.run_id if r is not None else None,
        dataset_ref=r.dataset_ref if r is not None else None,
    )


def _pd_is_na(v: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except Exception:
        return False


def _collect_test_metrics(ref: str, *, task_type: TaskType, format_name: str | None) -> list[CanonicalMetricsRef]:
    by_fmt = read_test_metrics_by_format(ref, include_internal=True)
    out: list[CanonicalMetricsRef] = []
    for fmt, csv_path in sorted(by_fmt.items()):
        if format_name and fmt != format_name.strip().lower():
            continue
        if not csv_path or not os.path.isfile(csv_path):
            continue
        row = read_test_metrics_row(ref, fmt)
        if not row:
            continue
        primary: dict[str, Any] = {}
        for col in METRIC_AGG_COLUMNS:
            if col not in row:
                continue
            v = row.get(col)
            if v is None or _pd_is_na(v):
                continue
            primary[str(col)] = float(v) if isinstance(v, (int, float)) else v
        secondary: dict[str, Any] = {
            str(k): v for k, v in row.items() if str(k) not in primary and str(k).lower() != "class"
        }
        ns = f"{task_type}/test_{fmt}"
        out.append(
            CanonicalMetricsRef(
                namespace=ns,
                primary_metrics=primary,
                secondary_metrics=secondary,
                raw_path=os.path.abspath(csv_path),
                producer="smartrain.metrics_reader",
                task_type=task_type,
            )
        )
    return out


def load_metrics(
    ref: str,
    *,
    source_kind: str | None = None,
    format_name: str | None = None,
    options: CanonicalGatewayOptions | None = None,
) -> list[CanonicalMetricsRef]:
    """
    Load test metrics CSVs for a run or model root as canonical metric refs (PR 6.5).

    Uses the same discovery rules as ``metrics_reader.read_test_metrics_by_format``.
    """
    base = load_target(ref, source_kind=source_kind, options=options)
    if not base.models:
        return []
    task_type = getattr(base.models[0], "task_type", "detection")
    return _collect_test_metrics(ref, task_type=task_type, format_name=format_name)


def load_predictions(
    ref: str,
    *,
    source_kind: str | None = None,
    options: CanonicalGatewayOptions | None = None,
) -> list[CanonicalPredictionRef]:
    """
    Reserved for prediction artifact discovery (PR 6.5).

    Currently returns an empty list when no standardized prediction bundle is registered.
    """
    _ = load_target(ref, source_kind=source_kind, options=options)
    return []
