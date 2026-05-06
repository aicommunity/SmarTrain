from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

from smartrain.adapters.canonical.read.factory import ReadAdapterFactory
from smartrain.adapters.canonical.read.resolvers import infer_source_kind
from smartrain.adapters.canonical.write.writer import WriteReport, write_canonical_snapshot
from smartrain.domain.canonical.context import TaskContext
from smartrain.domain.canonical.models import CanonicalMetricsRef, CanonicalPayload, CanonicalPredictionRef
from smartrain.domain.canonical.types import TaskType
from smartrain.domain.canonical.validators import validate_payload
from smartrain.workflows.analyze.metrics_reader import METRIC_AGG_COLUMNS, read_metrics_by_format_for_split
from smartrain.tasks.context import TaskExecutionContext
from smartrain.tasks.metrics import resolve_task_metrics_adapter


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


def _read_metrics_row_from_csv(metrics_path: str) -> dict[str, Any]:
    """
    Read a "canonical test metrics row" from a metrics CSV.

    This intentionally matches the behavior of ``metrics_reader.read_test_metrics_row``
    so that downstream consumers (e.g. format-compare) can use primary/secondary
    metrics consistently.
    """
    import pandas as pd

    if not metrics_path:
        return {}
    if not os.path.isfile(metrics_path):
        return {}

    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return {}
    if len(df) == 0:
        return {}
    df.columns = [str(c).strip() for c in df.columns]

    # Prefer explicit aggregate row if present.
    if "Class" in df.columns:
        cls = df["Class"].astype(str).str.strip().str.lower()
        all_mask = cls.eq("all")
        if bool(all_mask.any()):
            return df.loc[all_mask].iloc[0].to_dict()

    # If metrics are per-class without an "all" row, build macro-average.
    if "Class" in df.columns and len(df) > 1:
        out: dict[str, Any] = {}
        for col in METRIC_AGG_COLUMNS:
            if col in df.columns:
                out[col] = pd.to_numeric(df[col], errors="coerce").mean()
        if out:
            out["Class"] = "all"
            return out

    return df.iloc[0].to_dict()


def _collect_test_metrics(
    ref: str,
    *,
    task_type: TaskType,
    format_name: str | None,
    split: str,
) -> list[CanonicalMetricsRef]:
    by_fmt = read_metrics_by_format_for_split(ref, split, include_internal=True)
    out: list[CanonicalMetricsRef] = []
    task_metrics_adapter = resolve_task_metrics_adapter(str(task_type))
    for fmt, csv_path in sorted(by_fmt.items()):
        if format_name and fmt != format_name.strip().lower():
            continue
        if not csv_path or not os.path.isfile(csv_path):
            continue
        row = _read_metrics_row_from_csv(csv_path)
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
        normalized_task_metrics = task_metrics_adapter.normalize({str(k): v for k, v in row.items()})
        ns = TaskExecutionContext(task_type=str(task_type), stage=split, split=split).metrics_namespace(format_name=fmt)
        if normalized_task_metrics:
            for k, v in normalized_task_metrics.items():
                primary[k] = v
                secondary.pop(k, None)
        out.append(
            CanonicalMetricsRef(
                namespace=ns,
                primary_metrics=primary,
                secondary_metrics=secondary,
                raw_path=os.path.abspath(csv_path),
                producer="smartrain.workflows.analyze.metrics_reader",
                task_type=task_type,
            )
        )
    return out


def load_metrics(
    ref: str,
    *,
    source_kind: str | None = None,
    split: str = "test",
    format_name: str | None = None,
    options: CanonicalGatewayOptions | None = None,
) -> list[CanonicalMetricsRef]:
    """
    Load test metrics CSVs for a run or model root as canonical metric refs (PR 6.5).

    Uses the same discovery rules as ``metrics_reader.read_metrics_by_format_for_split``.
    """
    base = load_target(ref, source_kind=source_kind, options=options)
    if not base.models:
        return []
    task_type = getattr(base.models[0], "task_type", "detection")
    return _collect_test_metrics(ref, task_type=task_type, format_name=format_name, split=split)


def load_predictions(
    ref: str,
    *,
    source_kind: str | None = None,
    format_name: str | None = None,
    split: str | None = None,
    options: CanonicalGatewayOptions | None = None,
) -> list[CanonicalPredictionRef]:
    """
    Load available prediction artifacts as canonical prediction refs (PR 6.5).

    Discovery is file-based and intentionally conservative until a strict
    prediction bundle contract is finalized.
    """
    base = load_target(ref, source_kind=source_kind, options=options)
    if not base.models:
        return []
    task_type = getattr(base.models[0], "task_type", "detection")
    schema_version = base.schema_version
    root = os.path.abspath(os.path.expanduser(str(ref).strip()))
    patterns = [
        "**/deep_diagnostics/debug_test.jsonl",
        "**/deep_diagnostics/debug_val.jsonl",
        "**/predictions.jsonl",
        "**/predictions.json",
        "**/*pred*.jsonl",
        "**/*pred*.json",
    ]
    split_filter = str(split or "").strip().lower()
    fmt_filter = str(format_name or "").strip().lower()
    out: list[CanonicalPredictionRef] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in sorted(glob(os.path.join(root, pat), recursive=True)):
            ap = os.path.abspath(p)
            if ap in seen or not os.path.isfile(ap):
                continue
            name_l = os.path.basename(ap).lower()
            if split_filter:
                if split_filter == "test" and "test" not in name_l:
                    continue
                if split_filter == "val" and "val" not in name_l:
                    continue
            if fmt_filter:
                # Keep conservative format routing by path token only.
                if f"_{fmt_filter}." not in name_l and f"/{fmt_filter}/" not in ap.replace("\\", "/").lower():
                    continue
            cnt = 0
            try:
                if ap.endswith(".jsonl"):
                    with open(ap, "r", encoding="utf-8") as f:
                        cnt = sum(1 for ln in f if str(ln).strip())
                else:
                    import json

                    payload = json.loads(Path(ap).read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        cnt = len(payload)
            except Exception:
                cnt = 0
            out.append(
                CanonicalPredictionRef(
                    task_type=task_type,
                    items_path=ap,
                    schema_version=schema_version,
                    producer="smartrain.canonical_gateway",
                    count=int(max(0, cnt)),
                )
            )
            seen.add(ap)
    return out
