from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from smartrain.unified.io.write.writer import WriteReport, write_unified_snapshot
from smartrain.unified.domain.models import UnifiedPayload

DualWriteMode = Literal["unified_only", "dual_write_strict", "dual_write_best_effort"]


def normalize_dual_write_mode(raw: str) -> DualWriteMode:
    key = str(raw or "unified_only").strip().lower()
    if key == "canonical_only":
        return "unified_only"
    if key in {"unified_only", "dual_write_strict", "dual_write_best_effort"}:
        return key  # type: ignore[return-value]
    return "unified_only"


@dataclass(frozen=True)
class DualWriteReport:
    unified_status: str
    legacy_status: str | None
    diff_summary: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    rollback_hint: str | None = None
    write_report: WriteReport | None = None


def run_dual_write(
    *,
    payload: UnifiedPayload,
    target_root: str,
    mode: DualWriteMode,
    legacy_writer: Callable[[], None] | None = None,
) -> DualWriteReport:
    """
    Dual-write controller (PR 6.4): unified path is always attempted first.

    - unified_only: skip legacy.
    - dual_write_strict: legacy failure surfaces as legacy_status failed + rollback_hint.
    - dual_write_best_effort: legacy failure becomes warning; unified remains authoritative.
    """
    mode = normalize_dual_write_mode(mode)
    wr = write_unified_snapshot(payload, target_root)
    if mode == "unified_only":
        return DualWriteReport(
            unified_status="ok",
            legacy_status="skipped",
            diff_summary=None,
            warnings=(),
            rollback_hint=None,
            write_report=wr,
        )
    if legacy_writer is None:
        return DualWriteReport(
            unified_status="ok",
            legacy_status="skipped",
            diff_summary=None,
            warnings=("legacy_writer not provided",),
            rollback_hint=None,
            write_report=wr,
        )
    try:
        legacy_writer()
        return DualWriteReport(
            unified_status="ok",
            legacy_status="ok",
            diff_summary=None,
            warnings=(),
            rollback_hint=None,
            write_report=wr,
        )
    except Exception as exc:
        msg = str(exc)
        if mode == "dual_write_strict":
            return DualWriteReport(
                unified_status="ok",
                legacy_status="failed",
                diff_summary=msg,
                warnings=(),
                rollback_hint="Unified snapshot already written; repair legacy path or remove .smartrain/unified if policy requires atomic dual-write.",
                write_report=wr,
            )
        return DualWriteReport(
            unified_status="ok",
            legacy_status="failed",
            diff_summary=msg,
            warnings=(f"legacy write failed (best_effort): {msg}",),
            rollback_hint=None,
            write_report=wr,
        )
