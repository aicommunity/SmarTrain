from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from smartrain.adapters.canonical.write.writer import WriteReport, write_canonical_snapshot
from smartrain.domain.canonical.models import CanonicalPayload

DualWriteMode = Literal["canonical_only", "dual_write_strict", "dual_write_best_effort"]


@dataclass(frozen=True)
class DualWriteReport:
    canonical_status: str
    legacy_status: str | None
    diff_summary: str | None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    rollback_hint: str | None = None
    write_report: WriteReport | None = None


def run_dual_write(
    *,
    payload: CanonicalPayload,
    target_root: str,
    mode: DualWriteMode,
    legacy_writer: Callable[[], None] | None = None,
) -> DualWriteReport:
    """
    Dual-write controller (PR 6.4): canonical path is always attempted first.

    - canonical_only: skip legacy.
    - dual_write_strict: legacy failure surfaces as legacy_status failed + rollback_hint.
    - dual_write_best_effort: legacy failure becomes warning; canonical remains authoritative.
    """
    wr = write_canonical_snapshot(payload, target_root)
    if mode == "canonical_only":
        return DualWriteReport(
            canonical_status="ok",
            legacy_status="skipped",
            diff_summary=None,
            warnings=(),
            rollback_hint=None,
            write_report=wr,
        )
    if legacy_writer is None:
        return DualWriteReport(
            canonical_status="ok",
            legacy_status="skipped",
            diff_summary=None,
            warnings=("legacy_writer not provided",),
            rollback_hint=None,
            write_report=wr,
        )
    try:
        legacy_writer()
        return DualWriteReport(
            canonical_status="ok",
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
                canonical_status="ok",
                legacy_status="failed",
                diff_summary=msg,
                warnings=(),
                rollback_hint="Canonical snapshot already written; repair legacy path or remove .smartrain/canonical if policy requires atomic dual-write.",
                write_report=wr,
            )
        return DualWriteReport(
            canonical_status="ok",
            legacy_status="failed",
            diff_summary=msg,
            warnings=(f"legacy write failed (best_effort): {msg}",),
            rollback_hint=None,
            write_report=wr,
        )
