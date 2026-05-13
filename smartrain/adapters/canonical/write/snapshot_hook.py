"""Shared opt-in canonical snapshot write after successful pipeline steps (PR 6.4 / G1)."""

from __future__ import annotations

import os
from collections.abc import Callable


def maybe_dual_write_canonical_snapshot(
    root_dir: str,
    *,
    status_ok: bool,
    legacy_writer: Callable[[], None] | None = None,
    warn_prefix: str = "[WARN]",
) -> None:
    """
    When ``SMARTTRAIN_CANONICAL_WRITE=1`` and ``status_ok``, load canonical payload for ``root_dir``
    and write snapshot via ``run_dual_write`` (mode from ``SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE``).

    ``legacy_writer`` is invoked only when dual-write mode is not ``canonical_only`` (same contract
    as ``model_test_service.persist_target_test_artifacts_state``).
    """
    if not status_ok:
        return
    if str(os.getenv("SMARTTRAIN_CANONICAL_WRITE", "")).strip() != "1":
        return
    try:
        from smartrain.adapters.canonical.read.resolvers import infer_source_kind
        from smartrain.adapters.canonical.write.dual_write import run_dual_write
        from smartrain.orchestrators.canonical_gateway import CanonicalGatewayOptions, load_target

        dual_mode = str(os.getenv("SMARTTRAIN_CANONICAL_DUAL_WRITE_MODE", "canonical_only")).strip().lower()
        if dual_mode not in {"canonical_only", "dual_write_strict", "dual_write_best_effort"}:
            dual_mode = "canonical_only"
        source_kind = infer_source_kind(root_dir)
        payload = load_target(root_dir, source_kind=source_kind, options=CanonicalGatewayOptions(validate=True))
        lw: Callable[[], None] | None = legacy_writer if dual_mode != "canonical_only" else None
        run_dual_write(
            payload=payload,
            target_root=root_dir,
            mode=dual_mode,  # type: ignore[arg-type]
            legacy_writer=lw,
        )
    except Exception as exc:
        print(f"{warn_prefix} canonical snapshot not written for {root_dir!r}: {exc}")
