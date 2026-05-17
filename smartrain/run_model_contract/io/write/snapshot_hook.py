"""Shared opt-in unified snapshot write after successful pipeline steps (PR 6.4 / G1)."""

from __future__ import annotations

from collections.abc import Callable

from smartrain.unified.env import unified_dual_write_mode, unified_write_enabled


def maybe_dual_write_unified_snapshot(
    root_dir: str,
    *,
    status_ok: bool,
    legacy_writer: Callable[[], None] | None = None,
    warn_prefix: str = "[WARN]",
) -> None:
    """
    When unified write is enabled and ``status_ok``, load unified payload for ``root_dir``
    and write snapshot via ``run_dual_write`` (mode from env).

    ``legacy_writer`` is invoked only when dual-write mode is not ``unified_only``.
    """
    if not status_ok or not unified_write_enabled():
        return
    try:
        from smartrain.unified.io.read.resolvers import infer_source_kind
        from smartrain.unified.io.write.dual_write import run_dual_write
        from smartrain.unified.gateway import UnifiedGatewayOptions, load_target

        dual_mode = unified_dual_write_mode()
        source_kind = infer_source_kind(root_dir)
        payload = load_target(root_dir, source_kind=source_kind, options=UnifiedGatewayOptions(validate=True))
        lw: Callable[[], None] | None = legacy_writer if dual_mode != "unified_only" else None
        run_dual_write(
            payload=payload,
            target_root=root_dir,
            mode=dual_mode,
            legacy_writer=lw,
        )
    except Exception as exc:
        print(f"{warn_prefix} unified snapshot not written for {root_dir!r}: {exc}")
