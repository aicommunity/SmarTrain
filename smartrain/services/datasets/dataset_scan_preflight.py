"""Quiet workspace scan preflight before dataset-catalog-dependent CLI commands."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from smartrain.core.runtime.workspace_coordination import (
    AUTO_SCAN_LOCK_TIMEOUT_SEC,
    ScanLockBusy,
    WorkspaceSession,
    get_active_session,
    try_resolve_layout_from_argv,
)
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root
from smartrain.services.datasets.datasets_json_former import main as scan_main

_LEGACY_FUSION_FLAGS = frozenset(
    {"--source-path", "--target-path", "--datasets-info-path"}
)


def _peek_flag_value(argv: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix) :]
    return None


def _argv_has_token(argv: list[str], token: str) -> bool:
    prefix = f"{token}="
    return any(tok == token or tok.startswith(prefix) for tok in argv)


def try_resolve_workspace_from_argv(argv: list[str]) -> str | None:
    """Return workspace root when resolvable from argv/env; otherwise None."""
    try:
        return resolve_workspace_root(_peek_flag_value(argv, "--workspace"))
    except ValueError:
        return None


def _is_legacy_fusion_argv(argv: list[str]) -> bool:
    return all(_argv_has_token(argv, flag) for flag in _LEGACY_FUSION_FLAGS)


def should_run_auto_scan(argv: list[str], *, auto_scan_disabled: bool = False) -> bool:
    if auto_scan_disabled:
        return False
    if any(tok in ("--help", "-h") for tok in argv):
        return False
    if _is_legacy_fusion_argv(argv):
        return False
    return try_resolve_workspace_from_argv(argv) is not None


@contextmanager
def _suppress_scan_stdout() -> Iterator[object]:
    class _ErrorOnlyStdout:
        __slots__ = ("_inner", "had_error")

        def __init__(self, inner: object) -> None:
            self._inner = inner
            self.had_error = False

        def write(self, s: str) -> int:
            if "[ERROR]" in s:
                self.had_error = True
                return self._inner.write(s)  # type: ignore[union-attr]
            return len(s)

        def flush(self) -> None:
            flush = getattr(self._inner, "flush", None)
            if callable(flush):
                flush()

        def isatty(self) -> bool:
            isatty = getattr(self._inner, "isatty", None)
            return bool(isatty()) if callable(isatty) else False

        def reconfigure(self, **kwargs: object) -> None:
            reconfigure = getattr(self._inner, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(**kwargs)

    wrapper = _ErrorOnlyStdout(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = wrapper  # type: ignore[assignment]
    try:
        yield wrapper
    finally:
        sys.stdout = old_stdout


def run_quiet_workspace_scan(*, workspace_root: str, session: WorkspaceSession | None = None) -> bool:
    """Run workspace scan with minimal logging. Returns True on success."""
    layout = WorkspaceLayout(workspace_root)
    active = session or get_active_session()
    try:
        if active is not None:
            active.acquire_lock("scan", blocking=False, timeout_sec=AUTO_SCAN_LOCK_TIMEOUT_SEC)
            try:
                return _run_scan_locked(workspace_root)
            finally:
                active.release_lock("scan")
        with WorkspaceSession(layout, ["auto-scan"], register_peer=False, warn_peers=False) as local:
            local.acquire_lock("scan", blocking=False, timeout_sec=AUTO_SCAN_LOCK_TIMEOUT_SEC)
            try:
                return _run_scan_locked(workspace_root)
            finally:
                local.release_lock("scan")
    except ScanLockBusy as exc:
        print(f"[INFO] Skipping auto-scan: {exc.holder_summary()}", flush=True)
        return True


def _run_scan_locked(workspace_root: str) -> bool:
    print("[INFO] Running dataset scan…", flush=True)
    with _suppress_scan_stdout() as stdout_filter:
        scan_main(["--workspace", workspace_root, "--auto-scan"])
    if stdout_filter.had_error:
        print("[ERROR] Dataset scan failed.", flush=True)
        return False
    print("[INFO] Dataset scan completed.", flush=True)
    return True


def maybe_run_auto_scan(argv: list[str], *, ensure_scan: bool, auto_scan_disabled: bool) -> None:
    """Run quiet scan when enabled and applicable; raise SystemExit(1) on failure."""
    if not ensure_scan or not should_run_auto_scan(argv, auto_scan_disabled=auto_scan_disabled):
        return
    workspace_root = try_resolve_workspace_from_argv(argv)
    if workspace_root is None:
        return
    session = get_active_session()
    if session is not None:
        if not run_quiet_workspace_scan(workspace_root=workspace_root, session=session):
            raise SystemExit(1)
        return
    layout = try_resolve_layout_from_argv(argv)
    if layout is None:
        return
    try:
        with WorkspaceSession(layout, ["auto-scan"], register_peer=False, warn_peers=False) as local:
            if not run_quiet_workspace_scan(workspace_root=workspace_root, session=local):
                raise SystemExit(1)
    except ScanLockBusy as exc:
        print(f"[INFO] Skipping auto-scan: {exc.holder_summary()}", flush=True)


__all__ = [
    "maybe_run_auto_scan",
    "run_quiet_workspace_scan",
    "should_run_auto_scan",
    "try_resolve_workspace_from_argv",
]
