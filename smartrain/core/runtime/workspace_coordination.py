"""Multi-user workspace coordination: peers, SMB-safe locks, tiered policies."""

from __future__ import annotations

import atexit
import errno
import getpass
import json
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from smartrain.core.runtime.path_portable import posix_relpath
from smartrain.core.runtime.workspace_paths import WorkspaceLayout, resolve_workspace_root

HEARTBEAT_INTERVAL_SEC = 30
STALE_HEARTBEAT_MULTIPLIER = 3
DEFAULT_CATALOG_LOCK_TIMEOUT_SEC = 30
DEFAULT_WAIT_FOR_SCAN_SEC = 300
AUTO_SCAN_LOCK_TIMEOUT_SEC = 2

LOCK_SCAN = "scan"
LOCK_CATALOG = "catalog"
LOCK_QUEUE = "queue"

_TIER1_COMMANDS = frozenset(
    {
        "scan",
        "merge",
        "fusion",
        "augment",
        "split",
        "balance",
        "filter",
        "prune",
        "orient",
        "rotate",
    }
)
_TIER0_COMMANDS = frozenset(
    {
        "inference",
        "hash",
        "stats",
        "vis",
        "plot",
        "info",
        "deploy",
        "quickstart",
        "sync",
    }
)
_TIER2_COMMANDS = frozenset({"queue", "queue-run"})
_TIER3_COMMANDS = frozenset({"train", "test"})

_COORD_FLAGS = frozenset(
    {
        "--no-peer-warn",
        "--no-auto-scan",
        "--force-resource-lock",
        "--wait-for-scan",
        "--catalog-lock-timeout",
    }
)

_active_session: WorkspaceSession | None = None


class ScanLockBusy(Exception):
    """Raised when scan.lock is held by another live holder."""

    def __init__(self, lease: dict[str, Any], lock_path: Path) -> None:
        self.lease = lease
        self.lock_path = lock_path
        super().__init__(self.holder_summary())

    def holder_summary(self) -> str:
        host = self.lease.get("hostname", "?")
        cmd = self.lease.get("command", "?")
        return f"scan in progress on {host} ({cmd})"


class WorkspaceLockBusy(Exception):
    """Raised when a workspace lock cannot be acquired within timeout."""

    def __init__(self, lock_name: str, lease: dict[str, Any] | None, lock_path: Path) -> None:
        self.lock_name = lock_name
        self.lease = lease or {}
        self.lock_path = lock_path
        host = self.lease.get("hostname", "?")
        cmd = self.lease.get("command", "?")
        super().__init__(f"{lock_name} lock held by {host} ({cmd})")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _hostname() -> str:
    return socket.gethostname()


def _sanitize_segment(value: str, *, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9._+-]+", "-", str(value).strip()).strip("-") or "x"
    return s[:max_len]


def peers_dir(layout: WorkspaceLayout) -> Path:
    return Path(layout.root) / "tmp" / "peers"


def locks_dir(layout: WorkspaceLayout) -> Path:
    return Path(layout.root) / "tmp" / "locks"


def lock_path_for(layout: WorkspaceLayout, name: str) -> Path:
    if name.startswith("run-"):
        return locks_dir(layout) / f"{_sanitize_segment(name)}.lock"
    return locks_dir(layout) / f"{_sanitize_segment(name)}.lock"


def _read_lease(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_lease(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".lock", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    pid_i = int(pid)
    if sys.platform == "win32":
        # os.kill(pid, 0) is unreliable on Windows for non-existent PIDs.
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid_i))
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid_i, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _looks_like_smartrain(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if sys.platform == "win32":
        return _process_alive(pid)
    cmdline_path = Path(f"/proc/{int(pid)}/cmdline")
    if not cmdline_path.is_file():
        return _process_alive(pid)
    try:
        raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").lower()
    except OSError:
        return _process_alive(pid)
    return "smartrain" in raw or "smart-train" in raw


def _heartbeat_fresh(lease: dict[str, Any], *, now: datetime | None = None) -> bool:
    ref = _parse_iso(str(lease.get("last_heartbeat") or lease.get("acquired_at") or ""))
    if ref is None:
        return False
    current = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    age = (current - ref).total_seconds()
    return age < HEARTBEAT_INTERVAL_SEC * STALE_HEARTBEAT_MULTIPLIER


def _is_stale_lease(lease: dict[str, Any], *, local_host: str | None = None) -> bool:
    if not lease:
        return True
    host = str(lease.get("hostname", ""))
    pid = lease.get("pid")
    if local_host and host == local_host:
        if _process_alive(pid) and _looks_like_smartrain(pid):
            return False
        return not _heartbeat_fresh(lease)
    # Remote holder: PID is not meaningful on this host — trust heartbeat only.
    return not _heartbeat_fresh(lease)


def iter_peer_and_lock_files(layout: WorkspaceLayout) -> list[Path]:
    paths: list[Path] = []
    for base in (peers_dir(layout), locks_dir(layout)):
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                paths.append(p)
    return paths


def reconcile_local_holders(layout: WorkspaceLayout) -> int:
    host = _hostname()
    removed = 0
    for path in iter_peer_and_lock_files(layout):
        lease = _read_lease(path)
        if not lease:
            continue
        if str(lease.get("hostname", "")) != host:
            continue
        pid = lease.get("pid")
        if _process_alive(pid):
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
            rel = posix_relpath(path, layout.root)
            print(
                f"[INFO] Removed stale coordination file from previous interrupted run: "
                f"{rel} (pid {pid} not running)",
                flush=True,
            )
        except OSError:
            pass
    _reconcile_previous_session_sidecar(layout, host)
    return removed


def _last_session_sidecar_path(layout: WorkspaceLayout, host: str) -> Path:
    return peers_dir(layout) / f".last-session-{_sanitize_segment(host)}.json"


def _reconcile_previous_session_sidecar(layout: WorkspaceLayout, host: str) -> None:
    sidecar = _last_session_sidecar_path(layout, host)
    if not sidecar.is_file():
        return
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    previous = str(data.get("previous") or "").strip()
    if not previous:
        return
    for path in iter_peer_and_lock_files(layout):
        lease = _read_lease(path)
        if str(lease.get("session_id", "")) != previous:
            continue
        if _process_alive(lease.get("pid")):
            continue
        try:
            path.unlink(missing_ok=True)
            rel = posix_relpath(path, layout.root)
            print(f"[INFO] Removed stale file from previous session {previous}: {rel}", flush=True)
        except OSError:
            pass


def _update_last_session_sidecar(layout: WorkspaceLayout, session_id: str) -> None:
    host = _hostname()
    sidecar = _last_session_sidecar_path(layout, host)
    previous = ""
    if sidecar.is_file():
        try:
            prev_data = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(prev_data, dict):
                previous = str(prev_data.get("current") or "")
        except (OSError, json.JSONDecodeError):
            previous = ""
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    _write_lease(sidecar, {"current": session_id, "previous": previous})


def sweep_stale_peers(layout: WorkspaceLayout) -> int:
    removed = 0
    for path in iter_peer_and_lock_files(layout):
        lease = _read_lease(path)
        if lease and not _is_stale_lease(lease):
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            pass
    return removed


def list_active_peers(layout: WorkspaceLayout) -> list[dict[str, Any]]:
    sweep_stale_peers(layout)
    out: list[dict[str, Any]] = []
    peers_root = peers_dir(layout)
    if not peers_root.is_dir():
        return out
    for path in sorted(peers_root.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        lease = _read_lease(path)
        if lease and not _is_stale_lease(lease):
            lease = dict(lease)
            lease["_path"] = path.relative_to(layout.root).as_posix()
            out.append(lease)
    return out


def list_active_locks(layout: WorkspaceLayout) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    root = locks_dir(layout)
    if not root.is_dir():
        return out
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        lease = _read_lease(path)
        if not lease:
            continue
        if _is_stale_lease(lease):
            continue
        item = dict(lease)
        item["_path"] = path.relative_to(layout.root).as_posix()
        item["_lock_name"] = path.stem
        out.append(item)
    return out


def _try_create_lock_file(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False
        raise


def _remove_lock_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _peek_flag_value(argv: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix) :]
    return None


def _argv_has(argv: list[str], token: str) -> bool:
    prefix = f"{token}="
    return any(tok == token or tok.startswith(prefix) for tok in argv)


def _command_name_from_argv(argv: list[str]) -> str:
    for tok in argv:
        if tok.startswith("-"):
            continue
        return tok.lower()
    return ""


def _catalog_lock_timeout(argv: list[str]) -> float:
    raw = _peek_flag_value(argv, "--catalog-lock-timeout")
    if raw is None:
        return float(DEFAULT_CATALOG_LOCK_TIMEOUT_SEC)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_CATALOG_LOCK_TIMEOUT_SEC)


def _wait_for_scan_timeout(argv: list[str]) -> float:
    raw = _peek_flag_value(argv, "--wait-for-scan")
    if raw is None:
        return float(DEFAULT_WAIT_FOR_SCAN_SEC)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_WAIT_FOR_SCAN_SEC)


@dataclass
class CoordinationPolicy:
    tier: int
    catalog_lock: bool = False
    queue_lock: bool = False
    run_lock_name: str | None = None
    explicit_scan: bool = False
    catalog_timeout_sec: float = DEFAULT_CATALOG_LOCK_TIMEOUT_SEC
    wait_for_scan_sec: float = DEFAULT_WAIT_FOR_SCAN_SEC
    force_resource_lock: bool = False
    no_peer_warn: bool = False

    @contextmanager
    def locks(self, session: WorkspaceSession) -> Iterator[None]:
        held: list[str] = []
        try:
            if self.explicit_scan:
                session.acquire_lock(LOCK_SCAN, blocking=True, timeout_sec=self.wait_for_scan_sec)
                held.append(LOCK_SCAN)
            if self.queue_lock:
                session.acquire_lock(LOCK_QUEUE, blocking=True, timeout_sec=self.catalog_timeout_sec)
                held.append(LOCK_QUEUE)
            if self.catalog_lock:
                session.acquire_lock(LOCK_CATALOG, blocking=True, timeout_sec=self.catalog_timeout_sec)
                held.append(LOCK_CATALOG)
            if self.run_lock_name:
                try:
                    session.acquire_lock(
                        self.run_lock_name,
                        blocking=not self.force_resource_lock,
                        timeout_sec=2.0 if not self.force_resource_lock else self.catalog_timeout_sec,
                        allow_stale_takeover=self.force_resource_lock,
                    )
                    held.append(self.run_lock_name)
                except WorkspaceLockBusy as exc:
                    if self.force_resource_lock:
                        raise
                    print(f"[WARNING] {exc}; continuing without run lock", flush=True)
            yield
        finally:
            for name in reversed(held):
                session.release_lock(name)


def classify_command(argv: list[str], *, prog: str | None = None) -> CoordinationPolicy:
    cmd = prog or _command_name_from_argv(argv)
    cmd = cmd.lower().replace("_", "-")
    if cmd == "analyze":
        sub = ""
        for tok in argv:
            if not tok.startswith("-") and tok.lower() != "analyze":
                sub = tok.lower()
                break
        if sub in {"all", "inference-benchmark", "inference-plot", "test-metrics-plot"}:
            cmd = "analyze-heavy"
    policy = CoordinationPolicy(
        tier=0,
        catalog_timeout_sec=_catalog_lock_timeout(argv),
        wait_for_scan_sec=_wait_for_scan_timeout(argv),
        force_resource_lock=_argv_has(argv, "--force-resource-lock"),
        no_peer_warn=_argv_has(argv, "--no-peer-warn"),
    )
    if cmd in _TIER1_COMMANDS:
        policy.tier = 1
        policy.catalog_lock = cmd != "scan"
        if cmd == "scan":
            policy.explicit_scan = True
        return policy
    if cmd in _TIER2_COMMANDS or cmd.startswith("queue"):
        policy.tier = 2
        policy.queue_lock = True
        return policy
    if cmd in _TIER3_COMMANDS or cmd == "analyze-heavy":
        policy.tier = 3
        run = _peek_flag_value(argv, "--run") or _peek_flag_value(argv, "--run-dir")
        data = _peek_flag_value(argv, "--data")
        if run and data:
            policy.run_lock_name = f"run-{_sanitize_segment(data)}-{_sanitize_segment(run)}"
        return policy
    return policy


def try_resolve_layout_from_argv(argv: list[str]) -> WorkspaceLayout | None:
    try:
        root = resolve_workspace_root(_peek_flag_value(argv, "--workspace"))
    except ValueError:
        return None
    return WorkspaceLayout(root)


@contextmanager
def try_scan_lock(
    layout: WorkspaceLayout,
    *,
    session: WorkspaceSession | None = None,
    timeout_sec: float = 0.0,
    command: str = "auto-scan",
) -> Iterator[WorkspaceSession]:
    """Acquire scan.lock; non-blocking by default. Yields session holding the lock."""
    if session is None:
        with WorkspaceSession(layout, [command], register_peer=False, warn_peers=False) as local:
            local.acquire_lock(LOCK_SCAN, blocking=timeout_sec > 0, timeout_sec=timeout_sec)
            try:
                yield local
            finally:
                local.release_lock(LOCK_SCAN)
        return
    session.acquire_lock(LOCK_SCAN, blocking=timeout_sec > 0, timeout_sec=timeout_sec)
    try:
        yield session
    finally:
        session.release_lock(LOCK_SCAN)


@contextmanager
def catalog_write_lock(layout: WorkspaceLayout, *, session: WorkspaceSession | None = None) -> Iterator[None]:
    if session is not None:
        session.acquire_lock(LOCK_CATALOG, blocking=True, timeout_sec=DEFAULT_CATALOG_LOCK_TIMEOUT_SEC)
        try:
            yield
        finally:
            session.release_lock(LOCK_CATALOG)
        return
    with WorkspaceSession(layout, ["catalog-write"], register_peer=False, warn_peers=False) as local:
        local.acquire_lock(LOCK_CATALOG, blocking=True, timeout_sec=DEFAULT_CATALOG_LOCK_TIMEOUT_SEC)
        try:
            yield
        finally:
            local.release_lock(LOCK_CATALOG)


class WorkspaceSession:
    """Register peer presence, hold workspace locks, cleanup on exit or signal."""

    def __init__(
        self,
        layout: WorkspaceLayout,
        argv: list[str] | None = None,
        *,
        register_peer: bool = True,
        warn_peers: bool = True,
    ) -> None:
        self.layout = layout
        self.argv = list(argv or [])
        self.register_peer = register_peer
        self.warn_peers = warn_peers and not _argv_has(self.argv, "--no-peer-warn")
        self.started_at = _utc_now_iso()
        self.session_id = f"ws-{_sanitize_segment(_hostname())}-{os.getpid()}-{self.started_at.replace(':', '').replace('-', '')}"
        self.command = "smartrain " + " ".join(self.argv) if self.argv else "smartrain"
        self._peer_path: Path | None = None
        self._held_locks: list[str] = []
        self._lock_paths: dict[str, Path] = {}
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._releasing = False
        self._signals_installed = False
        self._previous_handlers: dict[int, Any] = {}

    def __enter__(self) -> WorkspaceSession:
        global _active_session
        reconcile_local_holders(self.layout)
        if self.warn_peers:
            self._warn_other_peers()
        if self.register_peer:
            self._register_peer()
            self._start_heartbeat()
        self._install_signal_handlers()
        _active_session = self
        atexit.register(self._release_all)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._release_all()
        global _active_session
        if _active_session is self:
            _active_session = None
        return False

    def _warn_other_peers(self) -> None:
        peers = list_active_peers(self.layout)
        others = [p for p in peers if str(p.get("session_id")) != self.session_id]
        if not others:
            return
        print(f"[WARNING] {len(others)} other active workspace session(s):", flush=True)
        for peer in others[:5]:
            host = peer.get("hostname", "?")
            user = peer.get("user", "?")
            cmd = peer.get("command", "?")
            print(f"  - {host} ({user}): {cmd}", flush=True)

    def _peer_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hostname": _hostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "command": self.command,
            "started_at": self.started_at,
            "last_heartbeat": _utc_now_iso(),
            "locks_held": list(self._held_locks),
        }

    def _register_peer(self) -> None:
        peers_root = peers_dir(self.layout)
        peers_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._peer_path = peers_root / f"{_sanitize_segment(_hostname())}_{os.getpid()}_{stamp}.json"
        _write_lease(self._peer_path, self._peer_payload())
        _update_last_session_sidecar(self.layout, self.session_id)

    def _touch_peer(self) -> None:
        if self._peer_path is None or not self._peer_path.is_file():
            return
        payload = self._peer_payload()
        _write_lease(self._peer_path, payload)

    def _start_heartbeat(self) -> None:
        def _loop() -> None:
            while not self._stop_event.wait(HEARTBEAT_INTERVAL_SEC):
                try:
                    self._touch_peer()
                    for name in list(self._held_locks):
                        path = self._lock_paths.get(name)
                        if path and path.is_file():
                            lease = _read_lease(path)
                            if str(lease.get("session_id")) == self.session_id:
                                lease["last_heartbeat"] = _utc_now_iso()
                                _write_lease(path, lease)
                except Exception:
                    pass

        self._heartbeat_thread = threading.Thread(target=_loop, name="smartrain-peer-heartbeat", daemon=True)
        self._heartbeat_thread.start()

    def _install_signal_handlers(self) -> None:
        if self._signals_installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass
        self._signals_installed = True

    def _on_signal(self, signum: int, frame) -> None:
        self._release_all()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    def _lease_payload(self, lock_name: str) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "hostname": _hostname(),
            "user": getpass.getuser(),
            "pid": os.getpid(),
            "command": self.command,
            "lock_name": lock_name,
            "acquired_at": _utc_now_iso(),
            "last_heartbeat": _utc_now_iso(),
        }

    def acquire_lock(
        self,
        name: str,
        *,
        blocking: bool = True,
        timeout_sec: float = DEFAULT_CATALOG_LOCK_TIMEOUT_SEC,
        allow_stale_takeover: bool = False,
    ) -> None:
        if name in self._held_locks:
            return
        path = lock_path_for(self.layout, name)
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            if _try_create_lock_file(path):
                _write_lease(path, self._lease_payload(name))
                self._held_locks.append(name)
                self._lock_paths[name] = path
                self._touch_peer()
                return
            lease = _read_lease(path)
            if allow_stale_takeover or _is_stale_lease(lease, local_host=_hostname()):
                _remove_lock_file(path)
                continue
            if not blocking or time.monotonic() >= deadline:
                if name == LOCK_SCAN:
                    raise ScanLockBusy(lease, path)
                raise WorkspaceLockBusy(name, lease, path)
            time.sleep(min(0.25, max(0.05, deadline - time.monotonic())))

    def release_lock(self, name: str) -> None:
        if name not in self._held_locks:
            return
        path = self._lock_paths.get(name) or lock_path_for(self.layout, name)
        lease = _read_lease(path)
        if not lease or str(lease.get("session_id")) == self.session_id:
            _remove_lock_file(path)
        if name in self._held_locks:
            self._held_locks.remove(name)
        self._lock_paths.pop(name, None)
        self._touch_peer()

    def _release_all(self) -> None:
        if self._releasing:
            return
        self._releasing = True
        try:
            atexit.unregister(self._release_all)
        except Exception:
            pass
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=1.0)
        for name in reversed(list(self._held_locks)):
            self.release_lock(name)
        if self._peer_path is not None:
            try:
                self._peer_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._peer_path = None
        if self._signals_installed:
            for sig, handler in self._previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
            self._signals_installed = False


def get_active_session() -> WorkspaceSession | None:
    return _active_session


__all__ = [
    "AUTO_SCAN_LOCK_TIMEOUT_SEC",
    "CoordinationPolicy",
    "DEFAULT_CATALOG_LOCK_TIMEOUT_SEC",
    "DEFAULT_WAIT_FOR_SCAN_SEC",
    "LOCK_CATALOG",
    "LOCK_QUEUE",
    "LOCK_SCAN",
    "ScanLockBusy",
    "WorkspaceLockBusy",
    "WorkspaceSession",
    "catalog_write_lock",
    "classify_command",
    "get_active_session",
    "list_active_locks",
    "list_active_peers",
    "lock_path_for",
    "reconcile_local_holders",
    "sweep_stale_peers",
    "try_resolve_layout_from_argv",
    "try_scan_lock",
]
