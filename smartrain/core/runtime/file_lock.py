"""Cross-process file locking helpers for workspace artifacts."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Windows ERROR_SHARING_VIOLATION: holder has the .lock open exclusively.
_WINERROR_SHARING_VIOLATION = 32


def use_smb_safe_locks() -> bool:
    """Use O_EXCL lock files when SMB coordination is requested or on Windows."""
    env = os.environ.get("SMART_TRAIN_SMB_LOCKS", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return sys.platform == "win32"


def _is_smb_lock_contention(exc: BaseException) -> bool:
    """Whether ``os.open(O_EXCL)`` failed because another holder owns the lock."""
    if isinstance(exc, FileExistsError):
        return True
    if isinstance(exc, PermissionError) and getattr(exc, "winerror", None) == _WINERROR_SHARING_VIOLATION:
        return True
    return False


@contextmanager
def smb_safe_locked_file(
    path: str | Path,
    *,
    timeout_sec: float = 30.0,
    poll_sec: float = 0.05,
) -> Iterator[None]:
    """Exclusive lock via atomic ``<path>.lock`` creation (SMB-safe).

    Contenders spin until the lock file can be created exclusively, then
    release by unlinking it. Suitable for local NTFS and SMB shares where
    ``fcntl`` is unavailable.
    """
    target = Path(path)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    deadline = time.monotonic() + max(0.0, timeout_sec)
    try:
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except OSError as exc:
                if not _is_smb_lock_contention(exc):
                    raise
                # FileExistsError: lock file exists (classic contention).
                # PermissionError winerror=32: Windows holder denies shared write.
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from None
                time.sleep(poll_sec)
        yield
    finally:
        if fd is not None:
            os.close(fd)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass


@contextmanager
def locked_file(path: str | Path) -> Iterator[None]:
    """Advisory exclusive lock via a sibling ``<path>.lock`` file."""
    if use_smb_safe_locks():
        with smb_safe_locked_file(path):
            yield
        return
    target = Path(path)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        import fcntl

        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fd is not None:
            try:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(fd)
        # Keep the lock file: exclusivity is flock(fd), not path presence.
        # Unlinking after LOCK_UN races with waiters that open a new inode.
