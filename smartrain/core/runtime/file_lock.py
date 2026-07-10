"""Cross-process file locking helpers for workspace artifacts."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def locked_file(path: str | Path) -> Iterator[None]:
    """Advisory exclusive lock via a sibling ``<path>.lock`` file."""
    target = Path(path)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    try:
        if sys.platform == "win32":
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            import fcntl

            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        if fd is not None:
            try:
                if sys.platform != "win32":
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass
            os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
