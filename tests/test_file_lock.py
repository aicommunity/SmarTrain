"""Tests for cross-process file locking."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from smartrain.core.runtime import file_lock
from smartrain.core.runtime.file_lock import locked_file, smb_safe_locked_file


def test_locked_file_allows_sequential_writes(tmp_path: Path) -> None:
    target = tmp_path / "datasets_info.json"
    with locked_file(target):
        target.write_text('{"a": 1}', encoding="utf-8")
    with locked_file(target):
        target.write_text('{"a": 2}', encoding="utf-8")
    assert target.read_text(encoding="utf-8") == '{"a": 2}'


def test_locked_file_serializes_concurrent_access(tmp_path: Path) -> None:
    target = tmp_path / "queue.txt"
    order: list[str] = []
    barrier = threading.Barrier(3)

    def worker(tag: str) -> None:
        barrier.wait(timeout=5)
        with locked_file(target):
            order.append(f"{tag}-enter")
            time.sleep(0.02)
            order.append(f"{tag}-leave")

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert len(order) == 6
    for i in range(0, 6, 2):
        assert order[i].endswith("-enter")
        assert order[i + 1].endswith("-leave")
        assert order[i].split("-")[0] == order[i + 1].split("-")[0]


def test_fcntl_locked_file_keeps_same_inode_across_holders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fcntl path must not unlink after LOCK_UN (avoids waiter/new-inode race)."""
    if sys.platform == "win32":
        pytest.skip("fcntl lock path is not used on Windows")
    monkeypatch.setenv("SMART_TRAIN_SMB_LOCKS", "0")
    target = tmp_path / "index.json"
    lock_path = target.with_suffix(target.suffix + ".lock")

    with locked_file(target):
        assert lock_path.exists()
        first_inode = lock_path.stat().st_ino

    assert lock_path.exists(), "fcntl path must keep the lock file after unlock"
    assert lock_path.stat().st_ino == first_inode

    with locked_file(target):
        assert lock_path.stat().st_ino == first_inode
        # Same path must reopen the existing inode (O_CREAT|O_RDWR, not O_EXCL).
        assert os.stat(lock_path).st_ino == first_inode


def test_smb_safe_lock_serializes_and_only_holder_unlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMART_TRAIN_SMB_LOCKS", "1")
    target = tmp_path / "artifact.json"
    order: list[str] = []
    barrier = threading.Barrier(2)

    def worker(tag: str) -> None:
        barrier.wait(timeout=5)
        with locked_file(target):
            order.append(f"{tag}-enter")
            time.sleep(0.05)
            order.append(f"{tag}-leave")

    threads = [
        threading.Thread(target=worker, args=("a",)),
        threading.Thread(target=worker, args=("b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive()
    assert order[0].endswith("-enter")
    assert order[1].endswith("-leave")
    assert order[2].endswith("-enter")
    assert order[3].endswith("-leave")
    lock_path = target.with_suffix(target.suffix + ".lock")
    assert not lock_path.exists()


def test_smb_safe_timeout_does_not_unlink_foreign_lock(tmp_path: Path) -> None:
    target = tmp_path / "held.json"
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.write_text("held", encoding="utf-8")
    with pytest.raises(TimeoutError, match="Timed out waiting for lock"):
        with smb_safe_locked_file(target, timeout_sec=0.15, poll_sec=0.02):
            pass
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == "held"


def test_smb_safe_real_permission_error_is_not_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "denied.json"
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(file_lock.os, "open", boom)
    with pytest.raises(PermissionError):
        with smb_safe_locked_file(target, timeout_sec=1.0, poll_sec=0.01):
            pass
    assert calls["n"] == 1


def test_smb_safe_windows_sharing_violation_spins_then_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "shared.json"
    calls = {"n": 0}

    def sharing(*_args, **_kwargs):
        calls["n"] += 1
        err = PermissionError(13, "Permission denied")
        err.winerror = 32  # type: ignore[attr-defined]
        raise err

    monkeypatch.setattr(file_lock.os, "open", sharing)
    with pytest.raises(TimeoutError, match="Timed out waiting for lock"):
        with smb_safe_locked_file(target, timeout_sec=0.12, poll_sec=0.02):
            pass
    assert calls["n"] >= 2
