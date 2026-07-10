"""Tests for cross-process file locking."""

from __future__ import annotations

import threading
from pathlib import Path

from smartrain.core.runtime.file_lock import locked_file


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

    def worker(tag: str) -> None:
        with locked_file(target):
            order.append(f"{tag}-enter")
            order.append(f"{tag}-leave")

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(order) == 6
    for i in range(0, 6, 2):
        assert order[i].endswith("-enter")
        assert order[i + 1].endswith("-leave")
