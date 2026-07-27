"""Queue retry / backoff unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from smartrain.workflows.queue import training_queue as tq


def test_run_queue_max_retries_two_gives_three_launches(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "queue.txt"
    status = tmp_path / "tmp" / "status.txt"
    queue.write_text("python3 -c 'raise SystemExit(1)'\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_start(cmd, cwd=None):
        calls.append(list(cmd))
        return 1

    idle = {"n": 0}

    def fake_sleep(sec):
        if float(sec) >= 5:
            idle["n"] += 1
            if idle["n"] >= 1 and len(calls) >= 3:
                raise KeyboardInterrupt
        return None

    monkeypatch.setattr(tq, "start_new_process", fake_start)
    monkeypatch.setattr(tq.time, "sleep", fake_sleep)
    monkeypatch.setattr(tq, "main_window", lambda *_a, **_k: None)

    with pytest.raises(KeyboardInterrupt):
        tq.run_queue(
            no_terminal=True,
            queue_path=str(queue),
            status_file=str(status),
            max_retries=2,
            retry_backoff_sec=0.0,
            retry_exit_codes={1},
        )
    assert len(calls) == 3


def test_run_queue_max_retries_zero_legacy(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "queue.txt"
    status = tmp_path / "tmp" / "status.txt"
    queue.write_text("python3 -c 'raise SystemExit(1)'\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_start(cmd, cwd=None):
        calls.append(list(cmd))
        return 1

    idle = {"n": 0}

    def fake_sleep(sec):
        if float(sec) >= 5:
            idle["n"] += 1
            if idle["n"] >= 1:
                raise KeyboardInterrupt
        return None

    monkeypatch.setattr(tq, "start_new_process", fake_start)
    monkeypatch.setattr(tq.time, "sleep", fake_sleep)
    monkeypatch.setattr(tq, "main_window", lambda *_a, **_k: None)

    with pytest.raises(KeyboardInterrupt):
        tq.run_queue(
            no_terminal=True,
            queue_path=str(queue),
            status_file=str(status),
            max_retries=0,
            retry_backoff_sec=0.0,
        )
    assert len(calls) == 1
