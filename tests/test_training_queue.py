"""Tests for training queue command parsing and subprocess execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from smartrain.workflows.queue import training_queue as tq


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ("# comment", None),
        ("smartrain train --data ds -y", ["smartrain", "train", "--data", "ds", "-y"]),
        ("/usr/bin/smartrain scan", ["/usr/bin/smartrain", "scan"]),
        ("python3 script.py --flag", ["python3", "script.py", "--flag"]),
        ("python train.py --epochs 1", ["python", "train.py", "--epochs", "1"]),
        ("train.py --epochs 1", ["python3", "train.py", "--epochs", "1"]),
    ],
)
def test_process_line_parses_to_argv(line: str, expected: list[str] | None) -> None:
    assert tq.process_line(line) == expected


def test_start_new_process_uses_shell_false() -> None:
    with patch("smartrain.workflows.queue.training_queue.subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 0
        code = tq.start_new_process(["echo", "ok"])
    assert code == 0
    popen.assert_called_once()
    kwargs = popen.call_args.kwargs
    assert kwargs["shell"] is False
    assert popen.call_args.args[0] == ["echo", "ok"]


def test_start_new_process_returns_exit_code() -> None:
    with patch("smartrain.workflows.queue.training_queue.subprocess.Popen") as popen:
        popen.return_value.wait.return_value = 7
        assert tq.start_new_process(["false"]) == 7


def test_get_queue_tasks_skips_comments_and_blanks(tmp_path) -> None:
    queue_file = tmp_path / "queue.txt"
    queue_file.write_text("# header\n\nsmartrain scan\n\n", encoding="utf-8")
    assert tq.get_queue_tasks(str(queue_file)) == ["smartrain scan"]


def test_save_statuses_writes_queue_order(tmp_path) -> None:
    status_file = tmp_path / "status.txt"
    tasks = ["smartrain scan", "smartrain train -y"]
    statuses = {"smartrain scan": "Done", "smartrain train -y": "Waiting to be completed"}
    tq.save_statuses(tasks, statuses, status_file=str(status_file))
    text = status_file.read_text(encoding="utf-8")
    assert "smartrain scan | Done" in text
    assert text.index("smartrain scan") < text.index("smartrain train -y")


def test_resolve_queue_status_paths_prefers_explicit_queue_file(tmp_path) -> None:
    queue_file = tmp_path / "custom_queue.txt"
    queue_path, status_path = tq.resolve_queue_status_paths(str(queue_file), None, None)
    assert queue_path == str(queue_file.resolve())
    assert Path(status_path).as_posix().endswith("tmp/status.txt")


def test_main_window_falls_back_without_gnome_terminal(capsys) -> None:
    with patch("smartrain.workflows.queue.training_queue.shutil.which", return_value=None):
        tq.main_window("/tmp/status.txt")
    out = capsys.readouterr().out
    assert "gnome-terminal not found" in out
    assert "/tmp/status.txt" in out


def test_update_status_ignores_invalid_index() -> None:
    tq.update_status(-1, "Done", ["smartrain scan"])
    tq.update_status(99, "Done", ["smartrain scan"])
